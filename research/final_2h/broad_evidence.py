"""Broader, seed-independent correction using supplied text and universe counts.

No change to retrieval or the incumbent's40-pair candidate set. Candidate routing
does not access truth and preserves all anchors outside the declared route.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research/sprint_6h/sibling'),str(ROOT/'code/business_entity_resolution')]
import train_sibling_matcher as sibling
import train_reverse_adapter as reverse
from src.preprocessing.normalize import normalize_name,normalize_address,name_core,accent_fold
from src.features.evidence import phonetic
from evaluate_sprint import decide_control,paired_evaluate,predictions_for

B=Path('research/sprint_6h/sibling')
EXTRA=['p1','p2','rank_p1','rank_p2','name_without_digits_sort','address_without_digits_sort',
 'address_without_digits_set','core_sort','core_set','core_exact','target_core_other_count',
 'reference_core_other_count','target_name_word_min_df','shared_name_word_min_df',
 'name_recovery_from_address','address_recovery_from_name','strong_seed_p2','strong_seed_count',
 'seed_name_sort_max','seed_address_sort_max','seed_address_nodigit_sort_max',
 'direct_address_char_ratio','direct_name_char_ratio','name_hard_word_disagreement']
NAMES=[*sibling.FEATURE_NAMES,*EXTRA]
TABLES={};REFS={};COUNTS={};WORD_COUNTS={}


def sha(path):return sibling.digest(path)


def route(frame):
    ordered=frame.sort_values(['source1_entity_id','first_stage','candidate_entity_id'],ascending=[True,False,True],kind='stable')
    rank=(ordered.groupby('source1_entity_id',sort=False).cumcount()+1).reindex(frame.index).to_numpy()
    # Seeds are contextual inputs, never a requirement for an entity's inclusion.
    mask=(frame.probability.to_numpy()<.995)|(rank<=4)
    return mask,rank,reverse.frozen_p2_rank(frame)


def clean(name,address,source):
    record=sibling.clean_target(name,address,source)
    record['core']=accent_fold(name_core(name))
    record['name_plain']=' '.join(t for t in record['name'].split() if not t.isdecimal())
    record['address_plain']=' '.join(t for t in record['address'].split() if not t.isdecimal())
    return record


def rows_job(rows):
    result=[]
    for e,c,country,p1,p2,r1,r2,seed_ids,seed_probabilities in rows:
        left=REFS[e];right=TABLES[c]
        direct=sibling.sibling_features(left,right)
        seeds=[TABLES[t] for t in seed_ids if t!=c]
        own_core=left['core'];target_core=right['core']
        same=int(own_core==target_core and bool(target_core))
        tc=COUNTS.get((country,target_core),0)-same if target_core else -1
        rc=COUNTS.get((country,own_core),0)-int(bool(own_core)) if own_core else -1
        nt=set(right['name'].split());common=set(left['name'].split())&nt
        tdf=min((WORD_COUNTS.get((country,t),0) for t in nt),default=0)
        shared_df=min((WORD_COUNTS.get((country,t),0) for t in common),default=0)
        support=lambda key: max((fuzz.token_sort_ratio(right[key],s[key])/100 for s in seeds if right[key] and s[key]),default=0.)
        hard_disagree=sum(len(t)>=4 and max((fuzz.ratio(t,u) for u in nt),default=0)<65 for t in set(left['name'].split()))
        values=[p1,p2,r1,r2,
            fuzz.token_sort_ratio(left['name_plain'],right['name_plain'])/100 if left['name_plain'] and right['name_plain'] else 0,
            fuzz.token_sort_ratio(left['address_plain'],right['address_plain'])/100 if left['address_plain'] and right['address_plain'] else 0,
            fuzz.token_set_ratio(left['address_plain'],right['address_plain'])/100 if left['address_plain'] and right['address_plain'] else 0,
            fuzz.token_sort_ratio(own_core,target_core)/100 if own_core and target_core else 0,
            fuzz.token_set_ratio(own_core,target_core)/100 if own_core and target_core else 0,same,
            math.log1p(max(0,tc)) if tc>=0 else -1,math.log1p(max(0,rc)) if rc>=0 else -1,
            math.log1p(tdf),math.log1p(shared_df),
            fuzz.token_set_ratio(left['name'],right['name']+' '+right['address'])/100 if left['name'] else 0,
            fuzz.token_set_ratio(left['address'],right['address']+' '+right['name'])/100 if left['address'] else 0,
            max(seed_probabilities,default=0),len(seeds),support('name'),support('address'),support('address_plain'),
            fuzz.ratio(left['address'],right['address'])/100 if left['address'] and right['address'] else 0,
            fuzz.ratio(left['name'],right['name'])/100 if left['name'] and right['name'] else 0,hard_disagree]
        result.append([e,c,*direct,*values])
    return result


def prepare(args):
    global TABLES,REFS,COUNTS,WORD_COUNTS
    fit=pd.read_parquet(B/'workbenches50k/residual_train/pairs.parquet')
    dev=pd.read_parquet(B/'learned30k/pairs.parquet')
    if set(fit.source1_entity_id)&set(dev.source1_entity_id):raise ValueError('Fit/development overlap')
    frames=[fit,dev];all_owners=set().union(*(set(f.source1_entity_id) for f in frames))
    source=args.source1_index
    conn=sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)
    build=json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
    if not build.get('complete') or build['records']!=2206821:raise ValueError('Wrong full training Source1 universe')
    COUNTS=Counter();WORD_COUNTS=Counter()
    for e,name,core,address,country in conn.execute('SELECT id,name,namecore,primary_address,country FROM records'):
        country=country.casefold();core=accent_fold(core)
        if core:COUNTS[country,core]+=1
        WORD_COUNTS.update((country,t) for t in set(name.split()))
        if e in all_owners:REFS[e]=clean(name,address,'S1')
    conn.close()
    if set(REFS)!=all_owners:raise ValueError('Missing reference')
    tasks=[];ranges=[];wanted=set()
    for f in frames:
        mask,r1,r2=route(f)
        seeds={}
        for owner,group in f.groupby('source1_entity_id',sort=False):
            strong=group[group.probability>=.9].sort_values(['probability','candidate_entity_id'],ascending=[False,True]).head(2)
            seeds[owner]=(strong.candidate_entity_id.tolist(),strong.probability.tolist())
            wanted.update(strong.candidate_entity_id)
        selected=np.flatnonzero(mask);wanted.update(f.candidate_entity_id.iloc[selected])
        jobs=[]
        for i in selected:
            e=f.source1_entity_id.iat[i];seed_ids,seed_p=seeds[e]
            jobs.append((e,f.candidate_entity_id.iat[i],str(f.country.iat[i]).casefold(),float(f.first_stage.iat[i]),float(f.probability.iat[i]),int(r1[i]),int(r2[i]),seed_ids,seed_p))
        ranges.append(len(jobs));tasks.extend(jobs)
    raw=sibling.load_targets(wanted,'models/index_train')
    TABLES={e:clean(r['name'],r['address'],r['source']) for e,r in raw.items()}
    chunks=[tasks[i:i+4000] for i in range(0,len(tasks),4000)]
    arrays=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number,values in enumerate(pool.map(rows_job,chunks)):
            arrays.extend(values)
            if number%10==0:print(json.dumps({'stage':'features','pairs':len(arrays),'seconds':time.monotonic()-start}),flush=True)
    data=pd.DataFrame(arrays,columns=[*sibling.KEYS,*NAMES])
    if data.duplicated(sibling.KEYS).any() or not np.isfinite(data[NAMES]).all().all():raise ValueError('Invalid features')
    args.output.mkdir(parents=True,exist_ok=False)
    data.iloc[:ranges[0]].to_parquet(args.output/'train.parquet',index=False)
    data.iloc[ranges[0]:].to_parquet(args.output/'development.parquet',index=False)
    (args.output/'manifest.json').write_text(json.dumps({'status':'complete','features':NAMES,'labels_read':False,
        'fit_reference_ids':sorted(set(fit.source1_entity_id)),'dev_reference_ids':sorted(set(dev.source1_entity_id)),
        'source1_index_manifest_sha256':sha(source.with_suffix(source.suffix+'.complete.json')),
        'fit_source_sha256':sha(B/'workbenches50k/residual_train/pairs.parquet'),'development_source_sha256':sha(B/'learned30k/pairs.parquet'),
        'train_sha256':sha(args.output/'train.parquet'),'development_sha256':sha(args.output/'development.parquet'),
        'code_sha256':sha(__file__),'route':'p2<.995 or first_stage_rank<=4; no seed requirement',
        'routed_pairs':ranges,'feature_seconds':time.monotonic()-start},indent=2)+'\n')


def fit(args):
    import lightgbm as lgb
    cache=json.loads((args.features/'manifest.json').read_text())
    if cache['code_sha256']!=sha(__file__) or cache['features']!=NAMES or cache['labels_read']:raise ValueError('Feature seal differs')
    for name in ['train','development']:
        if sha(args.features/(name+'.parquet'))!=cache[name+'_sha256']:raise ValueError('Cache changed')
    train=pd.read_parquet(args.features/'train.parquet');dev=pd.read_parquet(args.features/'development.parquet')
    truth=json.loads((B/'workbenches50k/residual_train/truth.json').read_text())
    et=json.loads((B/'workbenches50k/early_stop/truth.json').read_text())
    countries=json.loads((B/'workbenches50k/early_stop/countries.json').read_text())
    if len(truth)!=20000 or len(et)!=10000 or set(truth)&set(et) or set(truth)&set(cache['dev_reference_ids']):raise ValueError('Wrong fit/early identities')
    early=dev[dev.source1_entity_id.isin(et)]
    def dataset(f,t):
        y=np.array([int(c in t[e]) for e,c in f[sibling.KEYS].itertuples(index=False,name=None)])
        return lgb.Dataset(f[NAMES].to_numpy(dtype=np.float32),label=y,
            weight=1/f.groupby('source1_entity_id').source1_entity_id.transform('size').to_numpy(),
            init_score=reverse.base_logit(f.p2),feature_name=NAMES)
    params=dict(objective='binary',metric='binary_logloss',num_leaves=63,min_data_in_leaf=60,lambda_l2=3.,
        learning_rate=.05,max_bin=127,num_threads=args.workers,force_col_wise=True,deterministic=True,seed=42,
        verbosity=-1,boost_from_average=False)
    model=lgb.train(params,dataset(train,truth),num_boost_round=1600,valid_sets=[dataset(early,et)],
        callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(200)])
    args.output.mkdir(parents=True,exist_ok=False);model.save_model(str(args.output/'adapter.txt'))
    claims=pd.read_parquet(B/'learned30k/pairs.parquet')
    index=pd.MultiIndex.from_frame(claims[sibling.KEYS]).get_indexer(pd.MultiIndex.from_frame(dev[sibling.KEYS]))
    if (index<0).any() or not np.array_equal(claims.probability.to_numpy()[index],dev.p2.to_numpy()):raise ValueError('Base anchor/key changed')
    correction=model.predict(dev[NAMES].to_numpy(dtype=np.float32),raw_score=True,num_threads=args.workers)
    changed=claims.copy();changed['probability_original']=claims.probability
    changed.loc[index,'probability']=reverse.corrected_probability(dev.p2,correction)
    changed.to_parquet(args.output/'pairs.parquet',index=False)
    config=json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
    config={k:config[k] for k in ['threshold','t_first','t_rest']}
    base,_=decide_control(claims,claims,config);trial,_=decide_control(changed,changed,config)
    mask=claims.source1_entity_id.isin(et).to_numpy()
    bp=predictions_for(claims.loc[mask],base[mask],et);cp=predictions_for(changed.loc[mask],trial[mask],et)
    report=paired_evaluate(et,bp,cp,countries,role='development')
    report.update(status='early10k_only',parameters=params,features=NAMES,best_iteration=model.best_iteration,
        feature_manifest_sha256=sha(args.features/'manifest.json'),model_sha256=sha(args.output/'adapter.txt'),
        fit_owners=sorted(truth),early_owners=sorted(et),fresh_labels_read=False,selection_labels_read=False,
        created_at=datetime.now(timezone.utc).isoformat())
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['paired_macro_f05_delta','country_deltas','links','acceptance']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','fit'])
    p.add_argument('--source1-index',type=Path);p.add_argument('--features',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args()
    if not 1<=a.workers<=24 or a.output.exists():raise ValueError('Unused output and bounded workers required')
    (prepare if a.command=='prepare' else fit)(a)
