"""Predicted-sibling neural evidence on the original sealed candidate route.

Internal neural pair keys identify seed and target records. Competition keys
remain Source1/target IDs in edges.parquet; seed correctness is never consulted.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/sprint_6h'),str(ROOT/'scripts'),str(ROOT/'research/final_2h')]
import neural_adapter as nn
from prepare_neural_inputs import serialize,sha
from evaluate_sprint import decide_control,paired_evaluate,predictions_for
B=Path('research/sprint_6h/sibling')
NAMES=[*nn.FEATURES,'peer_neural_max','peer_neural_mean','peer_neural_count']


def prepare(a):
    if a.output.exists():raise ValueError('New output required')
    frame,fm=nn.load_features(a.features)
    claims=pd.read_parquet(a.pairs)
    if sha(a.pairs)!=fm['input_pairs_sha256']:raise ValueError('Wrong source claims')
    route=nn.sibling.select_route(claims)
    edges=nn.sibling.route_edges(claims,route)
    if set(map(tuple,edges[nn.sibling.KEYS].to_numpy()))!=set(map(tuple,frame[nn.sibling.KEYS].to_numpy())):
        raise ValueError('Route differs from sealed neural inputs')
    pairs=edges[['peer_entity_id','candidate_entity_id']].drop_duplicates().sort_values(['peer_entity_id','candidate_entity_id'])
    if (pairs.peer_entity_id==pairs.candidate_entity_id).any():raise ValueError('Self seed')
    wanted=set(pairs.peer_entity_id)|set(pairs.candidate_entity_id);records={};sources={}
    for source in [2,3]:
        path=a.data_dir/f'train_source{source}.tsv';sources[f'S{source}']=sha(path)
        for batch in pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False,chunksize=100000):
            for row in batch[batch.entity_id.isin(wanted)].to_dict('records'):
                if row['entity_id'] in records:raise ValueError('Duplicate provided target')
                records[row['entity_id']]=serialize(row)
    if set(records)!=wanted:raise ValueError('Missing seed/target')
    a.output.mkdir(parents=True);edges.to_parquet(a.output/'edges.parquet',index=False)
    with (a.output/'pairs.jsonl').open('x') as stream:
        for peer,target in pairs.itertuples(index=False,name=None):
            stream.write(json.dumps({'source1_entity_id':peer,'candidate_entity_id':target,
                'text_left':records[peer],'text_right':records[target]},ensure_ascii=False)+'\n')
    marker={'status':'complete','rows':len(pairs),'labels_read':False,'input_sha256':sha(a.output/'pairs.jsonl'),
        'edges_sha256':sha(a.output/'edges.parquet'),'feature_manifest_sha256':sha(a.features/'manifest.json'),
        'pair_keys_note':'Internal left record is a predicted target seed, not an output Source1 identifier',
        'sources':sources,'code_sha256':sha(__file__)}
    (a.output/'manifest.json').write_text(json.dumps(marker,indent=2)+'\n')
    print(json.dumps(marker),flush=True)


def joined(features,inputs,scores):
    frame,fm=nn.load_features(features);im=json.loads((inputs/'manifest.json').read_text());sm=json.loads((scores/'manifest.json').read_text())
    if im['labels_read'] or sm['labels_read'] or sha(inputs/'manifest.json')!=sm['input_manifest_sha256'] or im['input_sha256']!=sm['input_sha256']:
        raise ValueError('Peer score provenance differs')
    if sha(inputs/'edges.parquet')!=im['edges_sha256'] or sha(scores/'scores.jsonl')!=sm['scores_sha256'] or sha(features/'manifest.json')!=im['feature_manifest_sha256']:
        raise ValueError('Peer bytes differ')
    values=pd.read_json(scores/'scores.jsonl',lines=True).rename(columns={'source1_entity_id':'peer_entity_id'})
    edges=pd.read_parquet(inputs/'edges.parquet')
    if values.duplicated(['peer_entity_id','candidate_entity_id']).any():raise ValueError('Repeated neural seed pair')
    if set(map(tuple,values[['peer_entity_id','candidate_entity_id']].to_numpy()))!=set(map(tuple,edges[['peer_entity_id','candidate_entity_id']].to_numpy())):
        raise ValueError('Peer score coverage incomplete')
    combined=edges.merge(values,on=['peer_entity_id','candidate_entity_id'],validate='many_to_one',how='left')
    agg=combined.groupby(nn.sibling.KEYS,sort=False).neural_probability.agg(['max','mean','size']).reset_index().rename(columns={'max':'peer_neural_max','mean':'peer_neural_mean','size':'peer_neural_count'})
    result=frame.merge(agg,on=nn.sibling.KEYS,validate='one_to_one',how='left',sort=False)
    if not frame[nn.sibling.KEYS].equals(result[nn.sibling.KEYS]) or not np.isfinite(result[NAMES]).all().all():raise ValueError('Pair order or peer values invalid')
    return result,fm


def add_fine(frame,fm,scores,input_manifest,head_manifest):
    sm=json.loads((scores/'manifest.json').read_text());im=json.loads(input_manifest.read_text())
    hm=json.loads(head_manifest.read_text())
    # Older sequence-score manifests attest the input bytes rather than the
    # manifest file. Verify that byte hash plus the independent route lineage.
    if sm['labels_read'] or im['labels_read'] or sha(scores/'scores.jsonl')!=sm['scores_sha256'] or sm.get('input_manifest_sha256',sha(input_manifest))!=sha(input_manifest):
        raise ValueError('Fine neural score seal differs')
    if im['feature_manifest_sha256']!=fm['reverse_feature_manifest_sha256'] or sm['input_sha256']!=im['input_sha256'] or sm['checkpoint_manifest_sha256']!=sha(head_manifest):
        raise ValueError('Fine neural route or checkpoint differs')
    if set(frame.source1_entity_id)&set(hm['training_owners']):raise ValueError('Fine neural fitted owner entered calibration/evaluation')
    values=pd.read_json(scores/'scores.jsonl',lines=True).rename(columns={'neural_probability':'neural_fine_probability'})
    if values.duplicated(nn.sibling.KEYS).any() or not values.neural_fine_probability.between(0,1).all():raise ValueError('Invalid fine neural values')
    if set(map(tuple,values[nn.sibling.KEYS].to_numpy()))!=set(map(tuple,frame[nn.sibling.KEYS].to_numpy())):raise ValueError('Fine score keys differ')
    result=frame.merge(values,on=nn.sibling.KEYS,how='left',validate='one_to_one',sort=False)
    if not frame[nn.sibling.KEYS].equals(result[nn.sibling.KEYS]):raise ValueError('Fine neural changed key order')
    return result


def fit(a):
    import lightgbm as lgb
    if a.output.exists():raise ValueError('New output required')
    names=list(nn.FEATURES)
    if a.command=='fit-fine':
        train,tm=nn.load_features(a.features);dev,dm=nn.load_features(a.dev_features)
    else:
        train,tm=joined(a.features,a.inputs,a.scores);dev,dm=joined(a.dev_features,a.dev_inputs,a.dev_scores)
        sm=json.loads((a.scores/'manifest.json').read_text());dm_score=json.loads((a.dev_scores/'manifest.json').read_text())
        if sm['head_manifest_sha256']!=dm_score['head_manifest_sha256'] or sm['head_manifest_sha256']!=sha(a.peer_head_manifest):
            raise ValueError('Peer neural heads differ')
        hm=json.loads(a.peer_head_manifest.read_text())
        if (set(train.source1_entity_id)|set(dev.source1_entity_id))&set(hm['training_owners']):raise ValueError('Peer head fitted development owner')
        names=list(NAMES)
    if tm['neural_head_manifest_sha256']!=dm['neural_head_manifest_sha256']:raise ValueError('Original neural heads differ')
    if a.fine_scores:
        train=add_fine(train,tm,a.fine_scores,a.fine_input_manifest,a.fine_head_manifest)
        dev=add_fine(dev,dm,a.dev_fine_scores,a.dev_fine_input_manifest,a.fine_head_manifest)
        names.append('neural_fine_probability')
    elif a.command=='fit-fine':raise ValueError('Fine scores required')
    truth=json.loads((B/'workbenches50k/residual_train/truth.json').read_text())
    et=json.loads((B/'workbenches50k/early_stop/truth.json').read_text())
    countries=json.loads((B/'workbenches50k/early_stop/countries.json').read_text())
    if set(train.source1_entity_id)&set(dev.source1_entity_id) or not set(train.source1_entity_id)<=set(truth) or not set(et)<=set(dm['reference_ids']):raise ValueError('Split provenance differs')
    early=dev[dev.source1_entity_id.isin(et)]
    def ds(f,t):
        y=np.array([int(c in t[e]) for e,c in f[nn.sibling.KEYS].itertuples(index=False,name=None)])
        return lgb.Dataset(f[names].to_numpy(dtype=np.float32),label=y,
            weight=1/f.groupby('source1_entity_id').source1_entity_id.transform('size').to_numpy(),
            init_score=nn.reverse.base_logit(f.probability),feature_name=names)
    params=dict(objective='binary',metric='binary_logloss',num_leaves=31,min_data_in_leaf=40,
        lambda_l2=1.,learning_rate=.05,max_bin=127,num_threads=8,deterministic=True,force_col_wise=True,
        seed=42,verbosity=-1,boost_from_average=False)
    model=lgb.train(params,ds(train,truth),num_boost_round=800,valid_sets=[ds(early,et)],callbacks=[lgb.early_stopping(40,verbose=False)])
    claims=pd.read_parquet(B/'learned30k/pairs.parquet')
    positions=pd.MultiIndex.from_frame(claims[nn.sibling.KEYS]).get_indexer(pd.MultiIndex.from_frame(dev[nn.sibling.KEYS]))
    if (positions<0).any() or not np.array_equal(claims.probability.to_numpy()[positions],dev.probability.to_numpy()):raise ValueError('Foreign feature anchors')
    changed=claims.copy();changed['probability_original']=claims.probability
    changed.loc[positions,'probability']=nn.reverse.corrected_probability(dev.probability,model.predict(dev[names].to_numpy(dtype=np.float32),raw_score=True,num_threads=8))
    config=json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text());config={k:config[k] for k in ['threshold','t_first','t_rest']}
    base,_=decide_control(claims,claims,config);trial,_=decide_control(changed,changed,config);mask=claims.source1_entity_id.isin(et).to_numpy()
    report=paired_evaluate(et,predictions_for(claims[mask],base[mask],et),predictions_for(changed[mask],trial[mask],et),countries,role='development')
    a.output.mkdir(parents=True);model.save_model(str(a.output/'adapter.txt'));changed.to_parquet(a.output/'pairs.parquet',index=False)
    report.update(status='early10k_only',features=names,parameters=params,best_iteration=model.best_iteration,
        model_sha256=sha(a.output/'adapter.txt'),fit_owners=sorted(truth),source_sha256=sha(__file__),
        selection_labels_read=False,fresh_labels_read=False,original_neural_head_sha256=tm['neural_head_manifest_sha256'],
        peer_head_sha256=sha(a.peer_head_manifest) if a.peer_head_manifest else None,
        fine_head_sha256=sha(a.fine_head_manifest) if a.fine_head_manifest else None)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['paired_macro_f05_delta','paired_delta_95pct_ci','links','acceptance']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','fit','fit-fine'])
    for name in ['features','pairs','data_dir','output','inputs','scores','dev_features','dev_inputs','dev_scores',
        'fine_scores','dev_fine_scores','fine_input_manifest','dev_fine_input_manifest','fine_head_manifest','peer_head_manifest']:
        p.add_argument('--'+name.replace('_','-'),type=Path)
    a=p.parse_args()
    if a.command=='prepare':prepare(a)
    else:fit(a)


if __name__=='__main__':main()
