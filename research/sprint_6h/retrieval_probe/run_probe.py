"""One fixed name-only extension, exposed dev sample, frozen model scores.

No production code/index/model edits. Baseline candidates are never removed.
"""
import argparse, hashlib, json, multiprocessing, os, pickle, sys, time
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'code/business_entity_resolution'))
import run_frozen_pipeline as runner
import build_scale_cache as bsc
from src.blocking.contracts import Candidate
from src.blocking.disk_index import normalize_record
from src.evaluation.metrics import score_matches

ORIGINAL=bsc.retrieve
MODE='baseline'
TIMINGS=[]

def retrieval(reference,indexes):
    profile_before={k:sum(i.profile[k] for i in indexes) for k in ['retrieve_seconds','rerank_seconds']}
    started=time.perf_counter()
    base=ORIGINAL(reference,indexes)
    base_sec=time.perf_counter()-started
    profile_base={k:sum(i.profile[k] for i in indexes) for k in profile_before}
    added=[]
    ext_started=time.perf_counter()
    if MODE=='name_only':
        query=replace(reference,address='',address_tokens=frozenset(),address_digits=frozenset(),postcode_candidates=frozenset())
        current={c.candidate_entity_id for c,t,i in base}
        for index in indexes:
            eligible=[]
            for c,t in index.query(query):
                if t.entity_id in current or t.address or not reference.core or not t.core:
                    continue
                if fuzz.token_set_ratio(reference.core,t.core)<80:
                    continue
                eligible.append((c,t))
            for offset,(c,t) in enumerate(eligible[:2],1):
                candidate=Candidate(reference.entity_id,t.entity_id,index.source,
                    index.rerank(reference,t,('name_only_missing_address',)),
                    ('name_only_missing_address',),20+offset)
                added.append((candidate,t,index))
    extension_seconds=time.perf_counter()-ext_started
    profile_after={k:sum(i.profile[k] for i in indexes) for k in profile_before}
    TIMINGS.append({'source1_entity_id':reference.entity_id,'baseline_retrieval_seconds':base_sec,
        'extension_retrieval_seconds':extension_seconds,'baseline_count':len(base),'added_count':len(added),
        'baseline_raw_postings_seconds':profile_base['retrieve_seconds']-profile_before['retrieve_seconds'],
        'baseline_postings_rerank_seconds':profile_base['rerank_seconds']-profile_before['rerank_seconds'],
        'extension_raw_postings_seconds':profile_after['retrieve_seconds']-profile_base['retrieve_seconds'],
        'extension_postings_rerank_seconds':profile_after['rerank_seconds']-profile_base['rerank_seconds']})
    return base+added

def init(config):
    runner.init(config,ROOT/config['aliases']['path'],str(ROOT/'models/index_train'),
        ROOT/'models/frozen_v4_checkpoint_repair/replay/universe_counts_train.sqlite')
    bsc.retrieve=retrieval

def worker(task):
    global MODE,TIMINGS
    number,rows,out=task
    stats={};cached={}
    for mode in ['baseline','name_only']:
        MODE=mode;TIMINGS=[]
        started=time.perf_counter()
        cached[mode]={};meta=[]
        for raw in rows:
            e=raw['entity_id'];pairs=retrieval(normalize_record(raw),runner.WORKER[0])
            cached[mode][e]=[(c,t,i.source) for c,t,i in pairs]
            meta.extend((e,c.candidate_entity_id) for c,t,i in pairs)
        pd.DataFrame(meta,columns=['source1_entity_id','candidate_entity_id']).to_parquet(Path(out)/mode/f'{number:07d}.parquet',index=False)
        stats[mode]={'end_to_end_seconds':time.perf_counter()-started,'retrieval_timings':TIMINGS}
    with (Path(out)/f'cached_{number:03d}.pickle').open('wb') as stream:pickle.dump(cached,stream)
    (Path(out)/f'timings_{number:03d}.json').write_text(json.dumps(stats))
    return {'chunk':number,'seconds':{m:stats[m]['end_to_end_seconds'] for m in stats}}

def score_cached(task):
    number,rows,out=task
    cached=pickle.loads((Path(out)/f'cached_{number:03d}.pickle').read_bytes())
    indexes={i.source:i for i in runner.WORKER[0]};stats={}
    for mode in ['baseline','name_only']:
        bsc.retrieve=lambda r,ix:[(c,t,indexes[s]) for c,t,s in cached[mode][r.entity_id]]
        started=time.perf_counter();runner.score_chunk((number,rows,Path(out)/f'scored_{mode}'))
        stats[mode]=time.perf_counter()-started
    return stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);p.add_argument('--sample',type=Path,default=Path(__file__).with_name('sample.json'))
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('result'));args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir();[(args.output/m).mkdir() for m in ['baseline','name_only']]
    data=json.loads(args.sample.read_text());rows=data['references'];config=json.loads((ROOT/'models/frozen_v4_checkpoint_repair/frozen.json').read_text())
    protocol={'scope':'exposed_development_only','extension':'name-only postings top20/source, candidate address empty, core token_set>=80, add at most2/source; preserve all baseline pairs','sample_entities':len(rows),'sample_ids_sha256':hashlib.sha256('\n'.join(sorted(r['entity_id'] for r in rows)).encode()).hexdigest(),'frozen_sha256':hashlib.sha256((ROOT/'models/frozen_v4_checkpoint_repair/frozen.json').read_bytes()).hexdigest(),'workers':args.workers,'models_fitted':False,'heldout_labels_read':False,'new_truth_generated':False,'runtime_order':'paired baseline then name-only within each chunk; extension timing reuses warmed worker caches'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    started=time.perf_counter()
    tasks=[(n,rows[i:i+100],str(args.output.resolve())) for n,i in enumerate(range(0,len(rows),100))]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init,initargs=(config,)) as pool:
        for result in pool.map(worker,tasks):print(json.dumps(result),flush=True)
    wall=time.perf_counter()-started
    frames={m:pd.concat([pd.read_parquet(f) for f in sorted((args.output/m).glob('*.parquet'))],ignore_index=True) for m in ['baseline','name_only']}
    sets={m:f.groupby('source1_entity_id',sort=False).candidate_entity_id.agg(set).to_dict() for m,f in frames.items()}
    truth={e:set(v) for e,v in data['truth'].items()}
    reports={}
    timing=[json.loads(f.read_text()) for f in sorted(args.output.glob('timings_*.json'))]
    for mode in ['baseline','name_only']:
        cs=sets[mode];counts=np.array([len(cs.get(e,set())) for e in truth])
        ts=[r for t in timing for r in t[mode]['retrieval_timings']]
        retrieval_seconds=sum(r['baseline_retrieval_seconds']+r['extension_retrieval_seconds'] for r in ts)
        reports[mode]={'candidate_oracle':score_matches(truth,{e:truth[e]&cs.get(e,set()) for e in truth},data['countries']),'candidate_counts':{'mean':float(counts.mean()),'p95':float(np.quantile(counts,.95)),'max':int(counts.max())},'candidate_pairs':len(frames[mode]),'retrieval_worker_seconds':retrieval_seconds,'end_to_end_worker_seconds':sum(t[mode]['end_to_end_seconds'] for t in timing),'extension_worker_seconds':sum(r['extension_retrieval_seconds'] for r in ts),'mean_retrieval_ms':1000*retrieval_seconds/len(truth),**{k:sum(r[k] for r in ts) for k in ['baseline_raw_postings_seconds','baseline_postings_rerank_seconds','extension_raw_postings_seconds','extension_postings_rerank_seconds']}}
    lost=[];new=[];added=[];replay_mismatch=[]
    for e in truth:
        b=sets['baseline'].get(e,set());v=sets['name_only'].get(e,set())
        lost.extend((e,c) for c in sorted((b-v)&truth[e]));new.extend((e,c) for c in sorted((v-b)&truth[e]));added.extend((e,c) for c in sorted(v-b))
        if b!=set(data['baseline_candidates'][e]):replay_mismatch.append(e)
    changed_pairs=sum(len(sets['baseline'].get(e,set())-sets['name_only'].get(e,set())) for e in truth)
    assert changed_pairs==0,'Extension removed baseline pairs'
    assert frames['name_only'].drop_duplicates(['source1_entity_id','candidate_entity_id']).shape[0]==len(frames['name_only'])
    pd.DataFrame(new,columns=['source1_entity_id','candidate_entity_id']).to_parquet(args.output/'new_true_links.parquet',index=False)
    pd.DataFrame(lost,columns=['source1_entity_id','candidate_entity_id']).to_parquet(args.output/'lost_true_links.parquet',index=False)
    pd.DataFrame(added,columns=['source1_entity_id','candidate_entity_id']).to_parquet(args.output/'added_pairs.parquet',index=False)
    report={**protocol,'status':'complete','wall_seconds':wall,'variants':reports,'new_true_links':len(new),'lost_true_links':len(lost),'lost_current_pairs':changed_pairs,'added_pairs':len(added),'baseline_replay_mismatch_references':len(replay_mismatch),'baseline_replay_mismatch_ids':replay_mismatch,'candidate_oracle_delta':reports['name_only']['candidate_oracle']['macro_f05']-reports['baseline']['candidate_oracle']['macro_f05'],'recall_delta':reports['name_only']['candidate_oracle']['micro_recall']-reports['baseline']['candidate_oracle']['micro_recall'],'all_reported_candidate_pairs_model_scored':False}
    report['target_met']=report['candidate_oracle_delta']>=.001 and reports['name_only']['candidate_counts']['mean']<=44 and not replay_mismatch
    (args.output/'retrieval_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if report['target_met']:
        [(args.output/f'scored_{m}').mkdir() for m in ['baseline','name_only']]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init,initargs=(config,)) as pool:
            score_timing=list(pool.map(score_cached,tasks))
        scored={m:pd.concat([pd.read_parquet(f) for f in sorted((args.output/f'scored_{m}').glob('*.parquet'))],ignore_index=True) for m in ['baseline','name_only']}
        for m in scored:
            scored_pairs=set(zip(scored[m].source1_entity_id,scored[m].candidate_entity_id))
            assert scored_pairs==set(zip(frames[m].source1_entity_id,frames[m].candidate_entity_id))
            reports[m]['pairs_scored']=len(scored[m]);reports[m]['scoring_worker_seconds']=sum(t[m] for t in score_timing)
        new_set=set(new)
        new_scores=scored['name_only'][[(e,c) in new_set for e,c in zip(scored['name_only'].source1_entity_id,scored['name_only'].candidate_entity_id)]]
        report['new_true_links_above_current_threshold']=int((new_scores.probability>=config['threshold']).sum())
        new_scores.to_parquet(args.output/'new_true_model_scores.parquet',index=False)
        report['all_reported_candidate_pairs_model_scored']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
