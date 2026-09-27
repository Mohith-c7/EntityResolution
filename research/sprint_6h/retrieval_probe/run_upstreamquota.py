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
from src.blocking.cheap_ranker import rank_features
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
    upstream_raw=upstream_metadata=upstream_rerank=0.
    if MODE=='upstreamquota':
        query=replace(reference,address='',address_tokens=frozenset(),address_digits=frozenset(),postcode_candidates=frozenset())
        current={c.candidate_entity_id for c,t,i in base}
        for index in indexes:
            step=time.perf_counter()
            terms=index.rare_terms('name',reference.name_tokens,index.postings_config.max_postings,index.postings_config.tokens_per_field)
            pieces=[index.postings('name',term)[0] for term in terms]
            for core in (index.aliases.variants(reference.core) if index.aliases else [reference.core]):
                if core:pieces.append(index.postings('core',core)[0])
            ids=np.unique(np.concatenate(pieces)) if pieces else np.empty(0,dtype=np.int32)
            upstream_raw+=time.perf_counter()-step
            step=time.perf_counter();eligible=[]
            for start in range(0,len(ids),500):
                chunk=[int(v) for v in ids[start:start+500]]
                for row in index.connection.execute('SELECT id,name,core,address,country,name_digits,address_digits,postcodes,folded FROM records WHERE address=\'\' AND rowid IN ('+','.join('?' for v in chunk)+')',chunk):
                    t=index.record_values(row)
                    if t.entity_id in current or not reference.core or not t.core:continue
                    if fuzz.token_set_ratio(reference.core,t.core)>=80:eligible.append(t)
            upstream_metadata+=time.perf_counter()-step
            step=time.perf_counter()
            index.prefetch_frequencies([query,*eligible])
            probabilities=index.ranking_model.predict(np.asarray([rank_features(query,t,index) for t in eligible],dtype=np.float32),num_threads=1) if eligible else []
            chosen=sorted(zip(eligible,probabilities),key=lambda p:(-p[1],p[0].entity_id))[:2]
            upstream_rerank+=time.perf_counter()-step
            for offset,(t,probability) in enumerate(chosen,1):
                candidate=Candidate(reference.entity_id,t.entity_id,index.source,
                    index.rerank(reference,t,('name_only_missing_address',)),
                    ('name_only_missing_address',),20+offset)
                added.append((candidate,t,index))
    extension_seconds=time.perf_counter()-ext_started
    profile_after={k:sum(i.profile[k] for i in indexes) for k in profile_before}
    TIMINGS.append({'source1_entity_id':reference.entity_id,'baseline_retrieval_seconds':base_sec,
        'extension_retrieval_seconds':extension_seconds,'baseline_count':len(base),'added_count':len(added),
        'upstream_raw_postings_seconds':upstream_raw,'upstream_empty_address_metadata_seconds':upstream_metadata,'upstream_rerank_seconds':upstream_rerank,
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
    for mode in ['baseline','upstreamquota']:
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
    for mode in ['baseline','upstreamquota']:
        bsc.retrieve=lambda r,ix:[(c,t,indexes[s]) for c,t,s in cached[mode][r.entity_id]]
        started=time.perf_counter();runner.score_chunk((number,rows,Path(out)/f'scored_{mode}'))
        stats[mode]=time.perf_counter()-started
    return stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);p.add_argument('--sample',type=Path,default=Path(__file__).with_name('sample.json'))
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('result_upstreamquota'));args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir();[(args.output/m).mkdir() for m in ['baseline','upstreamquota']]
    data=json.loads(args.sample.read_text());rows=data['references'];config=json.loads((ROOT/'models/frozen_v4_checkpoint_repair/frozen.json').read_text())
    protocol={'scope':'exposed_development_only','version':'upstreamquota','extension':'name/core capped token-postings union before fused/field caps; fetch only empty-address metadata; exclude baseline candidates and require core token_set>=80; apply unchanged name-only learned ranker then add at most2/source; preserve all baseline pairs','sample_entities':len(rows),'sample_ids_sha256':hashlib.sha256('\n'.join(sorted(r['entity_id'] for r in rows)).encode()).hexdigest(),'frozen_sha256':hashlib.sha256((ROOT/'models/frozen_v4_checkpoint_repair/frozen.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'workers':args.workers,'models_fitted':False,'heldout_labels_read':False,'new_truth_generated':False,'runtime_order':'paired baseline then upstream quota within each chunk; extension timing reuses warmed worker caches'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    started=time.perf_counter()
    tasks=[(n,rows[i:i+100],str(args.output.resolve())) for n,i in enumerate(range(0,len(rows),100))]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init,initargs=(config,)) as pool:
        for result in pool.map(worker,tasks):print(json.dumps(result),flush=True)
    wall=time.perf_counter()-started
    frames={m:pd.concat([pd.read_parquet(f) for f in sorted((args.output/m).glob('*.parquet'))],ignore_index=True) for m in ['baseline','upstreamquota']}
    sets={m:f.groupby('source1_entity_id',sort=False).candidate_entity_id.agg(set).to_dict() for m,f in frames.items()}
    truth={e:set(v) for e,v in data['truth'].items()}
    reports={}
    timing=[json.loads(f.read_text()) for f in sorted(args.output.glob('timings_*.json'))]
    for mode in ['baseline','upstreamquota']:
        cs=sets[mode];counts=np.array([len(cs.get(e,set())) for e in truth])
        ts=[r for t in timing for r in t[mode]['retrieval_timings']]
        retrieval_seconds=sum(r['baseline_retrieval_seconds']+r['extension_retrieval_seconds'] for r in ts)
        reports[mode]={'candidate_oracle':score_matches(truth,{e:truth[e]&cs.get(e,set()) for e in truth},data['countries']),'candidate_counts':{'mean':float(counts.mean()),'p95':float(np.quantile(counts,.95)),'max':int(counts.max())},'candidate_pairs':len(frames[mode]),'retrieval_worker_seconds':retrieval_seconds,'end_to_end_worker_seconds':sum(t[mode]['end_to_end_seconds'] for t in timing),'extension_worker_seconds':sum(r['extension_retrieval_seconds'] for r in ts),'mean_retrieval_ms':1000*retrieval_seconds/len(truth),**{k:sum(r[k] for r in ts) for k in ['baseline_raw_postings_seconds','baseline_postings_rerank_seconds','extension_raw_postings_seconds','extension_postings_rerank_seconds','upstream_raw_postings_seconds','upstream_empty_address_metadata_seconds','upstream_rerank_seconds']}}
    lost=[];new=[];added=[];replay_mismatch=[]
    for e in truth:
        b=sets['baseline'].get(e,set());v=sets['upstreamquota'].get(e,set())
        lost.extend((e,c) for c in sorted((b-v)&truth[e]));new.extend((e,c) for c in sorted((v-b)&truth[e]));added.extend((e,c) for c in sorted(v-b))
        if b!=set(data['baseline_candidates'][e]):replay_mismatch.append(e)
    changed_pairs=sum(len(sets['baseline'].get(e,set())-sets['upstreamquota'].get(e,set())) for e in truth)
    assert changed_pairs==0,'Extension removed baseline pairs'
    assert frames['upstreamquota'].drop_duplicates(['source1_entity_id','candidate_entity_id']).shape[0]==len(frames['upstreamquota'])
    pd.DataFrame(new,columns=['source1_entity_id','candidate_entity_id']).to_parquet(args.output/'new_true_links.parquet',index=False)
    pd.DataFrame(lost,columns=['source1_entity_id','candidate_entity_id']).to_parquet(args.output/'lost_true_links.parquet',index=False)
    pd.DataFrame(added,columns=['source1_entity_id','candidate_entity_id']).to_parquet(args.output/'added_pairs.parquet',index=False)
    report={**protocol,'status':'complete','wall_seconds':wall,'variants':reports,'new_true_links':len(new),'lost_true_links':len(lost),'lost_current_pairs':changed_pairs,'added_pairs':len(added),'baseline_replay_mismatch_references':len(replay_mismatch),'baseline_replay_mismatch_ids':replay_mismatch,'candidate_oracle_delta':reports['upstreamquota']['candidate_oracle']['macro_f05']-reports['baseline']['candidate_oracle']['macro_f05'],'recall_delta':reports['upstreamquota']['candidate_oracle']['micro_recall']-reports['baseline']['candidate_oracle']['micro_recall'],'all_reported_candidate_pairs_model_scored':False}
    report['target_met']=report['candidate_oracle_delta']>=.001 and reports['upstreamquota']['candidate_counts']['mean']<=44 and not replay_mismatch
    (args.output/'retrieval_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if report['target_met']:
        [(args.output/f'scored_{m}').mkdir() for m in ['baseline','upstreamquota']]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=init,initargs=(config,)) as pool:
            score_timing=list(pool.map(score_cached,tasks))
        scored={m:pd.concat([pd.read_parquet(f) for f in sorted((args.output/f'scored_{m}').glob('*.parquet'))],ignore_index=True) for m in ['baseline','upstreamquota']}
        for m in scored:
            scored_pairs=set(zip(scored[m].source1_entity_id,scored[m].candidate_entity_id))
            assert scored_pairs==set(zip(frames[m].source1_entity_id,frames[m].candidate_entity_id))
            reports[m]['pairs_scored']=len(scored[m]);reports[m]['scoring_worker_seconds']=sum(t[m] for t in score_timing)
        new_set=set(new)
        new_scores=scored['upstreamquota'][[(e,c) in new_set for e,c in zip(scored['upstreamquota'].source1_entity_id,scored['upstreamquota'].candidate_entity_id)]]
        report['new_true_links_above_current_threshold']=int((new_scores.probability>=config['threshold']).sum())
        new_scores.to_parquet(args.output/'new_true_model_scores.parquet',index=False)
        report['all_reported_candidate_pairs_model_scored']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
