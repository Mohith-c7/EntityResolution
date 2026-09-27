"""Label-free full TEST direct empty route, actual reverse33 and supplied raw text.

Verified submission05 global empty owners; original forty top4 BEFORE excluding
old neural keys. This is a new direct route, never a fabricated sibling route.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research/sprint_6h'),str(ROOT/'research/sprint_6h/sibling')]
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import reverse_competition as reverse
import train_reverse_adapter as adapter
from prepare_neural_inputs import sha,serialize
KEYS=reverse.KEYS
VERSION='hybrid-test-empty-direct-v1-36'

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def select_extra(frame,old_keys):
    if frame.duplicated(KEYS).any() or frame.groupby(KEYS[0],sort=False).size().ne(40).any():
        raise ValueError('Every empty owner requires forty distinct original candidates')
    ranked=frame.sort_values([KEYS[0],'probability',KEYS[1]],ascending=[True,False,True],kind='stable')
    top=ranked.groupby(KEYS[0],sort=False).head(4)
    # Exclusion AFTER top4 is essential; never replace overlap with rank five.
    keep=[tuple(key) not in old_keys for key in top[KEYS].itertuples(index=False,name=None)]
    return top.loc[keep].sort_values(KEYS,kind='stable').reset_index(drop=True)

def worker(args):
    path,route_path,index_path,prefix,out=args
    pa.set_cpu_count(1)
    frame=pd.read_parquet(path);selected=pd.read_parquet(route_path)
    index=reverse.ReverseIndex(index_path,corpus='test')
    if index.meta['records']!=1732544:raise ValueError('Incomplete TEST source1 index')
    own=index.records(selected.source1_entity_id.unique())
    targets={};target_meta={}
    for source in [2,3]:
        target_path=Path(str(prefix)+f'_S{source}.sqlite').resolve()
        with sqlite3.connect(f'{target_path.as_uri()}?mode=ro',uri=True) as conn:
            meta=json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
            if not meta.get('complete'):raise ValueError('Incomplete supplied target index')
            target_meta[f'S{source}']=meta
            ids=selected.loc[selected.candidate_entity_id.str.startswith(f'S{source}-'),'candidate_entity_id'].unique()
            for begin in range(0,len(ids),500):
                batch=list(ids[begin:begin+500])
                for e,n,ad,c in conn.execute('SELECT id,name,address,country FROM records WHERE id IN ('+','.join('?' for _ in batch)+')',batch):
                    targets[e]=reverse.clean_record(dict(entity_id=e,business_name=n,business_address=ad,country=c))
    if set(own)!=set(selected.source1_entity_id) or set(targets)!=set(selected.candidate_entity_id):raise ValueError('Missing actual indexed record')
    rows=[]
    with (out/'diagnostics.jsonl').open('x') as stream:
        for e,c in selected[KEYS].itertuples(index=False,name=None):
            features,diag=reverse.pair_features(index,targets[c],own[e])
            if e in diag['rival_ids']:raise ValueError('Self rival')
            rows.append(dict(zip(KEYS,[e,c]),**features))
            stream.write(json.dumps(dict(zip(KEYS,[e,c]),**diag))+'\n')
    raw=pd.DataFrame(rows,columns=[*KEYS,*reverse.FEATURE_NAMES])
    raw[reverse.FEATURE_NAMES]=raw[reverse.FEATURE_NAMES].astype(np.float32)
    positions=pd.MultiIndex.from_frame(frame[KEYS]).get_indexer(pd.MultiIndex.from_frame(selected[KEYS]))
    if (positions<0).any():raise ValueError('Foreign direct keys')
    combined=adapter.join_reverse(frame,raw,set(positions))
    combined.to_parquet(out/'features.parquet',index=False)
    raw.to_parquet(out/'reverse33.parquet',index=False)
    marker={'status':'complete','rows':len(combined),'features_sha256':sha(out/'features.parquet'),
        'reverse33_sha256':sha(out/'reverse33.parquet'),'diagnostics_sha256':sha(out/'diagnostics.jsonl'),
        'part_pairs_sha256':sha(path),'part_route_sha256':sha(route_path),
        'index_manifest_sha256':sha(Path(str(index_path)+'.complete.json')),'index_metadata':index.index_manifest,
        'target_index_metadata':target_meta,'query_config':asdict(reverse.QueryConfig()),
        'feature_code_sha256':sha(reverse.__file__),'join_code_sha256':sha(adapter.__file__),'labels_read':False}
    index.close();write(out/'manifest.json',marker)
    return marker

def serialize_inputs(selected,data,out,protocol):
    records={};hashes={};owners=set(selected.source1_entity_id);targets=set(selected.candidate_entity_id)
    for source in [1,2,3]:
        path=data/f'test_source{source}.tsv';hashes[f'S{source}']=sha(path)
        wanted=owners if source==1 else targets
        for batch in pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False,chunksize=100000):
            for row in batch.loc[batch.entity_id.isin(wanted)].to_dict('records'):
                if row['entity_id'] in records:raise ValueError('Duplicate provided raw record')
                records[row['entity_id']]=serialize(row)
    if set(records)!=owners|targets:raise ValueError('Missing supplied raw text')
    out.mkdir()
    with (out/'pairs.jsonl').open('x') as stream:
        for e,c in selected[KEYS].itertuples(index=False,name=None):
            stream.write(json.dumps(dict(zip(KEYS,[e,c]),text_left=records[e],text_right=records[c]),ensure_ascii=False)+'\n')
    marker={'status':'complete','version':'hybrid-test-empty-direct-raw-v1','rows':len(selected),
        'labels_read':False,'audit_labels_read':False,'extension_labels_read':False,
        'role':'full_official_test_empty_extra','source_split':'test','input_sha256':sha(out/'pairs.jsonl'),
        'route_sha256':sha(out.parent/'route.parquet'),'pair_order_sha256':reverse.pair_order_hash(selected),
        'protocol_sha256':sha(protocol),'source_tsv_sha256':hashes,'code_sha256':sha(__file__)}
    write(out/'manifest.json',marker)
    print(json.dumps({'stage':'raw_inputs_complete','manifest':str(out/'manifest.json'),'rows':len(selected),'input_sha256':marker['input_sha256']}),flush=True)
    return marker

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('models/final_2h/test_full_base'))
    p.add_argument('--blue',type=Path,default=Path('output/submission_05'))
    p.add_argument('--old-features',type=Path,default=Path('models/final_2h/test_combined_neural05'))
    p.add_argument('--index',type=Path,default=Path('research/sprint_6h/reverse_competition/test_s1_v2.sqlite'))
    p.add_argument('--target-index-prefix',type=Path,default=Path('models/index_test'))
    p.add_argument('--data-dir',type=Path,default=Path('dataset/test'))
    p.add_argument('--output',type=Path,default=Path('models/final_2h/hybrid_test_rescue_v1'))
    p.add_argument('--workers',type=int,default=32)
    a=p.parse_args();start=time.monotonic();pa.set_cpu_count(1)
    if not 1<=a.workers<=32 or a.output.exists():raise ValueError('Require bounded workers and unused output')
    report=json.loads((a.blue/'submission_report.json').read_text())
    bm=json.loads((a.base/'manifest.json').read_text());fm=json.loads((a.old_features/'manifest.json').read_text())
    if report['status']!='complete' or report['validation']['strict']!='PASS' or report['validation']['official']!='PASS':raise ValueError('Blue05 not fully verified')
    checks=[(a.blue/'matching_results.tsv',report['files_sha256']['matching_results.tsv']),
        (a.blue/'frozen.json',report['candidate_frozen_sha256']),
        (a.old_features/'features.parquet',report['evidence_sha256']['features']),
        (a.old_features/'manifest.json',report['evidence_sha256']['features_manifest']),
        (a.base/'pairs.parquet',bm['pairs_sha256'])]
    for path,expected in checks:
        if sha(path)!=expected:raise ValueError(f'Gate changed: {path}')
    if bm['status']!='complete' or bm['split']!='test' or bm['pairs']!=69301760 or not bm['verified_current_pipeline']:raise ValueError('Original TEST graph not verified')
    frozen=json.loads((a.blue/'frozen.json').read_text())
    matching=pd.read_csv(a.blue/'matching_results.tsv',sep='\t',dtype=str,keep_default_na=False)
    if list(matching)!=['source1_entity_id','matched_entity_ids'] or matching.source1_entity_id.duplicated().any() or set(matching.source1_entity_id)!=set(bm['reference_ids']):raise ValueError('Global blue owner universe differs')
    empty=set(matching.loc[matching.matched_entity_ids.eq(''),'source1_entity_id'])
    old=pd.read_parquet(a.old_features/'features.parquet',columns=KEYS)
    if old.duplicated(KEYS).any() or len(old)!=801380:raise ValueError('Original neural keys differ')
    old_keys=set(old[KEYS].itertuples(index=False,name=None));del old,matching
    contexts=[];seen=[]
    for batch in pq.ParquetFile(a.base/'pairs.parquet').iter_batches(batch_size=200000,columns=[*KEYS,'first_stage','probability','candidate_order']):
        frame=batch.to_pandas()
        if len(frame)%40 or not np.array_equal(frame.candidate_order.to_numpy().reshape(-1,40),np.tile(np.arange(40),(len(frame)//40,1))):raise ValueError('Original contiguous40 context differs')
        ids=frame.source1_entity_id.to_numpy().reshape(-1,40)
        if not (ids==ids[:,0,None]).all():raise ValueError('Owner context split')
        seen.extend(ids[:,0]);contexts.append(frame.loc[frame.source1_entity_id.isin(empty)])
    if seen!=bm['reference_ids']:raise ValueError('Original graph owner order differs')
    context=pd.concat(contexts,ignore_index=True);del contexts
    if set(context.source1_entity_id)!=empty or len(context)!=40*len(empty):raise ValueError('Missing global empty owner context')
    selected=select_extra(context,old_keys)
    a.output.mkdir(parents=True)
    selected.to_parquet(a.output/'route.parquet',index=False)
    protocol={'status':'fixed_hybrid_label_free_test_preparation','route':'Verified global blue05 empty owners; ORIGINAL frozenp2 top4 targetID ties; THEN exclude original neural keys without rank5 replacement',
        'empty_owners':len(empty),'rows':len(selected),'top4_before_exclusion':4*len(empty),'overlap_removed':4*len(empty)-len(selected),
        'blue_matching_sha256':report['files_sha256']['matching_results.tsv'],'blue_report_sha256':sha(a.blue/'submission_report.json'),
        'blue_frozen_sha256':report['candidate_frozen_sha256'],'blue_decision_config':frozen['decision_config'],
        'original_base_manifest_sha256':sha(a.base/'manifest.json'),'original_base_pairs_sha256':bm['pairs_sha256'],
        'original_pair_order_sha256':bm['pair_order_sha256'],'original_neural_keys_sha256':report['evidence_sha256']['features'],
        'route_sha256':sha(a.output/'route.parquet'),'route_pair_order_sha256':reverse.pair_order_hash(selected),
        'input_source_sha256':sha(__file__),'labels_read':False,'audit_labels_read':False,'extension_labels_read':False,
        'release_requires_fresh_hybrid_audit_pass':True,'original05_unchanged':True}
    write(a.output/'protocol.json',protocol)
    print(json.dumps({'stage':'route_complete','empty_owners':len(empty),'rows':len(selected),'seconds':time.monotonic()-start}),flush=True)
    owners=list(context.source1_entity_id.unique());jobs=[]
    for number,ids in enumerate(np.array_split(np.asarray(owners),a.workers)):
        if not len(ids):continue
        sub=context.loc[context.source1_entity_id.isin(ids)].reset_index(drop=True)
        route=selected.loc[selected.source1_entity_id.isin(ids)].reset_index(drop=True)
        part=a.output/'parts'/str(number);part.mkdir(parents=True)
        sub.to_parquet(part/'pairs.parquet',index=False);route.to_parquet(part/'route.parquet',index=False)
        jobs.append((part/'pairs.parquet',part/'route.parquet',a.index,a.target_index_prefix,part))
    del context,sub,route
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(worker,j) for j in jobs]
        input_marker=serialize_inputs(selected,a.data_dir,a.output/'neural_inputs',a.output/'protocol.json')
        markers=[f.result() for f in futures]
    combined=pd.concat([pd.read_parquet(j[-1]/'features.parquet') for j in jobs],ignore_index=True).sort_values(KEYS,kind='stable').reset_index(drop=True)
    if not combined[KEYS].equals(selected[KEYS]) or not np.isfinite(combined[adapter.FEATURES].to_numpy()).all():raise ValueError('Feature union/order differs from raw input route')
    first=markers[0]
    for m in markers:
        for field in ['index_manifest_sha256','query_config','feature_code_sha256','join_code_sha256','target_index_metadata']:
            if m[field]!=first[field]:raise ValueError('Worker provenance differs')
    combined.to_parquet(a.output/'features.parquet',index=False)
    marker={'status':'complete','version':VERSION,'features':adapter.FEATURES,'rows':len(combined),'empty_owners':len(empty),
        'features_sha256':sha(a.output/'features.parquet'),'feature_pair_order_sha256':reverse.pair_order_hash(combined),
        'protocol_sha256':sha(a.output/'protocol.json'),'route_sha256':sha(a.output/'route.parquet'),
        'input_manifest_sha256':sha(a.output/'neural_inputs/manifest.json'),'input_sha256':input_marker['input_sha256'],
        'role':'full_official_test_empty_extra','split':'test','labels_read':False,'audit_labels_read':False,
        'index_manifest_sha256':first['index_manifest_sha256'],'index_metadata':first['index_metadata'],
        'query_config':first['query_config'],'target_index_metadata':first['target_index_metadata'],
        'source_tsv_sha256':input_marker['source_tsv_sha256'],'source_sha256':sha(__file__),
        'feature_code_sha256':first['feature_code_sha256'],'join_code_sha256':first['join_code_sha256'],
        'part_manifest_sha256':[sha(j[-1]/'manifest.json') for j in jobs],'workers':a.workers,'seconds':time.monotonic()-start}
    write(a.output/'manifest.json',marker)
    print(json.dumps({'stage':'complete','rows':len(combined),'seconds':time.monotonic()-start}),flush=True)

if __name__=='__main__':main()
