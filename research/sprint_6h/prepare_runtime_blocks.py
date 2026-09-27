"""Pin a full-test route and two representative blocks; read no test labels."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import heapq
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import pandas as pd
import pyarrow as pa
import build_gated_submission as reuse
import train_sibling_matcher as sibling
from stream_route import collect,select


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reuse_manifest','test_dir','output']:
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--references-per-block',type=int,default=10000)
    a=p.parse_args();pa.set_cpu_count(1);start=time.perf_counter()
    if a.output.exists():raise ValueError('Require unused runtime asset destination')
    m=json.loads(a.reuse_manifest.read_text())
    source=a.test_dir/'test_source1.tsv'
    if reuse.digest(source)!=m['test_files']['test_source1.tsv']['sha256']:raise ValueError('Source1 changed')
    rows=reuse.read_source1(source);ids=[r['entity_id'] for r in rows]
    if len(rows)!=m['entities'] or reuse.ids_digest(ids)!=m['input_ids_sha256']:raise ValueError('Source coverage changed')
    countries=Counter(r['country'].strip().casefold() for r in rows)
    if min(countries.values())<2:raise ValueError('Insufficient country strata for disjoint blocks')
    quotas={c:max(1,int(a.references_per_block*n/len(rows))) for c,n in countries.items()}
    while sum(quotas.values())<a.references_per_block:
        c=max(countries,key=lambda c:(a.references_per_block*countries[c]/len(rows)-quotas[c],c));quotas[c]+=1
    selected={c:heapq.nsmallest(2*quotas[c],(r['entity_id'] for r in rows if r['country'].strip().casefold()==c),
        key=lambda e:hashlib.sha256(('runtime-neural-v2:'+e).encode()).digest()) for c in countries}
    block_ids=[set().union(*(set(v[b*quotas[c]:(b+1)*quotas[c]]) for c,v in selected.items())) for b in [0,1]]
    if block_ids[0]&block_ids[1]:raise ValueError('Runtime blocks overlap')
    a.output.mkdir(parents=True);priorities={};edges=[];frames=[[],[]];cursor=0
    for number,evidence in enumerate(m['chunks']):
        if evidence['chunk']!=number:raise ValueError('Noncontiguous baseline chunks')
        marker=json.loads(reuse.resolve_evidence(evidence['marker'],ROOT).read_text())
        path=reuse.resolve_evidence(evidence['parquet'],ROOT);frame=pd.read_parquet(path)
        reuse.validate_pairs(frame)
        expected=ids[cursor:cursor+marker['entities']];cursor+=marker['entities']
        if marker['input_ids_sha256']!=reuse.ids_digest(expected) or len(frame)!=marker['pairs'] or set(frame.source1_entity_id)-set(expected):raise ValueError('Baseline chunk coverage changed')
        frame['candidate_order']=frame.groupby('source1_entity_id',sort=False).cumcount().to_numpy()
        current,part=collect(frame,ids)
        if set(current)&set(priorities):raise ValueError('Reference repeated between chunks')
        priorities.update(current);edges.append(part)
        for b in [0,1]:
            chosen=frame[frame.source1_entity_id.isin(block_ids[b])]
            if len(chosen):frames[b].append(chosen.copy())
        if (number+1)%1000==0:print(json.dumps({'stage':'global_proposals','chunks':number+1,'seconds':time.perf_counter()-start}),flush=True)
    if cursor!=len(ids):raise ValueError('Incomplete baseline graph')
    owners=set(select(priorities,ids));route=pd.concat(edges,ignore_index=True)
    route=route[route.source1_entity_id.isin(owners)].reset_index(drop=True)
    global_route=a.output/'global_route';global_route.mkdir();route.to_parquet(global_route/'route.parquet',index=False)
    reuse.write_json(global_route/'manifest.json',{'status':'complete','route':asdict(sibling.RouteConfig()),
        'global_references':len(ids),'global_pair_order_sha256':m['pair_order_sha256'],
        'route_sha256':reuse.digest(global_route/'route.parquet'),'frozen_sha256':m['baseline_frozen']['sha256'],
        'labels_read':False,'streaming_code_sha256':reuse.digest(ROOT/'research/sprint_6h/stream_route.py')})
    blocks=[]
    for b in [0,1]:
        destination=a.output/('block_'+str(b));destination.mkdir()
        records=[r for r in rows if r['entity_id'] in block_ids[b]]
        pairs=pd.concat(frames[b],ignore_index=True);pairs.to_parquet(destination/'pairs.parquet',index=False)
        reuse.write_json(destination/'references.json',records)
        reuse.write_json(destination/'manifest.json',{'status':'complete','verified_current_pipeline':True,
            'split':'test','role':'runtime_block','reference_ids':[r['entity_id'] for r in records],
            'pairs':len(pairs),'pairs_sha256':reuse.digest(destination/'pairs.parquet'),
            'pair_order_sha256':sibling.pair_order_hash(pairs),'frozen_sha256':m['baseline_frozen']['sha256'],
            'routing_universe_pair_order_sha256':m['pair_order_sha256'],'audit_labels_read':False,
            'parent_reuse_manifest_sha256':reuse.digest(a.reuse_manifest)})
        blocks.append({'path':str(destination),'references':len(records),'pairs':len(pairs),
            'countries':dict(Counter(r['country'].strip().casefold() for r in records))})
    reuse.write_json(a.output/'plan.json',{'status':'complete','labels_read':False,'blocks':blocks,
        'full_references':len(ids),'full_pairs':m['pairs'],'routed_references':len(owners),
        'routed_pairs':route[sibling.KEYS].drop_duplicates().shape[0],
        'route_manifest_sha256':reuse.digest(global_route/'manifest.json'),
        'full_route_setup_seconds':time.perf_counter()-start,'code_sha256':reuse.digest(__file__)})
    print(json.dumps({'stage':'complete','blocks':blocks,'routed_pairs':route[sibling.KEYS].drop_duplicates().shape[0],
        'seconds':time.perf_counter()-start}),flush=True)


if __name__=='__main__':main()
