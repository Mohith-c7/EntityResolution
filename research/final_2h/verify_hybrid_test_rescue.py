"""Independent completed full-test direct-route seal verification; no labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/final_2h'),str(ROOT/'research/sprint_6h'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from prepare_neural_inputs import sha
import prepare_hybrid_test_rescue as prep

def verify(directory,old_features):
    m=json.loads((directory/'manifest.json').read_text());p=json.loads((directory/'protocol.json').read_text())
    im=json.loads((directory/'neural_inputs/manifest.json').read_text())
    if (m.get('status')!='complete' or m.get('version')!=prep.VERSION or m['features']!=prep.adapter.FEATURES
            or m.get('labels_read') is not False or m.get('audit_labels_read') is not False
            or m['split']!='test' or m['role']!='full_official_test_empty_extra'
            or m['source_sha256']!=sha(prep.__file__) or m['feature_code_sha256']!=sha(prep.reverse.__file__)
            or m['join_code_sha256']!=sha(prep.adapter.__file__)
            or m['features_sha256']!=sha(directory/'features.parquet') or m['route_sha256']!=sha(directory/'route.parquet')
            or m['protocol_sha256']!=sha(directory/'protocol.json') or m['input_manifest_sha256']!=sha(directory/'neural_inputs/manifest.json')
            or im.get('labels_read') is not False or im.get('source_split')!='test'
            or im['input_sha256']!=sha(directory/'neural_inputs/pairs.jsonl') or im['route_sha256']!=m['route_sha256']
            or im['source_tsv_sha256']!=m['source_tsv_sha256']):raise ValueError('Typed completed feature/input/source seals differ')
    frame=pd.read_parquet(directory/'features.parquet');route=pd.read_parquet(directory/'route.parquet')
    old=pd.read_parquet(old_features/'features.parquet',columns=prep.KEYS)
    if (frame.duplicated(prep.KEYS).any() or len(frame)!=m['rows'] or not frame[prep.KEYS].equals(route[prep.KEYS])
            or prep.reverse.pair_order_hash(frame)!=m['feature_pair_order_sha256']
            or not np.isfinite(frame[prep.adapter.FEATURES].to_numpy()).all()
            or not frame.candidate_rank.between(1,4).all()
            or frame.groupby(prep.KEYS[0]).size().gt(4).any()
            or pd.MultiIndex.from_frame(frame[prep.KEYS]).isin(pd.MultiIndex.from_frame(old)).any()
            or im['pair_order_sha256']!=m['feature_pair_order_sha256']):raise ValueError('Direct route universe/ranks/schema/order differ')
    if sha(old_features/'features.parquet')!=p['original_neural_keys_sha256']:raise ValueError('Original route keys changed')
    parts=sorted((directory/'parts').iterdir(),key=lambda x:int(x.name));diagnostic_rows=0
    if [sha(part/'manifest.json') for part in parts]!=m['part_manifest_sha256']:raise ValueError('Part manifest union changed')
    seen=set()
    for part in parts:
        pm=json.loads((part/'manifest.json').read_text())
        if (pm['status']!='complete' or pm['labels_read'] is not False or pm['features_sha256']!=sha(part/'features.parquet')
                or pm['reverse33_sha256']!=sha(part/'reverse33.parquet')
                or pm['diagnostics_sha256']!=sha(part/'diagnostics.jsonl')
                or pm['part_pairs_sha256']!=sha(part/'pairs.parquet') or pm['part_route_sha256']!=sha(part/'route.parquet')
                or pm['index_manifest_sha256']!=m['index_manifest_sha256']):raise ValueError('Part actual-feature/index/input seals differ')
        partkeys=set(pd.read_parquet(part/'route.parquet',columns=prep.KEYS).itertuples(index=False,name=None));count=0
        with (part/'diagnostics.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line);key=tuple(row[k] for k in prep.KEYS)
                if key in seen or key not in partkeys or key[0] in row['rival_ids'] or len(row['rival_ids'])>3:raise ValueError('Duplicate/foreign/self rival diagnostic')
                seen.add(key);count+=1
        if count!=pm['rows'] or count!=len(partkeys):raise ValueError('Incomplete part diagnostics')
        diagnostic_rows+=count
    if seen!=set(frame[prep.KEYS].itertuples(index=False,name=None)):raise ValueError('Diagnostic union differs')
    order=hashlib.sha256();rows=0
    with (directory/'neural_inputs/pairs.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if set(row)!=set([*prep.KEYS,'text_left','text_right']) or not isinstance(row['text_left'],str) or not isinstance(row['text_right'],str):raise ValueError('Invalid raw serialized pair schema')
            order.update((row[prep.KEYS[0]]+'\t'+row[prep.KEYS[1]]+'\n').encode());rows+=1
    if rows!=len(frame) or order.hexdigest()!=m['feature_pair_order_sha256']:raise ValueError('Raw input exact key order/coverage differs')
    result={'status':'passed','feature_manifest_sha256':sha(directory/'manifest.json'),'protocol_sha256':sha(directory/'protocol.json'),
        'raw_manifest_sha256':sha(directory/'neural_inputs/manifest.json'),'raw_input_sha256':im['input_sha256'],
        'features_sha256':m['features_sha256'],'rows':rows,'empty_owners':p['empty_owners'],'overlap_removed':p['overlap_removed'],
        'diagnostics_verified':diagnostic_rows,'workers':len(parts),'exact_input_feature_key_order':True,'original_neural_route_disjoint':True,
        'rank_limited_to_original_top4':True,'actual_own_excluded_reverse_queries':True,'labels_read':False,'audit_labels_read':False,
        'source_sha256':sha(__file__),'builder_sha256':sha(prep.__file__)}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=Path('models/final_2h/hybrid_test_rescue_v1'))
    p.add_argument('--old-features',type=Path,default=Path('models/final_2h/test_combined_neural05'))
    a=p.parse_args();result=verify(a.directory,a.old_features)
    output=a.directory/'validation.json'
    if output.exists():raise ValueError('Do not overwrite completed validation')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
