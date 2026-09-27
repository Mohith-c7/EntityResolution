"""Serialize both runtime blocks once from the original official test records."""
import argparse
import json
from pathlib import Path
import time
import pandas as pd
from prepare_neural_inputs import serialize,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blocks',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.perf_counter();keys=['source1_entity_id','candidate_entity_id']
    frames=[];markers=[]
    for b in [0,1]:
        directory=a.blocks/('block_'+str(b))/'combined_reverse'
        marker=json.loads((directory/'manifest.json').read_text())
        if marker['status']!='complete' or sha(directory/'features.parquet')!=marker['features_sha256']:
            raise ValueError('Runtime features are not sealed')
        frames.append(pd.read_parquet(directory/'features.parquet',columns=keys));markers.append(marker)
    owners=set().union(*(set(f.source1_entity_id) for f in frames))
    targets=set().union(*(set(f.candidate_entity_id) for f in frames));records={};hashes={}
    for source in [1,2,3]:
        path=a.data_dir/f'test_source{source}.tsv';hashes['S'+str(source)]=sha(path)
        wanted=owners if source==1 else targets
        for chunk in pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False,chunksize=100000):
            for row in chunk[chunk.entity_id.isin(wanted)].to_dict('records'):
                if row['entity_id'] in records:raise ValueError('Duplicate official ID')
                records[row['entity_id']]=serialize(row)
    if set(records)!=owners|targets:raise ValueError('Missing official records')
    a.output.mkdir(parents=True,exist_ok=False)
    for b,frame in enumerate(frames):
        directory=a.output/('runtime_block_'+str(b));directory.mkdir();path=directory/'pairs.jsonl'
        with path.open('x') as stream:
            for owner,target in frame.itertuples(index=False,name=None):
                stream.write(json.dumps({keys[0]:owner,keys[1]:target,'text_left':records[owner],
                    'text_right':records[target]},ensure_ascii=False)+'\n')
        (directory/'manifest.json').write_text(json.dumps({'status':'complete','rows':len(frame),
            'labels_read':False,'input_sha256':sha(path),'source_tsv_sha256':hashes,
            'feature_manifest_sha256':sha(a.blocks/('block_'+str(b))/'combined_reverse/manifest.json'),
            'code_sha256':sha(__file__),'source_split':'test'},indent=2)+'\n')
    (a.output/'timing.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'rows':sum(map(len,frames))})+'\n')
    print('Runtime neural inputs sealed',flush=True)


if __name__=='__main__':main()
