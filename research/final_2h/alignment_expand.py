"""Join corrected token alignment to the independently prepared broad cache."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/business_entity_resolution'),str(ROOT/'scripts'),str(ROOT/'research/final_2h')]
import broad_evidence as broad
from src.features.token_alignment import AlignmentStatistics,FEATURE_NAMES,build_alignment_features
STATS=None;RECORDS={}


def job(keys):
    return [[e,c,*build_alignment_features(RECORDS[e],RECORDS[c],STATS).values()]
        for e,c in keys]


def main():
    global STATS,RECORDS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--statistics',type=Path,required=True)
    p.add_argument('--source1-index',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    a=p.parse_args();started=time.monotonic()
    if a.output.exists() or not 1<=a.workers<=24:raise ValueError('Unused output/bounded workers required')
    marker=json.loads((a.features/'manifest.json').read_text())
    if marker['labels_read'] or marker['features']!=broad.NAMES:raise ValueError('Wrong broad cache')
    frames=[];wanted=set();keys=[]
    for name in ['train','development']:
        path=a.features/(name+'.parquet')
        if broad.sha(path)!=marker[name+'_sha256']:raise ValueError('Broad cache seal differs')
        frame=pd.read_parquet(path)
        mask=(frame.p2>=.003)|(frame.rank_p1<=4)
        pairs=list(frame.loc[mask,broad.sibling.KEYS].itertuples(index=False,name=None))
        frames.append(frame);keys.extend(pairs)
        wanted.update(e for pair in pairs for e in pair)
    STATS=AlignmentStatistics.from_dict(json.loads(a.statistics.read_text()))
    if STATS.record_count!=2206821:raise ValueError('Wrong Source1 statistics corpus')
    for source in [1,2,3]:
        path=a.source1_index if source==1 else Path(f'models/index_train_S{source}.sqlite')
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as conn:
            if source==1:
                meta=json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
                if meta['fingerprint']['source_sha256'] not in set(STATS.source_sha256.values()):raise ValueError('Statistics from another corpus')
            ids=sorted(e for e in wanted if e.startswith(f'S{source}-'))
            address='primary_address' if source==1 else 'address'
            for start in range(0,len(ids),500):
                batch=ids[start:start+500]
                for e,name,addr,country in conn.execute(f'SELECT id,name,{address},country FROM records WHERE id IN ('+','.join('?'*len(batch))+')',batch):
                    RECORDS[e]=SimpleNamespace(name=name,address=addr,country=country)
    if set(RECORDS)!=wanted:raise ValueError('Missing provided record')
    values=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for number,result in enumerate(pool.map(job,[keys[i:i+2000] for i in range(0,len(keys),2000)])):
            values.extend(result)
            if number%20==0:print(json.dumps({'stage':'alignment','pairs':len(values),'seconds':time.monotonic()-started}),flush=True)
    aligned=pd.DataFrame(values,columns=[*broad.sibling.KEYS,*FEATURE_NAMES])
    if aligned.duplicated(broad.sibling.KEYS).any() or not np.isfinite(aligned[FEATURE_NAMES]).all().all():raise ValueError('Invalid alignment rows')
    aligned['alignment_scored']=1.
    names=[*broad.NAMES,*FEATURE_NAMES,'alignment_scored']
    a.output.mkdir(parents=True)
    for name,frame in zip(['train','development'],frames):
        added=aligned[aligned.source1_entity_id.isin(set(frame.source1_entity_id))]
        combined=frame.merge(added,on=broad.sibling.KEYS,how='left',sort=False,validate='one_to_one')
        combined[[*FEATURE_NAMES,'alignment_scored']]=combined[[*FEATURE_NAMES,'alignment_scored']].fillna(0)
        if not frame[broad.sibling.KEYS].equals(combined[broad.sibling.KEYS]):raise ValueError('Pair order changed')
        combined.to_parquet(a.output/(name+'.parquet'),index=False)
    result={**marker,'features':names,'parent_manifest_sha256':broad.sha(a.features/'manifest.json'),
        'statistics_sha256':broad.sha(a.statistics),'alignment_code_sha256':broad.sha(ROOT/'code/business_entity_resolution/src/features/token_alignment.py'),
        'alignment_version':STATS.version,'alignment_pairs':len(aligned),'alignment_route':'p2>=.003 or first_stage_rank<=4',
        'code_sha256':broad.sha(__file__),'seconds':time.monotonic()-started}
    for name in ['train','development']:result[name+'_sha256']=broad.sha(a.output/(name+'.parquet'))
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'complete','pairs':len(aligned),'features':len(names),'seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
