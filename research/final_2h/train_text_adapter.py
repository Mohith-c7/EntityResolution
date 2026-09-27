"""Train and evaluate declared broad text feature variants on existing splits."""
import argparse
import json
from pathlib import Path
import sys
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research/final_2h'),str(ROOT/'research/sprint_6h/sibling')]
import broad_evidence as broad
from evaluate_sprint import decide_control,paired_evaluate,predictions_for


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args();start=time.monotonic()
    if a.output.exists() or not 1<=a.workers<=24:raise ValueError('Unused output/bounded workers required')
    m=json.loads((a.features/'manifest.json').read_text());names=m['features']
    if m['status']!='complete' or m['labels_read'] or names[:60]!=broad.NAMES or len(names) not in [60,77]:raise ValueError('Wrong feature variant')
    if len(set(names))!=len(names):raise ValueError('Repeated feature')
    data=[]
    for name in ['train','development']:
        path=a.features/(name+'.parquet')
        if broad.sha(path)!=m[name+'_sha256']:raise ValueError('Feature bytes differ')
        frame=pd.read_parquet(path)
        if frame.duplicated(broad.sibling.KEYS).any() or not np.isfinite(frame[names]).all().all():raise ValueError('Invalid features')
        data.append(frame)
    train,dev=data;B=broad.B
    truth=json.loads((B/'workbenches50k/residual_train/truth.json').read_text())
    et=json.loads((B/'workbenches50k/early_stop/truth.json').read_text())
    countries=json.loads((B/'workbenches50k/early_stop/countries.json').read_text())
    if len(truth)!=20000 or len(et)!=10000 or set(truth)&set(et) or set(dev.source1_entity_id)&set(truth):raise ValueError('Wrong train/early split')
    if set(train.source1_entity_id)!=set(truth):raise ValueError('Training identities differ')
    early=dev[dev.source1_entity_id.isin(et)]
    def dataset(f,t):
        y=np.array([int(c in t[e]) for e,c in f[broad.sibling.KEYS].itertuples(index=False,name=None)])
        return lgb.Dataset(f[names].to_numpy(dtype=np.float32),label=y,
            weight=1/f.groupby('source1_entity_id').source1_entity_id.transform('size').to_numpy(),
            init_score=broad.reverse.base_logit(f.p2),feature_name=names)
    params=dict(objective='binary',metric='binary_logloss',num_leaves=63,min_data_in_leaf=60,lambda_l2=3.,
        learning_rate=.05,max_bin=127,num_threads=a.workers,force_col_wise=True,deterministic=True,
        seed=42,verbosity=-1,boost_from_average=False)
    model=lgb.train(params,dataset(train,truth),num_boost_round=1600,valid_sets=[dataset(early,et)],
        callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(200)])
    a.output.mkdir(parents=True);model.save_model(str(a.output/'adapter.txt'))
    claims=pd.read_parquet(B/'learned30k/pairs.parquet')
    positions=pd.MultiIndex.from_frame(claims[broad.sibling.KEYS]).get_indexer(pd.MultiIndex.from_frame(dev[broad.sibling.KEYS]))
    if (positions<0).any() or not np.array_equal(claims.probability.to_numpy()[positions],dev.p2.to_numpy()):raise ValueError('Wrong anchors/keys')
    changed=claims.copy();changed['probability_original']=claims.probability
    correction=model.predict(dev[names].to_numpy(dtype=np.float32),raw_score=True,num_threads=a.workers)
    changed.loc[positions,'probability']=broad.reverse.corrected_probability(dev.p2,correction)
    changed.to_parquet(a.output/'pairs.parquet',index=False)
    config=json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
    config={k:config[k] for k in ['threshold','t_first','t_rest']}
    base,_=decide_control(claims,claims,config);trial,_=decide_control(changed,changed,config)
    mask=claims.source1_entity_id.isin(et).to_numpy()
    bp=predictions_for(claims.loc[mask],base[mask],et);cp=predictions_for(changed.loc[mask],trial[mask],et)
    result=paired_evaluate(et,bp,cp,countries,role='development')
    result.update(status='early10k_only',features=names,parameters=params,best_iteration=model.best_iteration,
        model_sha256=broad.sha(a.output/'adapter.txt'),feature_manifest_sha256=broad.sha(a.features/'manifest.json'),
        source_sha256=broad.sha(__file__),fit_owners=sorted(truth),early_owners=sorted(et),
        selection_labels_read=False,fresh_labels_read=False,seconds=time.monotonic()-start)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['paired_macro_f05_delta','paired_delta_95pct_ci','links','acceptance','seconds']}),flush=True)


if __name__=='__main__':main()
