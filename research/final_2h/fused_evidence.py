"""Fit one anchored residual using previously isolated evidence families.

Only declared residual training and early stopping labels are read. Existing
feature artifacts are verified by their own loaders; audit labels are forbidden.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'research/sprint_6h'), str(ROOT/'research/sprint_6h/sibling'), str(ROOT/'scripts')]
import neural_adapter as neural
import train_anchored_adapter as raw
from evaluate_sprint import decide_control, paired_evaluate, predictions_for

B = Path('research/sprint_6h/sibling')
F = Path('research/sprint_6h/coordination/baseline_freeze/frozen.json')
KEYS = neural.sibling.KEYS


def fused(nn_path, raw_path):
    n, nm = neural.load_features(nn_path)
    r, rm = raw.load_features(raw_path)
    if n.duplicated(KEYS).any() or r.duplicated(KEYS).any(): raise ValueError('Repeated keys')
    if set(map(tuple,n[KEYS].to_numpy())) != set(map(tuple,r[KEYS].to_numpy())):
        raise ValueError('Evidence routes differ')
    r = r.rename(columns={c:'sibling_'+c for c in raw.FEATURES})
    result = n.merge(r[KEYS+['sibling_'+c for c in raw.FEATURES]], on=KEYS, validate='one_to_one')
    names = neural.FEATURES + ['sibling_'+c for c in raw.FEATURES]
    if not np.isfinite(result[names]).all().all(): raise ValueError('Nonfinite evidence')
    return result, names, {'neural':neural.sibling.digest(nn_path/'manifest.json'), 'raw':neural.sibling.digest(raw_path/'manifest.json')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); start=time.monotonic()
    if a.output.exists(): raise ValueError('Use a new output directory')
    a.output.mkdir(parents=True)
    train,names,tm=fused(B/'neural_v1_features_fit', B/'adapter_v1_features_fit')
    dev,other,dm=fused(B/'neural_v1_features30k', B/'adapter_v1_features30k')
    if names!=other: raise ValueError('Feature order differs')
    truth=json.loads((B/'workbenches50k/residual_train/truth.json').read_text())
    et=json.loads((B/'workbenches50k/early_stop/truth.json').read_text())
    countries=json.loads((B/'workbenches50k/early_stop/countries.json').read_text())
    if len(truth)!=20000 or len(et)!=10000 or set(truth)&set(et) or set(dev.source1_entity_id)&set(truth):
        raise ValueError('Fit/early split differs')
    train=train[train.source1_entity_id.isin(truth)]
    early=dev[dev.source1_entity_id.isin(et)]
    def dataset(frame,t):
        y=np.array([int(c in t[e]) for e,c in frame[KEYS].itertuples(index=False,name=None)])
        return lgb.Dataset(frame[names].to_numpy(dtype=np.float32),label=y,
            weight=1/frame.groupby('source1_entity_id').source1_entity_id.transform('size').to_numpy(),
            init_score=neural.reverse.base_logit(frame.probability),feature_name=names)
    params=dict(objective='binary',metric='binary_logloss',num_leaves=63,min_data_in_leaf=40,
        lambda_l2=3.,learning_rate=.035,max_bin=127,num_threads=12,deterministic=True,
        force_col_wise=True,seed=42,verbosity=-1,boost_from_average=False)
    model=lgb.train(params,dataset(train,truth),num_boost_round=1200,
        valid_sets=[dataset(early,et)],valid_names=['early10k'],
        callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(100)])
    model.save_model(str(a.output/'adapter.txt'))
    claims=pd.read_parquet(B/'learned30k/pairs.parquet')
    positions=pd.MultiIndex.from_frame(claims[KEYS]).get_indexer(pd.MultiIndex.from_frame(dev[KEYS]))
    if (positions<0).any() or not np.array_equal(claims.probability.to_numpy()[positions],dev.probability.to_numpy()):
        raise ValueError('Base probability/key mismatch')
    changed=claims.copy(); delta=model.predict(dev[names].to_numpy(dtype=np.float32),raw_score=True,num_threads=12)
    changed['probability_original']=claims.probability
    changed.loc[positions,'probability']=neural.reverse.corrected_probability(dev.probability,delta)
    changed.to_parquet(a.output/'pairs.parquet',index=False)
    config=json.loads(F.read_text()); config={k:config[k] for k in ['threshold','t_first','t_rest']}
    base,_=decide_control(claims,claims,config); trial,_=decide_control(changed,changed,config)
    mask=claims.source1_entity_id.isin(et).to_numpy()
    bp=predictions_for(claims.loc[mask],base[mask],et)
    cp=predictions_for(changed.loc[mask],trial[mask],et)
    universe={'kind':'complete_development','complete':True,'scoring_complete':True,
        'supervised_training_excluded':True,'declared_references':30000,'scored_references':30000}
    report=paired_evaluate(et,bp,cp,countries,universe=universe,role='development')
    report.update(status='early10k_only',feature_count=len(names),features=names,
        parameters=params,best_iteration=model.best_iteration,train_pairs=len(train),
        early_pairs=len(early),fit_owners=sorted(truth),early_owners=sorted(et),
        model_sha256=neural.sibling.digest(a.output/'adapter.txt'),source_sha256=neural.sibling.digest(__file__),
        feature_manifests={'fit':tm,'development':dm},selection_labels_read=False,
        fresh_labels_read=False,seconds=time.monotonic()-start)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['paired_macro_f05_delta','country_deltas','links','acceptance','seconds']}),flush=True)


if __name__=='__main__':main()
