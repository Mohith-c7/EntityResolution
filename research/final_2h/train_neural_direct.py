"""Test direct classification within the sealed neural route, calibrating on early10k."""
import argparse
import json
from pathlib import Path
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/final_2h'),str(ROOT/'scripts')]
import neural_peer as peer
from evaluate_sprint import decide_control,paired_evaluate,predictions_for
from export_sprint_workbench import sha
B=peer.B


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fine_scores','dev_fine_scores','fine_input_manifest','dev_fine_input_manifest','fine_head_manifest','output']:
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Unused output required')
    train,tm=peer.nn.load_features(B/'neural_v1_features_fit');dev,dm=peer.nn.load_features(B/'neural_v1_features30k')
    if tm['role']!='residual_train' or len(tm['reference_ids'])!=20000 or len(dm['reference_ids'])!=30000:
        raise ValueError('Wrong sealed development roles/cardinality')
    train=peer.add_fine(train,tm,a.fine_scores,a.fine_input_manifest,a.fine_head_manifest)
    dev=peer.add_fine(dev,dm,a.dev_fine_scores,a.dev_fine_input_manifest,a.fine_head_manifest)
    names=[*peer.nn.FEATURES,'neural_fine_probability']
    truth=json.loads((B/'workbenches50k/residual_train/truth.json').read_text())
    et=json.loads((B/'workbenches50k/early_stop/truth.json').read_text());countries=json.loads((B/'workbenches50k/early_stop/countries.json').read_text())
    if len(truth)!=20000 or len(et)!=10000 or set(truth)&set(et) or set(train.source1_entity_id)&set(dev.source1_entity_id):raise ValueError('Owner split differs')
    early=dev[dev.source1_entity_id.isin(et)]
    def ds(f,t):
        y=np.array([int(c in t[e]) for e,c in f[peer.nn.sibling.KEYS].itertuples(index=False,name=None)])
        return lgb.Dataset(f[names].to_numpy(dtype=np.float32),label=y,
            weight=1/f.groupby('source1_entity_id').source1_entity_id.transform('size').to_numpy(),feature_name=names)
    params=dict(objective='binary',metric='binary_logloss',num_leaves=31,min_data_in_leaf=40,
        lambda_l2=1.,learning_rate=.05,max_bin=127,num_threads=8,deterministic=True,force_col_wise=True,seed=42,verbosity=-1)
    model=lgb.train(params,ds(train,truth),num_boost_round=800,valid_sets=[ds(early,et)],callbacks=[lgb.early_stopping(40,verbose=False)])
    claims=pd.read_parquet(B/'learned30k/pairs.parquet')
    positions=pd.MultiIndex.from_frame(claims[peer.nn.sibling.KEYS]).get_indexer(pd.MultiIndex.from_frame(dev[peer.nn.sibling.KEYS]))
    if (positions<0).any() or not np.array_equal(claims.probability.to_numpy()[positions],dev.probability.to_numpy()):raise ValueError('Foreign routed anchor')
    config=json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text());config={k:config[k] for k in ['threshold','t_first','t_rest']}
    base,_=decide_control(claims,claims,config);mask=claims.source1_entity_id.isin(et).to_numpy();bp=predictions_for(claims[mask],base[mask],et)
    from src.evaluation.metrics import score_matches
    baseline_metrics=score_matches(et,bp,countries)
    raw=model.predict(dev[names].to_numpy(dtype=np.float32),raw_score=True,num_threads=8)
    grid=[];best=None
    for shift in np.arange(-1.,1.61,.2):
        changed=claims.copy();changed['probability_original']=claims.probability
        changed.loc[positions,'probability']=1/(1+np.exp(-np.clip(raw+shift,-50,50)))
        chosen,_=decide_control(changed,changed,config)
        cp=predictions_for(changed[mask],chosen[mask],et);metrics=score_matches(et,cp,countries)
        eligible=metrics['micro_precision']>=baseline_metrics['micro_precision'] and metrics['singleton_false_positives']<=baseline_metrics['singleton_false_positives']
        grid.append({'shift':float(shift),'macro_f05':metrics['macro_f05'],'precision':metrics['micro_precision'],'singleton_false_positives':metrics['singleton_false_positives'],'eligible':eligible})
        if eligible and (best is None or metrics['macro_f05']>best[0]):best=(metrics['macro_f05'],float(shift),changed,cp)
    if best is None:raise ValueError('No predeclared calibration satisfies precision/singleton guard')
    _,shift,changed,cp=best
    result=paired_evaluate(et,bp,cp,countries,role='development')
    a.output.mkdir(parents=True);model.save_model(str(a.output/'adapter.txt'));changed.to_parquet(a.output/'pairs.parquet',index=False)
    result.update(status='early10k_only',features=names,parameters=params,best_iteration=model.best_iteration,
        mode='direct_neural',calibration_logit_shift=shift,calibration_grid=grid,
        fit_owners=sorted(truth),model_sha256=sha(a.output/'adapter.txt'),source_sha256=sha(__file__),
        fine_head_sha256=sha(a.fine_head_manifest),selection_labels_read=False,fresh_labels_read=False)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['paired_macro_f05_delta','paired_delta_95pct_ci','links','acceptance','calibration_logit_shift']}),flush=True)


if __name__=='__main__':main()
