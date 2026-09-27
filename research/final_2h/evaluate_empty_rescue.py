"""One fixed38 empty-rescue diagnostic; no fit, gate or threshold tuning.

Preserves every prior routed probability. New keyed corrections replace only
extra pair probabilities using the original frozenp2 anchor exactly once.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research/sprint_6h/sibling'),str(ROOT/'research/sprint_6h'),str(ROOT/'research/final_2h')]
import train_reverse_adapter as adapter
import neural_adapter as nn
import neural_peer
from export_sprint_workbench import sha
from evaluate_sprint import decide_control,predictions_for,paired_evaluate,entity_f05
KEYS=['source1_entity_id','candidate_entity_id']
B=Path('research/sprint_6h/sibling')
SELECTED=Path('research/final_2h/neural_last4_fine_v1')


def load_role(directory,role,protocol,preparation):
    fm=json.loads((directory/'manifest.json').read_text())
    im=json.loads((directory/'neural_inputs/manifest.json').read_text())
    if (fm['status']!='complete' or fm['version']!='empty-rescue-direct-v1-36'
        or fm['features']!=adapter.FEATURES or fm['role']!=role or im['role']!=role
        or fm['labels_read'] is not False or im['labels_read'] is not False
        or im['source_split']!='development' or im['feature_manifest_sha256']!=sha(directory/'manifest.json')
        or im['feature_file_sha256']!=fm['features_sha256']
        or sha(directory/'manifest.json')!=preparation['roles'][role]['feature_manifest_sha256']
        or sha(directory/'neural_inputs/manifest.json')!=preparation['roles'][role]['input_manifest_sha256']
        or sha(directory/'features.parquet')!=fm['features_sha256']
        or sha(directory/'neural_inputs/pairs.jsonl')!=im['input_sha256']):
        raise ValueError('New direct route feature/input seals differ; never accept old version')
    frame=pd.read_parquet(directory/'features.parquet')
    if frame.duplicated(KEYS).any() or not np.isfinite(frame[adapter.FEATURES]).all().all():raise ValueError('Invalid36 features')
    for kind,head_field,column,code in [('old','head_manifest_sha256','neural_probability','research/sprint_6h/neural_frozen_head.py'),
        ('fine','checkpoint_manifest_sha256','neural_fine_probability','research/final_2h/neural_layers.py')]:
        scores=directory/f'neural_{kind}'
        m=json.loads((scores/'manifest.json').read_text())
        if (m.get('status')!='complete' or m.get('labels_read') is not False or m['rows']!=len(frame)
            or m['input_sha256']!=im['input_sha256'] or m.get('input_manifest_sha256',sha(directory/'neural_inputs/manifest.json'))!=sha(directory/'neural_inputs/manifest.json')
            or m[head_field]!=protocol['heads'][kind] or sha(scores/'scores.jsonl')!=m['scores_sha256']
            or m['code_sha256']!=sha(code)):
            raise ValueError(f'{kind} head/score/input/code lineage differs')
        values=pd.read_json(scores/'scores.jsonl',lines=True).rename(columns={'neural_probability':column})
        if (values.duplicated(KEYS).any() or not values[column].between(0,1).all()
            or set(map(tuple,values[KEYS].to_numpy()))!=set(map(tuple,frame[KEYS].to_numpy()))):
            raise ValueError(f'{kind} neural pair keys/probabilities differ')
        before=frame[KEYS].copy()
        frame=frame.merge(values[KEYS+[column]],on=KEYS,how='left',sort=False,validate='one_to_one')
        if not before.equals(frame[KEYS]):raise ValueError('Key order changed')
    return frame


def extra_only_scatter(original,baseline,features,existing_keys,model,names,threads=4):
    if not original[KEYS].equals(baseline[KEYS]):raise ValueError('Baseline graph/order differs')
    keep=np.array([tuple(x) not in existing_keys for x in features[KEYS].itertuples(index=False,name=None)])
    extra=features.loc[keep].copy()
    positions=pd.MultiIndex.from_frame(original[KEYS]).get_indexer(pd.MultiIndex.from_frame(extra[KEYS]))
    if (positions<0).any() or len(set(positions))!=len(positions):raise ValueError('Foreign/duplicate extra keys')
    for column,expected in [('probability',original.probability.to_numpy()),('first_stage',original.first_stage.to_numpy()),
        ('candidate_rank',adapter.frozen_p2_rank(original))]:
        if not np.array_equal(extra[column].to_numpy(),expected[positions]):raise ValueError('Originalp2/p1/full40rank differs')
    # Extra keys must have remained at original p2 in the selected baseline.
    if not np.array_equal(baseline.probability.to_numpy()[positions],original.probability.to_numpy()[positions]):
        raise ValueError('Extra key already corrected in baseline')
    correction=model.predict(extra[names].to_numpy(dtype=np.float32),raw_score=True,num_threads=threads)
    result=baseline.copy()
    values=result.probability.to_numpy(copy=True)
    values[positions]=adapter.corrected_probability(original.probability.to_numpy()[positions],correction)
    result['probability']=values
    untouched=np.ones(len(result),dtype=bool);untouched[positions]=False
    if not np.array_equal(result.probability.to_numpy()[untouched],baseline.probability.to_numpy()[untouched]):
        raise ValueError('Existing route/outside extras changed')
    return result,extra,positions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,default=Path('research/final_2h/empty_rescue_v1'))
    p.add_argument('--output',type=Path,default=Path('research/final_2h/empty_rescue_fixed38_v1'))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('One fixed candidate evaluation only')
    prep=json.loads((a.pilot/'manifest.json').read_text())
    validation=json.loads((a.pilot/'validation.json').read_text())
    protocol=json.loads((a.pilot/'protocol.json').read_text())
    if prep['status']!='complete' or validation['status']!='passed' or validation['preparation_manifest_sha256']!=sha(a.pilot/'manifest.json'):
        raise ValueError('Pilot preparation/validation incomplete')
    if (prep['protocol_sha256']!=sha(a.pilot/'protocol.json')
        or prep['source_sha256']!=sha(ROOT/'research/final_2h/prepare_empty_rescue.py')
        or validation['source_sha256']!=sha(ROOT/'research/final_2h/verify_empty_rescue.py')):
        raise ValueError('Preparation protocol/builder/verification source seal differs')
    report=json.loads((SELECTED/'report.json').read_text())
    if sha(SELECTED/'report.json')!=protocol['selected_report_sha256'] or sha(SELECTED/'adapter.txt')!=protocol['selected_adapter_sha256']:
        raise ValueError('Selected fixed adapter changed')
    names=[*adapter.FEATURES,'neural_probability','neural_fine_probability']
    model=lgb.Booster(model_file=str(SELECTED/'adapter.txt'))
    if report['features']!=names or model.feature_name()!=names:raise ValueError('Exact native38 schema differs')
    a.output.mkdir(parents=True)
    declaration={'status':'fixed_candidate_predeclared_before_rescue_selection_results','declared_at':datetime.now(timezone.utc).isoformat(),
        'candidate':'Selected native38 unchanged; apply once from ORIGINALp2 to extra empty top4 keys; existing corrections exact unchanged',
        'quality_checks':{'paired_gain_ci_lower_bound_above':0.,'micro_precision_must_not_fall':True,
            'singleton_false_predictions_must_not_rise':True,'country_delta_floor':-.0005},
        'no_training_gate_threshold_or_strength_tuning':True,'baseline_expected_macro_f05':.9802609267254342,
        'scope':'Frozen development30k only; no full331k ownership parity claim',
        'pilot_manifest_sha256':sha(a.pilot/'manifest.json'),'validation_sha256':sha(a.pilot/'validation.json'),
        'adapter_sha256':sha(SELECTED/'adapter.txt'),'source_sha256':sha(__file__),
        'audit_labels_read':False,'extension_labels_read':False}
    (a.output/'predeclare.json').write_text(json.dumps(declaration,indent=2)+'\n')
    features={r:load_role(a.pilot/r,r,protocol,prep) for r in ['residual_train','early_stop','selection']}
    original=pd.read_parquet(B/'learned30k/pairs.parquet')
    baseline=pd.read_parquet(SELECTED/'pairs.parquet')
    if (len(original)!=1200000 or original.source1_entity_id.nunique()!=30000
        or original.groupby('source1_entity_id').size().ne(40).any()
        or original.duplicated(KEYS).any()):
        raise ValueError('Complete frozen30k graph with all40 candidates required')
    for role in ['early_stop','selection']:
        fm=json.loads((a.pilot/role/'manifest.json').read_text())
        if (fm['baseline_claims_sha256']!=sha(B/'learned30k/pairs.parquet')
            or fm['selected_four_claims_sha256']!=sha(SELECTED/'pairs.parquet')):
            raise ValueError('Original/selected full graph source seals differ')
    if not np.array_equal(baseline.probability_original,original.probability):raise ValueError('Baseline frozen original p2 changed')
    existing=pd.read_parquet(B/'neural_v1_features30k/features.parquet',columns=KEYS)
    existing_keys=set(existing.itertuples(index=False,name=None))
    devfeatures=pd.concat([features['early_stop'],features['selection']],ignore_index=True)
    candidate,extra,positions=extra_only_scatter(original,baseline,devfeatures,existing_keys,model,names)
    if len(extra)!=2460+4735:raise ValueError('New dev extra key set count differs')
    expected_counts={'early_stop':2460,'selection':4735}
    for role,n in expected_counts.items():
        ids=json.loads((B/f'workbenches50k/{role}/reference_ids.json').read_text())
        if extra.source1_entity_id.isin(ids).sum()!=n:raise ValueError('Role extra keys differ')
    # Residual uses the existing calibrated in-sample baseline only for a
    # mechanical scatter proof; no residual truth or gate fitting is performed.
    train_path=B/'workbenches50k/residual_train/pairs.parquet'
    train_original=pd.read_parquet(train_path)
    old,fm=nn.load_features(B/'neural_v1_features_fit')
    old=neural_peer.add_fine(old,fm,Path('models/sprint_6h/neural/last4_residual20k'),
        Path('models/sprint_6h/neural/neural_inputs/residual20k/manifest.json'),
        Path('models/final_2h/neural_last4_continue_v1/manifest.json'))
    if sha(train_path)!=fm['input_pairs_sha256']:raise ValueError('Residual baseline graph bytes differ')
    train_positions=pd.MultiIndex.from_frame(train_original[KEYS]).get_indexer(pd.MultiIndex.from_frame(old[KEYS]))
    if (train_positions<0).any() or not np.array_equal(old.probability,train_original.probability.to_numpy()[train_positions]):
        raise ValueError('Residual prior route anchors differ')
    train_baseline=train_original.copy()
    train_baseline.loc[train_positions,'probability']=adapter.corrected_probability(old.probability,model.predict(old[names].to_numpy(dtype=np.float32),raw_score=True,num_threads=4))
    train_candidate,train_extra,_=extra_only_scatter(train_original,train_baseline,features['residual_train'],set(old[KEYS].itertuples(index=False,name=None)),model,names)
    if len(train_extra)!=4633:raise ValueError('Residual extra key count differs')
    candidate.to_parquet(a.output/'pairs.parquet',index=False)
    config=json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
    config={k:config[k] for k in ['threshold','t_first','t_rest']}
    base_chosen,_=decide_control(baseline,baseline,config)
    new_chosen,_=decide_control(candidate,candidate,config)
    results={}
    for role in ['early_stop','selection']:
        wb=B/f'workbenches50k/{role}'
        truth=json.loads((wb/'truth.json').read_text());countries=json.loads((wb/'countries.json').read_text())
        mask=original.source1_entity_id.isin(truth).to_numpy()
        base_predictions=predictions_for(baseline[mask],base_chosen[mask],truth)
        if role=='selection' and float(np.mean([entity_f05(truth[e],base_predictions[e]) for e in sorted(truth)]))!=.9802609267254342:
            raise ValueError('Exact selected selection replay differs')
        results[role]=paired_evaluate(truth,base_predictions,predictions_for(candidate[mask],new_chosen[mask],truth),countries,role='development')
    chosen_result=results['selection']
    result={'status':'one_fixed_candidate_development_evaluation_complete','results':results,
        'acceptance_pass':chosen_result['acceptance']['quality_improvement_pass'],
        'predeclare_sha256':sha(a.output/'predeclare.json'),'pilot_validation_sha256':sha(a.pilot/'validation.json'),
        'candidate_pairs_sha256':sha(a.output/'pairs.parquet'),'source_sha256':sha(__file__),
        'extra_keys':{'residual_train':4633,'early_stop':2460,'selection':4735},
        'existing_routed_probabilities_preserved_exactly':True,'outside_extra_keys_unchanged':True,
        'residual_in_sample_gate_or_other_model_trained':False,'audit_labels_read':False,'extension_labels_read':False,
        'production_changes':False}
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'score':chosen_result['candidate']['macro_f05'],'gain':chosen_result['paired_macro_f05_delta'],
        'ci':chosen_result['paired_delta_95pct_ci'],'baseline':chosen_result['baseline'],'candidate':chosen_result['candidate'],
        'acceptance':chosen_result['acceptance'],'links':chosen_result['links']}),flush=True)


if __name__=='__main__':main()
