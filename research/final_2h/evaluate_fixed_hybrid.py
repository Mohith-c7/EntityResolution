"""One prospective NN8/original-route + fixed4/extra-empty-key combination.

Saved corrected probabilities are overwritten by exact pair key. No correction
is added to an already corrected probability. No fitting or parameter search.
Old components remain rejected under their original guards.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts')]
from export_sprint_workbench import sha
from evaluate_sprint import decide_control,predictions_for,paired_evaluate,entity_f05
KEYS=['source1_entity_id','candidate_entity_id']
B=Path('research/sprint_6h/sibling')
FOUR=Path('research/final_2h/neural_last4_fine_v1')
EIGHT=Path('research/final_2h/neural_last8_fine_v1')
RESCUE=Path('research/final_2h/empty_rescue_fixed38_v1')
PILOT=Path('research/final_2h/empty_rescue_v1')


def prospective_pass(result):
    return (result['paired_macro_f05_delta']>=.0003 and result['paired_delta_95pct_ci'][0]>0
        and min(result['country_deltas'].values())>=-.0005)


def combine(original,baseline,eight,rescue,original_route,extra_keys):
    if any(not original[KEYS].equals(f[KEYS]) for f in [baseline,eight,rescue]):
        raise ValueError('Saved candidate graph/order differs')
    index=pd.MultiIndex.from_frame(original[KEYS])
    if index.has_duplicates or original[KEYS].isna().any().any():raise ValueError('Invalid original pair keys')
    old_index=pd.MultiIndex.from_frame(original_route[KEYS])
    new_index=pd.MultiIndex.from_frame(extra_keys[KEYS])
    if old_index.has_duplicates or new_index.has_duplicates or len(old_index.intersection(new_index)):
        raise ValueError('Extra and original route must be unique/disjoint')
    old_positions=index.get_indexer(old_index);new_positions=index.get_indexer(new_index)
    if (old_positions<0).any() or (new_positions<0).any():raise ValueError('Foreign route pair key')
    for f in [original,baseline,eight,rescue]:
        if not f.probability.between(0,1).all():raise ValueError('Invalid probability')
    out=eight.copy();values=out.probability.to_numpy(copy=True)
    values[new_positions]=rescue.probability.to_numpy()[new_positions]
    out['probability']=values
    old_mask=np.zeros(len(out),dtype=bool);old_mask[old_positions]=True
    new_mask=np.zeros(len(out),dtype=bool);new_mask[new_positions]=True
    if not np.array_equal(eight.probability.to_numpy()[~old_mask],original.probability.to_numpy()[~old_mask]):
        raise ValueError('NN8 changed a key outside its original route')
    if not np.array_equal(rescue.probability.to_numpy()[~new_mask],baseline.probability.to_numpy()[~new_mask]):
        raise ValueError('Saved rescue changed outside extra keys')
    if not np.array_equal(out.probability.to_numpy()[old_positions],eight.probability.to_numpy()[old_positions]):
        raise ValueError('NN8 original route changed during combination')
    outside=~(old_mask|new_mask)
    if not np.array_equal(out.probability.to_numpy()[outside],baseline.probability.to_numpy()[outside]):
        raise ValueError('Hybrid changed outside original/extra key union')
    return out,old_positions,new_positions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('research/final_2h/fixed_hybrid_v1'))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Single prospective candidate only')
    sources={'original':B/'learned30k/pairs.parquet','baseline05':FOUR/'pairs.parquet',
        'eight':EIGHT/'pairs.parquet','rescue':RESCUE/'pairs.parquet',
        'original_route':B/'neural_v1_features30k/features.parquet',
        'eight_report':EIGHT/'report.json','four_report':FOUR/'report.json',
        'old_eight_selection':Path('research/final_2h/neural_last8_fine_v1_selection/report.json'),
        'old_rescue_result':RESCUE/'report.json','pilot_protocol':PILOT/'protocol.json',
        'pilot_validation':PILOT/'validation.json'}
    hashes={name:sha(path) for name,path in sources.items()}
    model_hashes={name:sha(path) for name,path in {'native8':EIGHT/'adapter.txt','native4':FOUR/'adapter.txt',
        'head8':Path('models/final_2h/neural_last8_continue_v1/manifest.json'),
        'head4':Path('models/final_2h/neural_last4_continue_v1/manifest.json'),
        'old_head':Path('models/sprint_6h/neural/frozen_head120k_v1/manifest.json')}.items()}
    a.output.mkdir(parents=True)
    declaration={'status':'prospective_fixed_hybrid_declared_before_combined_results',
        'declared_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),
        'candidate':'NN8 saved probabilities on original native route; fixed4 saved empty-rescue probabilities on ONLY disjoint extra keys; exact corrected-probability overwrite, no double anchor',
        'baseline':'Selected frozen05 four-layer global30k decisions',
        'criteria':{'primary':'Paired macroF0.5 confidence interval lower bound strictly above0 versus05',
            'minimum_point_macro_gain':.0003,
            'country_macro_delta_floor':-.0005,'microprecision_and_singleton_false_predictions':'Diagnostics only under NEW prospective rule'},
        'old_component_rejections_retained':True,'no_search_fitting_threshold_or_strength_tuning':True,
        'scope':'Exposed early10k/selection20k, complete frozen30k claimant graph; no full331k ownership parity claim',
        'fresh_or_extension_audit_labels_read':False,'input_sha256':hashes,'models_sha256':model_hashes,
        'bootstrap':{'seed':42,'iterations':2000,'unit':'Source1 owner','paired':True}}
    (a.output/'protocol.json').write_text(json.dumps(declaration,indent=2)+'\n')
    original,baseline,eight,rescue=[pd.read_parquet(sources[k]) for k in ['original','baseline05','eight','rescue']]
    if len(original)!=1200000 or original.source1_entity_id.nunique()!=30000 or original.groupby('source1_entity_id').size().ne(40).any():
        raise ValueError('Complete30k/all40 graph required')
    for frame in [baseline,eight,rescue]:
        if not np.array_equal(frame.probability_original,original.probability):raise ValueError('Frozen ORIGINALp2 anchor differs')
        for column in ['first_stage','candidate_order']:
            if not np.array_equal(frame[column],original[column]):raise ValueError('Frozen p1/rank source order differs')
    report8=json.loads(sources['eight_report'].read_text());report4=json.loads(sources['four_report'].read_text())
    if report8['model_sha256']!=model_hashes['native8'] or report4['model_sha256']!=model_hashes['native4']:
        raise ValueError('Native adapter seal differs')
    if report8['fine_head_sha256']!=model_hashes['head8'] or report4['fine_head_sha256']!=model_hashes['head4']:
        raise ValueError('Fine-head lineage differs')
    old8=json.loads(sources['old_eight_selection'].read_text());old_rescue=json.loads(sources['old_rescue_result'].read_text())
    if old8['predeclared_acceptance_pass'] is not False or old_rescue['acceptance_pass'] is not False:
        raise ValueError('Old component rejection metadata must remain unchanged')
    if old_rescue['candidate_pairs_sha256']!=hashes['rescue'] or old8['eight_layer_report_sha256']!=hashes['eight_report']:
        raise ValueError('Saved component lineage differs')
    old_route=pd.read_parquet(sources['original_route'],columns=KEYS)
    old_keys=set(old_route.itertuples(index=False,name=None))
    roles={r:json.loads((B/f'workbenches50k/{r}/reference_ids.json').read_text()) for r in ['early_stop','selection']}
    if [len(roles[r]) for r in roles]!=[10000,20000] or set(roles['early_stop'])&set(roles['selection']) or set().union(*map(set,roles.values()))!=set(original.source1_entity_id):
        raise ValueError('Authorized development roles differ')
    extra=[];counts={}
    for role,expected in [('early_stop',2460),('selection',4735)]:
        f=pd.read_parquet(PILOT/role/'features.parquet',columns=KEYS)
        marker=json.loads((PILOT/role/'manifest.json').read_text())
        if marker['version']!='empty-rescue-direct-v1-36' or sha(PILOT/role/'features.parquet')!=marker['features_sha256']:
            raise ValueError('New empty-route feature source seal differs')
        if not set(f.source1_entity_id)<=set(roles[role]):raise ValueError('Extra route crossed role boundary')
        f=f[[tuple(key) not in old_keys for key in f.itertuples(index=False,name=None)]].reset_index(drop=True)
        if len(f)!=expected:raise ValueError('Disjoint extra key count differs')
        counts[role]=len(f);extra.append(f)
    extra=pd.concat(extra,ignore_index=True)
    hybrid,old_pos,new_pos=combine(original,baseline,eight,rescue,old_route,extra)
    hybrid.to_parquet(a.output/'pairs.parquet',index=False)
    config=json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
    config={k:config[k] for k in ['threshold','t_first','t_rest']}
    base_keep,_=decide_control(baseline,baseline,config);keep,_=decide_control(hybrid,hybrid,config)
    results={}
    for role in ['early_stop','selection']:
        wb=B/f'workbenches50k/{role}'
        truth=json.loads((wb/'truth.json').read_text());countries=json.loads((wb/'countries.json').read_text())
        if set(truth)!=set(roles[role]) or set(countries)!=set(truth):raise ValueError('Role truth/country scope differs')
        mask=original.source1_entity_id.isin(truth).to_numpy()
        base_pred=predictions_for(baseline[mask],base_keep[mask],truth)
        if role=='selection' and float(np.mean([entity_f05(truth[e],base_pred[e]) for e in sorted(truth)]))!=.9802609267254342:
            raise ValueError('Exact frozen05 selection baseline replay differs')
        result=paired_evaluate(truth,base_pred,predictions_for(hybrid[mask],keep[mask],truth),countries,role='development')
        result['legacy_guard_diagnostics']=result.pop('acceptance')
        result['new_prospective_metric_rule_pass']=prospective_pass(result)
        results[role]=result
    result={'status':'one_fixed_hybrid_development_evaluation_complete','results':results,
        'prospective_acceptance_pass':prospective_pass(results['selection']),
        'protocol_sha256':sha(a.output/'protocol.json'),'source_sha256':sha(__file__),
        'hybrid_pairs_sha256':sha(a.output/'pairs.parquet'),'original_route_keys':len(old_pos),'extra_keys':counts,
        'NN8_original_route_preserved_exactly':True,'outside_original_and_extra_keys_unchanged':True,
        'corrections_added_to_corrected_anchor':False,'old_component_failures_remain_failed':True,
        'extension_audit_labels_read':False,'production_changes':False,'models_sha256':model_hashes,'input_sha256':hashes}
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    selection=results['selection']
    print(json.dumps({'score':selection['candidate']['macro_f05'],'gain':selection['paired_macro_f05_delta'],
        'ci':selection['paired_delta_95pct_ci'],'new_rule_pass':result['prospective_acceptance_pass'],
        'countries':selection['country_deltas'],'baseline':selection['baseline'],'candidate':selection['candidate'],'links':selection['links']}),flush=True)


if __name__=='__main__':main()
