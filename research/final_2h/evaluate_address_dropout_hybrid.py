"""One fixed dropout8 substitution on exposed30k; retain fixed4 empty extras.

Uses identical current hybrid thresholds and selected-only ownership. Reads
only previously exposed early10k/selection20k labels, never audit labels.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research/final_2h')]
from evaluate_sprint import decide_control,predictions_for,paired_evaluate
from export_sprint_workbench import sha
from evaluate_fixed_hybrid import combine
from post_selection_exclusivity import exclusive_selected
B=Path('research/sprint_6h/sibling')
KEYS=['source1_entity_id','candidate_entity_id']
EXPECTED=.9809431572046812
CONFIG={'threshold':.83,'t_first':.6999999999999998,'t_rest':.79}
CRITERIA={'minimum_point_macro_gain':.0003,'paired_ci_lower_bound_strictly_above':0.,'country_macro_delta_floor':-.0005,'precision_singletons_diagnostic_only':True}

def passed(result):
    return result['paired_macro_f05_delta']>=CRITERIA['minimum_point_macro_gain'] and result['paired_delta_95pct_ci'][0]>0 and min(result['country_deltas'].values())>=CRITERIA['country_macro_delta_floor']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,default=Path('research/final_2h/neural_address_dropout_fine_v1'))
    p.add_argument('--output',type=Path,default=Path('research/final_2h/neural_address_dropout_hybrid_v1'))
    p.add_argument('--declare-only',action='store_true')
    a=p.parse_args()
    protocol_path=a.output/'protocol.json'
    if a.declare_only:
        a.output.mkdir(parents=True,exist_ok=False)
        protocol={'status':'declared_before_new_model_results','declared_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'candidate':'new balanced-target-dropout8 native38 original-route probabilities plus EXACT saved fixed4 disjoint empty-extra probabilities; anchored ORIGINALp2 once; one candidate no search','baseline':'current calibrated/exclusive hybrid8_empty4','baseline_pairs_sha256':sha('research/final_2h/fixed_hybrid_v1/pairs.parquet'),'baseline_development_macro_f05':EXPECTED,'decision_config':CONFIG,'post_selection_policy':'existing exclusive_selected probability then lexical source1ID','post_selection_helper_sha256':sha(ROOT/'research/final_2h/post_selection_exclusivity.py'),'criteria':CRITERIA,'fit':'existing residual20k only, logloss early-stop early10k only, same anchored38 native hyperparameters as8','evaluation':'complete frozen30k graph; exposed early10k and selection20k only; one selection comparison','audit_labels_read':False,'extension_audit_labels_read':False,'production_modified':False}
        protocol_path.write_text(json.dumps(protocol,indent=2)+'\n');return
    protocol=json.loads(protocol_path.read_text())
    if (protocol['source_sha256']!=sha(__file__) or protocol['decision_config']!=CONFIG or protocol['criteria']!=CRITERIA or sha('research/final_2h/fixed_hybrid_v1/pairs.parquet')!=protocol['baseline_pairs_sha256'] or sha(ROOT/'research/final_2h/post_selection_exclusivity.py')!=protocol['post_selection_helper_sha256']):raise ValueError('Prospective source/config/baseline changed')
    if (a.output/'report.json').exists():raise ValueError('Single evaluation only')
    original=pd.read_parquet(B/'learned30k/pairs.parquet')
    four=pd.read_parquet('research/final_2h/neural_last4_fine_v1/pairs.parquet')
    baseline=pd.read_parquet('research/final_2h/fixed_hybrid_v1/pairs.parquet')
    native=pd.read_parquet(a.native/'pairs.parquet')
    rescue=pd.read_parquet('research/final_2h/empty_rescue_fixed38_v1/pairs.parquet')
    if len(original)!=1200000 or original.source1_entity_id.nunique()!=30000 or original.groupby('source1_entity_id').size().ne(40).any():raise ValueError('Incomplete30k graph')
    for frame in (four,baseline,native,rescue):
        if not frame[KEYS].equals(original[KEYS]) or not np.array_equal(frame.probability_original,original.probability):raise ValueError('Wrong graph or anchor')
        for col in ['first_stage','candidate_order']:
            if not np.array_equal(frame[col],original[col]):raise ValueError('p1/order changed')
    report=json.loads((a.native/'report.json').read_text());head=Path('models/final_2h/neural_address_dropout8_v1/manifest.json')
    if report['model_sha256']!=sha(a.native/'adapter.txt') or report['fine_head_sha256']!=sha(head) or report['selection_labels_read'] is not False or report['fresh_labels_read'] is not False:raise ValueError('Native fit lineage invalid')
    route=pd.read_parquet(B/'neural_v1_features30k/features.parquet',columns=KEYS);old_keys=set(route.itertuples(index=False,name=None))
    extra=[]
    for role,count in [('early_stop',2460),('selection',4735)]:
        path=Path('research/final_2h/empty_rescue_v1')/role/'features.parquet'
        marker=json.loads(path.with_name('manifest.json').read_text())
        if marker['version']!='empty-rescue-direct-v1-36' or sha(path)!=marker['features_sha256']:raise ValueError('Saved extra seals changed')
        f=pd.read_parquet(path,columns=KEYS)
        f=f[[tuple(k) not in old_keys for k in f.itertuples(index=False,name=None)]]
        if len(f)!=count:raise ValueError('Extra keys differ')
        extra.append(f)
    candidate,old_pos,new_pos=combine(original,four,native,rescue,route,pd.concat(extra,ignore_index=True))
    candidate.to_parquet(a.output/'pairs.parquet',index=False)
    before,_=decide_control(baseline,baseline,CONFIG);before,base_removed=exclusive_selected(baseline,before)
    after,_=decide_control(candidate,candidate,CONFIG);after,candidate_removed=exclusive_selected(candidate,after)
    results={}
    for role,count in [('early_stop',10000),('selection',20000)]:
        wb=B/'workbenches50k'/role
        truth=json.loads((wb/'truth.json').read_text());countries=json.loads((wb/'countries.json').read_text())
        if len(truth)!=count or set(countries)!=set(truth):raise ValueError('Wrong authorized role')
        mask=original.source1_entity_id.isin(truth).to_numpy()
        r=paired_evaluate(truth,predictions_for(baseline[mask],before[mask],truth),predictions_for(candidate[mask],after[mask],truth),countries,role='development')
        if role=='selection' and abs(r['baseline']['macro_f05']-EXPECTED)>1e-14:raise ValueError('Current hybrid baseline replay differs')
        r['legacy_quality_diagnostics']=r.pop('acceptance');r['prospective_pass']=passed(r);results[role]=r
    out={'status':'single_fixed_model_development_evaluation_complete','results':results,'prospective_pass':passed(results['selection']),'protocol_sha256':sha(protocol_path),'source_sha256':sha(__file__),'candidate_pairs_sha256':sha(a.output/'pairs.parquet'),'native_report_sha256':sha(a.native/'report.json'),'new_head_manifest_sha256':sha(head),'original_route_keys':len(old_pos),'fixed4_extra_keys':len(new_pos),'base_exclusivity_removed':int(base_removed.sum()),'candidate_exclusivity_removed':int(candidate_removed.sum()),'claimant_scope':'frozen30k/1200000pairs; notfull331k','audit_labels_read':False,'extension_audit_labels_read':False,'production_modified':False}
    (a.output/'report.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'selection':results['selection'],'prospective_pass':out['prospective_pass']}),flush=True)

if __name__=='__main__':main()
