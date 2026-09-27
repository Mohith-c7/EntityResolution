"""One selection20k comparison after a passing predeclared early pilot."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from export_sprint_workbench import sha
from evaluate_sprint import decide_control,paired_evaluate,predictions_for


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','candidate','model','features','truth','countries','frozen','pilot','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Selection may only run once')
    pilot=json.loads(a.pilot.read_text())
    if pilot['paired_macro_f05_delta']<.0005 or pilot['acceptance']['quality_non_inferiority_pass'] is not True:
        raise ValueError('Pilot did not meet the declared selection gate')
    sm=json.loads((a.source/'manifest.json').read_text());cm=json.loads((a.candidate/'manifest.json').read_text())
    mm=json.loads((a.model/'manifest.json').read_text());fm=json.loads((a.features/'manifest.json').read_text())
    if sm['status']!='complete' or sm.get('new_component_fit_owners_excluded') is not True or cm['adapter_model_sha256']!=mm['model_sha256'] or sha(a.model/'adapter.txt')!=mm['model_sha256']:
        raise ValueError('Model/held-out claimant provenance differs')
    if sha(a.frozen)!=sm['frozen_sha256'] or sha(a.features/'manifest.json')!=cm['reverse_feature_manifest_sha256'] or sha(a.features/'features.parquet')!=fm['features_sha256']:
        raise ValueError('Freeze/features differ')
    before=pd.read_parquet(a.source/'pairs.parquet');after=pd.read_parquet(a.candidate/'pairs.parquet')
    if sha(a.source/'pairs.parquet')!=sm['pairs_sha256'] or sha(a.candidate/'pairs.parquet')!=cm['pairs_sha256'] or not before[['source1_entity_id','candidate_entity_id','candidate_order']].equals(after[['source1_entity_id','candidate_entity_id','candidate_order']]):
        raise ValueError('Claims changed pair keys or order')
    if not np.array_equal(before.probability.to_numpy(),after.probability_original.to_numpy()):raise ValueError('Baseline changed')
    ids=json.loads((a.source/'reference_ids.json').read_text())
    if len(ids)!=30000 or set(ids)&set(mm['training_owner_ids']):raise ValueError('Wrong or fitted claim graph')
    # Labels are opened only after all candidate identity checks above.
    truth=json.loads(a.truth.read_text());countries=json.loads(a.countries.read_text())
    if len(truth)!=20000 or set(truth)!=set(ids)-set(mm['early_owner_ids']) or set(countries)!=set(truth):raise ValueError('Selection cohort differs')
    frozen=json.loads(a.frozen.read_text());config={k:frozen[k] for k in ['threshold','t_first','t_rest']}
    bm,_=decide_control(before,before,config);am,_=decide_control(after,after,config)
    mask=before.source1_entity_id.isin(truth).to_numpy()
    bp=predictions_for(before[mask],bm[mask],truth);ap=predictions_for(after[mask],am[mask],truth)
    report=paired_evaluate(truth,bp,ap,countries,role='development',iterations=2000,
        universe={'kind':'complete_development','complete':True,'scoring_complete':True,'declared_references':len(ids),'scored_references':len(ids),'new_component_fit_owners_excluded':True})
    report.update(stage='selection20k_first_neural_evaluation',fresh_confirmation_labels_read=False,fresh_audit_labels_read=False,
        hashes={'source':sha(a.source/'manifest.json'),'candidate':sha(a.candidate/'manifest.json'),'model':mm['model_sha256'],'pilot':sha(a.pilot),'truth':sha(a.truth),'countries':sha(a.countries)})
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['baseline','candidate','paired_macro_f05_delta','paired_delta_95pct_ci','country_deltas','links','acceptance']}),flush=True)


if __name__=='__main__':main()
