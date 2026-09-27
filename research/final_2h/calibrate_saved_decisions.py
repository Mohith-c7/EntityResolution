"""Select decision thresholds on existing early10k for a differently scaled model."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research/final_2h')]
from early_macro_metric import EarlyMacroMetric
from evaluate_sprint import decide_control,paired_evaluate,predictions_for
from export_sprint_workbench import sha
B=Path('research/sprint_6h/sibling')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Unused output required')
    model_report=json.loads((a.candidate/'report.json').read_text())
    if model_report.get('mode')!='direct':raise ValueError('This calibration is predeclared for direct probabilities only')
    claims=pd.read_parquet(B/'learned30k/pairs.parquet');trial=pd.read_parquet(a.candidate/'pairs.parquet')
    keys=['source1_entity_id','candidate_entity_id','candidate_order']
    if not claims[keys].equals(trial[keys]):raise ValueError('Candidate graph differs')
    truth=json.loads((B/'workbenches50k/early_stop/truth.json').read_text());countries=json.loads((B/'workbenches50k/early_stop/countries.json').read_text())
    if len(truth)!=10000 or set(truth)&set(model_report['fit_owners']):raise ValueError('Early cohort differs/contains training owners')
    config=json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text());config={k:config[k] for k in ['threshold','t_first','t_rest']}
    mask=claims.source1_entity_id.isin(truth).to_numpy();base,_=decide_control(claims,claims,config)
    bp=predictions_for(claims[mask],base[mask],truth)
    early=trial[mask].reset_index(drop=True);prob=early.probability.to_numpy()
    metric=EarlyMacroMetric(early,early,truth)
    rows=[]
    for rest in np.arange(.60,.91,.01):
        for first in [.55,.60,.65,.70,.75]:
            if first>rest:continue
            metric.t_first=first;metric.t_rest=float(rest)
            rows.append({'t_rest':float(rest),'t_first':first,'proxy_macro_f05':metric(prob)[1]})
    rows.sort(key=lambda r:(-r['proxy_macro_f05'],-r['t_rest'],-r['t_first']))
    # Verify the best proxy setting against complete30k ownership, without
    # consulting selection20k labels. Other grid settings are never promoted.
    winner=rows[0];chosen_config={'threshold':winner['t_rest'],'t_rest':winner['t_rest'],'t_first':winner['t_first']}
    selected,_=decide_control(trial,trial,chosen_config)
    result=paired_evaluate(truth,bp,predictions_for(trial[mask],selected[mask],truth),countries,role='development')
    a.output.mkdir(parents=True);trial.to_parquet(a.output/'pairs.parquet',index=False)
    report={**model_report,**result,'threshold_selection':'existing early10k only, not a fresh result','decision_config':chosen_config,
        'grid':rows,'parent_report_sha256':sha(a.candidate/'report.json'),'parent_pairs_sha256':sha(a.candidate/'pairs.parquet'),
        'source_sha256':sha(__file__),'selection_labels_read':False,'fresh_labels_read':False}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['decision_config','paired_macro_f05_delta','paired_delta_95pct_ci','links','acceptance']}),flush=True)


if __name__=='__main__':main()
