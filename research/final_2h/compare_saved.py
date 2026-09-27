"""Compare saved score graphs on the already exposed selection20k only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_sprint import decide_control,paired_evaluate,predictions_for
B=Path('research/sprint_6h/sibling')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('New output required')
    claims=pd.read_parquet(B/'learned30k/pairs.parquet')
    candidate=pd.read_parquet(a.candidate/'pairs.parquet')
    nn=pd.read_parquet(B/'neural_v2_scored30k/pairs.parquet')
    keys=['source1_entity_id','candidate_entity_id','candidate_order']
    if not claims[keys].equals(candidate[keys]) or not claims[keys].equals(nn[keys]):raise ValueError('Claim graph differs')
    manifest=json.loads((a.candidate/'report.json').read_text())
    if set(manifest['fit_owners'])&set(claims.source1_entity_id):raise ValueError('Candidate fitted selection/early owner')
    truth=json.loads((B/'workbenches50k/selection/truth.json').read_text())
    countries=json.loads((B/'workbenches50k/selection/countries.json').read_text())
    if len(truth)!=20000:raise ValueError('Expected existing selection20k only')
    config=json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
    config={k:config[k] for k in ['threshold','t_first','t_rest']}
    mask=claims.source1_entity_id.isin(truth).to_numpy();predictions=[]
    for i,frame in enumerate([claims,nn,candidate]):
        chosen,_=decide_control(frame,frame,manifest.get('decision_config',config) if i==2 else config)
        predictions.append(predictions_for(frame.loc[mask],chosen[mask],truth))
    a.output.mkdir(parents=True)
    report={'scope':'already exposed selection20k, complete declared30k claimant graph; not a fresh audit',
        'vs_incumbent':paired_evaluate(truth,predictions[0],predictions[2],countries,role='development'),
        'vs_neural_v2':paired_evaluate(truth,predictions[1],predictions[2],countries,role='development')}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{'score':v['candidate']['macro_f05'],'delta':v['paired_macro_f05_delta'],
        'ci':v['paired_delta_95pct_ci'],'links':v['links'],'acceptance':v['acceptance']} for k,v in report.items() if k!='scope'}),flush=True)


if __name__=='__main__':main()
