"""Add label-free complete-graph collision checks to a development report."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_sprint import decide_control, predictions_for
from export_sprint_workbench import sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("source","candidate","frozen","report"):p.add_argument("--"+name,type=Path,required=True)
 a=p.parse_args();r=json.loads(a.report.read_text())
 if r.get('role')!='development':raise ValueError('Collision supplement only allowed on development report')
 config=json.loads(a.frozen.read_text());maps=[];counts=[]
 ids=json.loads((a.source/'reference_ids.json').read_text())
 for name,path in (('baseline',a.source),('candidate',a.candidate)):
  m=json.loads((path/'manifest.json').read_text());pairs=path/'pairs.parquet'
  if sha(pairs)!=m['pairs_sha256']:raise ValueError('Scorechecksumchanged')
  frame=pd.read_parquet(pairs,columns=['source1_entity_id','candidate_entity_id','probability','candidate_order'])
  chosen,_=decide_control(frame,frame,config)
  predicted=predictions_for(frame,chosen,ids)
  early=json.loads((a.report.parent/(name+'_predictions.json')).read_text())
  if any(set(v)!=predicted[e] for e,v in early.items()):raise ValueError('Earlyslice differs from completegraphdecisions')
  owners=frame.loc[chosen].groupby('candidate_entity_id').source1_entity_id.agg(lambda v:frozenset(v)).to_dict()
  maps.append(owners);counts.append({'accepted_links':int(chosen.sum()),'collision_targets':sum(len(v)>1 for v in owners.values())})
 base,trial=maps
 r['complete_graph_collision_stress']={'claimants':len(ids),'baseline':counts[0],'candidate':counts[1],
  'owner_targets_changed':sum(base.get(t,frozenset())!=trial.get(t,frozenset()) for t in set(base)|set(trial)),
  'early_slice_decisions_equal_complete_graph':True,'labels_read':False,'prepared_at':datetime.now(timezone.utc).isoformat(),
  'scope':'Complete30k decision-only stress; quality measured only on early10k.'}
 a.report.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['complete_graph_collision_stress']),flush=True)
if __name__=='__main__':main()
