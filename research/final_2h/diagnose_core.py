"""Read already exposed development labels and show the remaining error modes."""
from collections import Counter
import json
from pathlib import Path
import sys
import sqlite3
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'research/sprint_6h/sibling')]
import train_reverse_adapter as reverse
from evaluate_sprint import decide_control,predictions_for

B=Path('research/sprint_6h/sibling')
frame=pd.read_parquet(B/'neural_v2_scored30k/pairs.parquet')
truth=json.loads((B/'workbenches50k/selection/truth.json').read_text())
config=json.loads(Path('research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
chosen,lost=decide_control(frame,frame,{k:config[k] for k in ['threshold','t_first','t_rest']})
mask=frame.source1_entity_id.isin(truth).to_numpy()
pred=predictions_for(frame.loc[mask],chosen[mask],truth)
routes=set(map(tuple,pd.read_parquet(B/'neural_v1_features30k/features.parquet',columns=reverse.sibling.KEYS).to_numpy()))
fn=[];fp=[];buckets=Counter()
lookup=frame.set_index(reverse.sibling.KEYS)
for owner,targets in truth.items():
 for target in set(targets)-set(pred[owner]):
  key=(owner,target)
  if key not in lookup.index: buckets['unretrieved']+=1;continue
  row=lookup.loc[key]; probability=float(row.probability)
  buckets['rejected_routed' if key in routes else 'rejected_unrouted']+=1
  buckets['rejected_p_'+str(int(probability*10)/10)]+=1
  fn.append((owner,target,probability,float(row.first_stage),key in routes))
 for target in set(pred[owner])-set(targets):
  row=lookup.loc[(owner,target)];fp.append((owner,target,float(row.probability),float(row.first_stage),(owner,target) in routes))
def get_records(conn,ids):
 out={}
 for start in range(0,len(ids),400):
  batch=ids[start:start+400]
  for row in conn.execute('SELECT id,name,address,country FROM records WHERE id IN ('+','.join('?'*len(batch))+')',batch):out[row[0]]=row[1:]
 return out
sample=sorted(fn,key=lambda x:(x[2],x[0]))[::max(1,len(fn)//45)][:45]+sorted(fp,key=lambda x:-x[2])[:20]
owners=list({e for e,_,*rest in sample});targets=list({c for _,c,*rest in sample})
records={}
wanted=set(owners)
for chunk in pd.read_csv('dataset/train/train_source1.tsv',sep='\t',dtype=str,keep_default_na=False,chunksize=100000):
 for row in chunk.loc[chunk.entity_id.isin(wanted)].itertuples(index=False):
  records[row.entity_id]=[row.business_name,row.business_address,row.country]
for source in [2,3]:
 with sqlite3.connect(Path(f'models/index_train_S{source}.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as c:records.update(get_records(c,[i for i in targets if i.startswith(f'S{source}-')]))
report={'scope':'already exposed selection20k, never fresh audit','counts':dict(buckets),'false_links':len(fp),'examples':[
 {'type':'rejected_true' if c in truth[e] else 'false_link','source1':e,'target':c,'p2':p,'p1':p1,'routed':r,
  'left':records[e],'right':records[c]} for e,c,p,p1,r in sample]}
out=Path('research/final_2h/core_diagnosis.json');out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'counts':report['counts'],'false_links':len(fp)}),flush=True)
