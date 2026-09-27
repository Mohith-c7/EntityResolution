"""Simple exact same-scope learned-grid readout; shared validation owns acceptance."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import train_sibling_matcher as sibling
from run_frozen_pipeline import decide
BASE=ROOT/'research/sprint_6h/sibling'
pa.set_cpu_count(1)
start=time.perf_counter()
role=BASE/'learned30k'
frame,manifest=sibling.load_pairs(role/'pairs.parquet',role/'manifest.json')
truth=json.loads((role/'truth.json').read_text())
ids=json.loads((role/'evaluation_reference_ids.json').read_text())
countries=json.loads((role/'countries.json').read_text())
config=json.loads((ROOT/'research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
if sibling.digest(ROOT/'research/sprint_6h/coordination/baseline_freeze/frozen.json')!=manifest['frozen_sha256']:
 raise ValueError('Frozen decision configuration changed')

def metrics(frame,mask):
 values=sibling.entity_values(frame,mask,truth)
 pred={e:set() for e in ids}
 for (s,c),keep in zip(frame[sibling.KEYS].itertuples(index=False,name=None),mask):
  if keep and s in pred: pred[s].add(c)
 tp=sum(len(pred[e]&set(truth[e])) for e in ids)
 fp=sum(len(pred[e]-set(truth[e])) for e in ids)
 fn=sum(len(set(truth[e])-pred[e]) for e in ids)
 return {'macro_f05':float(np.mean([values[e] for e in ids])),
 'country_macro_f05':{c:float(np.mean([values[e] for e in ids if countries[e]==c])) for c in sorted(set(countries.values()))},
 'tp':tp,'fp':fp,'fn':fn,'micro_precision':tp/(tp+fp),'micro_recall':tp/(tp+fn),
 'singleton_overlink':sum(len(pred[e])>1 for e in ids if len(truth[e])==1),
 'empty_target_false_link_entities':sum(bool(pred[e]) for e in ids if not truth[e])}
control,_=decide(frame,frame,config)
control_metrics=metrics(frame,control)
if abs(control_metrics['macro_f05']-manifest['evaluation_expected_macro_f05'])>1e-12:
 raise ValueError('Current30k control differs from declared same-scope metric')
trials=[]
for weight in sibling.LEARNED_WEIGHTS:
 directory=BASE/('learned_weight_'+str(weight).replace('.','p'))
 record=json.loads((directory/'manifest.json').read_text())
 if record['status']!='scored_requires_global_decision_replay' or sibling.digest(directory/'pairs.parquet')!=record['pairs_sha256']:
  raise ValueError('Unsealed changed pair table')
 changed=pd.read_parquet(directory/'pairs.parquet')
 if sibling.pair_order_hash(changed)!=manifest['pair_order_sha256'] or not np.array_equal(changed.first_stage,frame.first_stage):
  raise ValueError('Changed retrieval or firststage scores')
 if not np.array_equal(changed.probability_original,frame.probability): raise ValueError('Original p2 changed')
 accepted,_=decide(changed,changed,config)
 result=metrics(changed,accepted)
 result.update(weight=weight,gain=result['macro_f05']-control_metrics['macro_f05'],
 country_gain={c:result['country_macro_f05'][c]-control_metrics['country_macro_f05'][c] for c in result['country_macro_f05']},
 changed_decisions=int(np.sum(control!=accepted)),pairs_sha256=record['pairs_sha256'])
 trials.append(result)
 print(json.dumps(result),flush=True)
report={'status':'frozen_grid_same_scope_readout','scope':'Complete30k claimant graph; selected20k evaluation; no fresh confirmation/audit labels. Paired guardrail validation separate.',
 'control':control_metrics,'trials':trials,'reference_count':len(ids),'claimant_count':len(truth),'seconds':time.perf_counter()-start,
 'source_manifest_sha256':sibling.digest(role/'manifest.json'),'evaluation_ids_sha256':sibling.digest(role/'evaluation_reference_ids.json')}
sibling.write_json(BASE/'learned_grid_report.json',report)
