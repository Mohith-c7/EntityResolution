"""Read-only early10k accepted-link explanation for authorized pilot only."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
import train_sibling_matcher as s
from run_frozen_pipeline import decide
B=ROOT/'research/sprint_6h/sibling';R=B/'workbenches50k/early_stop';pa.set_cpu_count(1)
f,m=s.load_pairs(B/'learned30k/pairs.parquet',B/'learned30k/manifest.json')
cm=json.loads((B/'adapter_v1_scored30k/manifest.json').read_text());p=B/'adapter_v1_scored30k/pairs.parquet'
if s.digest(p)!=cm['pairs_sha256']:raise ValueError('Changedpilotseal')
a=pd.read_parquet(p)
if s.pair_order_hash(a)!=m['pair_order_sha256'] or not np.array_equal(a.probability_original,f.probability):raise ValueError('Changedbaselinekeys/scores')
t=json.loads((R/'truth.json').read_text());countries=json.loads((R/'countries.json').read_text());ids=set(t)
config=json.loads((ROOT/'research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text());before,_=decide(f,f,config);after,_=decide(a,a,config)
pred_before={e:set() for e in ids};pred_after={e:set() for e in ids};change_rows=[]
for i,(e,c) in enumerate(f[s.KEYS].itertuples(index=False,name=None)):
 if e not in ids:continue
 if before[i]:pred_before[e].add(c)
 if after[i]:pred_after[e].add(c)
 if before[i]!=after[i]:change_rows.append({'source1_entity_id':e,'candidate_entity_id':c,'country':countries[e],'correct':c in t[e],'added':bool(after[i]),'baseline_p2':float(f.probability.iat[i]),'adapter_p2':float(a.probability.iat[i]),'correction':float(a.adapter_correction.iat[i]),'routed':bool(a.adapter_correction.iat[i]!=0)})
records=s.load_targets({r['candidate_entity_id'] for r in change_rows},str(ROOT/'models/index_train'))
for row in change_rows:row['target_address_missing']=not bool(records[row['candidate_entity_id']]['address'])
mask=f.source1_entity_id.isin(ids).to_numpy();bv=s.entity_values(f[mask],before[mask],t);av=s.entity_values(a[mask],after[mask],t)
def describe(owners):
 rows=[r for r in change_rows if r['source1_entity_id'] in owners]
 counts={kind:sum(r['added']==add and r['correct']==correct for r in rows) for kind,add,correct in [('TP_added',True,True),('FP_added',True,False),('TP_removed',False,True),('FP_removed',False,False)]}
 counts.update(missing_address_TP_added=sum(r['added'] and r['correct'] and r['target_address_missing'] for r in rows),missing_address_FP_added=sum(r['added'] and not r['correct'] and r['target_address_missing'] for r in rows),baseline_macro_f05=float(np.mean([bv[e] for e in owners])),adapter_macro_f05=float(np.mean([av[e] for e in owners])),macro_gain=float(np.mean([av[e]-bv[e] for e in owners])))
 for name,pred in [('baseline',pred_before),('adapter',pred_after)]:
  tp=sum(len(pred[e]&set(t[e])) for e in owners);fp=sum(len(pred[e]-set(t[e])) for e in owners);fn=sum(len(set(t[e])-pred[e]) for e in owners)
  counts[name]={'tp':tp,'fp':fp,'fn':fn,'micro_precision':tp/max(tp+fp,1),'micro_recall':tp/max(tp+fn,1),'empty_truth_false_link_entities':sum(not t[e] and bool(pred[e]) for e in owners),'known_positive_empty_predictions':sum(bool(t[e]) and not pred[e] for e in owners),'one_true_target_overlinked_entities':sum(len(t[e])==1 and len(pred[e])>1 for e in owners)}
 return counts
mm=json.loads((B/'adapter_v1/manifest.json').read_text());model=lgb.Booster(model_file=str(B/'adapter_v1/adapter.txt'));importance=sorted(zip(model.feature_name(),map(float,model.feature_importance('gain'))),key=lambda v:-v[1])
report={'scope':'Early10k stopping/tuning evidence only, complete30k claimant graph; selected20k labels/metrics not read.','best_iteration':mm['best_iteration'],'early_logloss':mm['early_logloss'],'overall':describe(ids),'countries':{c:describe({e for e in ids if countries[e]==c}) for c in sorted(set(countries.values()))},'top_feature_gain':importance[:20]}
s.write_json(B/'adapter_v1_early_changes.json',report);pd.DataFrame(change_rows).to_parquet(B/'adapter_v1_early_changed_links.parquet',index=False)
print(json.dumps(report),flush=True)
