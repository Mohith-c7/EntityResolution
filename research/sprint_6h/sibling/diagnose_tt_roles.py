"""Read-only TT evidence on fitting20k and disjoint early10k development roles."""
import json
from pathlib import Path
import sys
import time
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow as pa
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import train_sibling_matcher as s
from run_frozen_pipeline import decide
B=ROOT/'research/sprint_6h/sibling'; pa.set_cpu_count(1)
TT=lgb.Booster(model_file=str(B/'compatibility/model.txt'))
RES=lgb.Booster(model_file=str(B/'residual/residual.txt'))
config=json.loads((ROOT/'research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
INDIC={'DEVANAGARI','BENGALI','GUJARATI','GURMUKHI','TAMIL','TELUGU','KANNADA','MALAYALAM','ORIYA'}
start=time.perf_counter();reports={}
def summaries(data,group):
 out=[]
 for keys,g in data.groupby(group,observed=True):
  if not isinstance(keys,tuple):keys=(keys,)
  row=dict(zip(group,map(str,keys)));row['count']=len(g)
  for col in ['probability','first_stage','sibling_min','sibling_max','sibling_mean','residual_p']:
   row[col+'_mean']=float(g[col].mean());row[col+'_q10']=float(g[col].quantile(.1));row[col+'_q90']=float(g[col].quantile(.9))
  row['seed_any_true_fraction']=float(g.seed_any_true.mean());out.append(row)
 return out
for role in ['residual_train','early_stop']:
 R=B/'workbenches50k'/role
 truth=json.loads((R/'truth.json').read_text());owner=s.owner_map(truth);wanted=set(truth)
 if role=='residual_train':
  frame,m=s.load_pairs(R/'pairs.parquet',R/'manifest.json');full,_=s.aligned_support(frame,B/'residual_support/support.parquet',B/'residual_support/manifest.json');route=s.select_route(frame,universe_ids=m['reference_ids'])
 else:
  frame,m=s.load_pairs(B/'learned30k/pairs.parquet',B/'learned30k/manifest.json');full,_=s.aligned_support(frame,B/'learned30k_support/support.parquet',B/'learned30k_support/manifest.json');route,_=s.load_pinned_route(frame,m,B/'learned30k_route')
 route={i:p for i,p in route.items() if frame.source1_entity_id.iat[i] in wanted}
 ids={frame.candidate_entity_id.iat[j] for i,peers in route.items() for j in [i,*peers]}
 records=s.load_targets(ids,str(ROOT/'models/index_train'));owner.update(s.lookup_training_owners(ids,ROOT/'models/scale_v1_plan/aliases/counts.sqlite'))
 data=full.loc[list(route)].copy();data['label']=[int(c in truth[e]) for e,c in data[s.KEYS].itertuples(index=False,name=None)]
 data['status']=np.where(data.accepted,np.where(data.label,'TP','FP'),np.where(data.label,'FN','TN'))
 data['missing_address']=[not bool(records[c]['address']) for c in data.candidate_entity_id]
 data['indic_target']=[bool((records[c]['name_scripts']|records[c]['address_scripts'])&INDIC) for c in data.candidate_entity_id]
 data['seed_any_true']=[any(frame.candidate_entity_id.iat[j] in truth[frame.source1_entity_id.iat[i]] for j in route[i]) for i in data.index]
 data['residual_p']=RES.predict(data[s.RESIDUAL_NAMES].to_numpy(dtype=np.float32),num_threads=1)
 edges=[];features=[]
 for i,peers in route.items():
  e=frame.source1_entity_id.iat[i];a=frame.candidate_entity_id.iat[i]
  for j in peers:
   b=frame.candidate_entity_id.iat[j];a_true=a in truth[e];b_true=b in truth[e]
   label=1 if a_true and b_true else 0 if a_true!=b_true else int(owner[a]==owner[b]) if a in owner and b in owner else -1
   v=s.sibling_features(records[a],records[b]);features.append(v)
   edges.append({'compat_label':label,'candidate_true':a_true,'seed_true':b_true,'name_sim':float(v[2]),'address_sim':float(v[7]),'candidate_missing_addr':not bool(records[a]['address']),'candidate_indic':bool((records[a]['name_scripts']|records[a]['address_scripts'])&INDIC)})
 edges=pd.DataFrame(edges);edges['TT_p']=TT.predict(np.asarray(features,dtype=np.float32),num_threads=1)
 bins=[]
 for label,g in edges.groupby('compat_label'):
  bins.append({'compat_label':int(label),'count':len(g),'TT_mean':float(g.TT_p.mean()),'TT_q10':float(g.TT_p.quantile(.1)),'TT_q90':float(g.TT_p.quantile(.9)),'TT_ge90':int((g.TT_p>=.9).sum())})
 negatives=edges[edges.compat_label==0]
 collisions=[]
 for kind,mask in [('same_address',negatives.address_sim>=.9),('same_name',negatives.name_sim>=.9),('same_both',(negatives.address_sim>=.9)&(negatives.name_sim>=.9))]:
  g=negatives[mask];collisions.append({'kind':kind,'count':len(g),'TT_mean':float(g.TT_p.mean()) if len(g) else None,'TT_ge90':int((g.TT_p>=.9).sum())})
 # Same universe and frozen policy, using already frozen .75 solely to explain rejection.
 changed=frame.copy();p=RES.predict(full.loc[list(route),s.RESIDUAL_NAMES].to_numpy(dtype=np.float32),num_threads=1)
 changed.loc[list(route),'probability']=.25*frame.loc[list(route),'probability'].to_numpy()+.75*p
 chosen,_=decide(changed,changed,config)
 old,_=decide(frame,frame,config);causes=Counter()
 for i in np.flatnonzero(chosen!=old):
  e=frame.source1_entity_id.iat[i]
  if e in wanted:
   truthrow=frame.candidate_entity_id.iat[i] in truth[e]
   causes[('true_' if truthrow else 'false_')+('added' if chosen[i] else 'removed')+('_routed' if i in route else '_ownership_other')]+=1
 report={'role':role,'scope':'Already-authorized development labels only. Fitting-role summaries are in-sample; early_stop is disjoint residual owner evidence on complete30k claimant graph. No selection labels or next fit.',
 'reference_count':len(wanted),'routed_references':int(data.source1_entity_id.nunique()),'routed_rows':len(data),'status_support':summaries(data,['status']),
 'missing_address_support':summaries(data,['status','missing_address']),'target_script_support':summaries(data,['status','indic_target']),
 'TT_edge_classes':bins,'hard_negative_collisions':collisions,'seed_correct_fraction':float(edges.seed_true.mean()),
 'frozen75_changed_link_causes':dict(causes)}
 reports[role]=report
 s.write_json(B/(role+'_diagnostic_rows.json'),report)
 print(json.dumps({'role':role,'edge_classes':bins,'collisions':collisions,'seed_correct_fraction':report['seed_correct_fraction'],'changed_causes':dict(causes)}),flush=True)
s.write_json(B/'tt_role_diagnostic.json',{'status':'read_only_complete','seconds':time.perf_counter()-start,'roles':reports})
