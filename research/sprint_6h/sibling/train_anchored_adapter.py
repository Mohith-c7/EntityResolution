"""Authorized single anchored-logit pilot; support building never reads labels."""
import argparse
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
from src.preprocessing.normalize import normalize_name, normalize_address
VERSION='sibling-anchored-v1-113'
FEATURES=[*(f'direct_{n}' for n in sibling.FEATURE_NAMES),*(f'peer_mean_{n}' for n in sibling.FEATURE_NAMES),*(f'peer_max_{n}' for n in sibling.FEATURE_NAMES),'first_stage','probability','candidate_rank','seed_count','max_seed_p1']
assert len(FEATURES)==113
CLIP=1e-5

def base_logit(probability):
 p=np.clip(np.asarray(probability,dtype=np.float64),CLIP,1-CLIP)
 return np.log(p)-np.log1p(-p)

def corrected_probability(base,correction):
 margin=base_logit(base)+np.asarray(correction,dtype=np.float64)
 correction=np.asarray(correction,dtype=np.float64)
 if not np.isfinite(margin).all() or not np.isfinite(correction).all(): raise ValueError("Nonfiniteadaptercorrection")
 return np.where(correction==0,np.asarray(base,dtype=np.float64),1/(1+np.exp(-np.clip(margin,-50,50))))

def raw_features(frame,route,queries,targets):
 """Only declared raw text + stored p1/p2 and ranks. No truth/owner inputs."""
 rows=[]
 ranks=frame.candidate_order.to_numpy()+1 if 'candidate_order' in frame else frame.groupby('source1_entity_id',sort=False).cumcount().to_numpy()+1
 for i,peers in route.items():
  e=frame.source1_entity_id.iat[i];c=frame.candidate_entity_id.iat[i]
  if e not in queries: raise ValueError('Missing routed rawquery')
  if not peers or len(peers)>2: raise ValueError('Must have1..2 independentseeds')
  if any(frame.candidate_entity_id.iat[j]==c for j in peers): raise ValueError('Selfsupport')
  direct=sibling.sibling_features(queries[e],targets[c])
  matrix=np.asarray([sibling.sibling_features(targets[c],targets[frame.candidate_entity_id.iat[j]]) for j in peers])
  values=np.r_[direct,matrix.mean(axis=0),matrix.max(axis=0),frame.first_stage.iat[i],frame.probability.iat[i],ranks[i],len(peers),max(frame.first_stage.iat[j] for j in peers)].astype(np.float32)
  row={**dict(zip(sibling.KEYS,(e,c))),**dict(zip(FEATURES,values))}
  # Keep exact frozen double p2 for the training/inference offset.
  row['probability']=float(frame.probability.iat[i]);row['first_stage']=float(frame.first_stage.iat[i])
  rows.append(row)
 return pd.DataFrame(rows,columns=[*sibling.KEYS,*FEATURES])

def load_features(directory):
 manifest=json.loads((directory/'manifest.json').read_text());path=directory/'features.parquet'
 if manifest.get('source_sha256')!=sibling.digest(__file__):raise ValueError('Rawadaptercodechanged')
 if manifest.get('status')!='complete' or manifest.get('version')!=VERSION or manifest.get('features')!=FEATURES or sibling.digest(path)!=manifest['features_sha256']:
  raise ValueError('Rawadapter featureseal differs')
 data=pd.read_parquet(path)
 if not set([*sibling.KEYS,*FEATURES])<=set(data) or not np.isfinite(data[FEATURES].to_numpy(dtype=np.float64)).all():raise ValueError('Missingornonfinitefeatures')
 if data.duplicated(sibling.KEYS).any() or sibling.pair_order_hash(data)!=manifest['feature_pair_order_sha256']:
  raise ValueError('Rawadapter keys differ')
 return data,manifest

def score_frame(frame,features,model):
 if not set(FEATURES)<=set(features) or not np.isfinite(features[FEATURES].to_numpy(dtype=np.float64)).all():raise ValueError('Missingornonfinitefeatures')
 lookup=pd.MultiIndex.from_frame(frame[sibling.KEYS]);keys=pd.MultiIndex.from_frame(features[sibling.KEYS])
 positions=lookup.get_indexer(keys)
 if (positions<0).any(): raise ValueError('Adapterfeatures outsidepairuniverse')
 base=frame.probability.to_numpy()[positions]
 if not np.array_equal(base,features.probability.to_numpy()): raise ValueError('Adapterfrozenp2 differs')
 out=frame.copy();out['probability_original']=frame.probability;out['adapter_correction']=0.
 correction=model.predict(features[FEATURES].to_numpy(dtype=np.float32),raw_score=True,num_threads=1) if len(features) else np.zeros(0)
 if not np.isfinite(correction).all():raise ValueError('Nonfiniteadaptercorrection')
 out.loc[positions,'adapter_correction']=correction
 out.loc[positions,'probability']=corrected_probability(base,correction)
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=['build','fit','rescore','early-evaluate'])
 for key in ['pairs','manifest','queries','route_plan','output','feature_dir','early_feature_dir','train_truth','early_truth','model_dir','decision_config']:
  p.add_argument('--'+key.replace('_','-'),type=Path)
 p.add_argument('--index-prefix',default='models/index_train');p.add_argument('--threads',type=int,default=1)
 a=p.parse_args();start=time.perf_counter()
 if not 1<=a.threads<=16:raise ValueError('CPUallocation1..16')
 pa.set_cpu_count(a.threads)
 if a.output is None or a.output.exists(): raise ValueError('Require unused output directory')
 a.output.mkdir(parents=True)
 if a.command=='build':
  if a.train_truth or a.early_truth: raise ValueError('No labels allowed in rawsupport building')
  frame,m=sibling.load_pairs(a.pairs,a.manifest)
  route,pinned=sibling.load_pinned_route(frame,m,a.route_plan) if a.route_plan else (sibling.select_route(frame,universe_ids=m['reference_ids']),None)
  raw=json.loads(a.queries.read_text());wanted={frame.source1_entity_id.iat[i] for i in route}
  queries={r['entity_id']:sibling.clean_target(normalize_name(r['business_name']),normalize_address(r['business_address']),'S1') for r in raw if r['entity_id'] in wanted}
  targets=sibling.load_targets({frame.candidate_entity_id.iat[j] for i,peers in route.items() for j in [i,*peers]},a.index_prefix)
  data=raw_features(frame,route,queries,targets);data.to_parquet(a.output/'features.parquet',index=False)
  sibling.write_json(a.output/'manifest.json',{'status':'complete','version':VERSION,'features':FEATURES,'feature_pair_order_sha256':sibling.pair_order_hash(data),'features_sha256':sibling.digest(a.output/'features.parquet'),'input_pairs_sha256':m['pairs_sha256'],'input_pair_order_sha256':m['pair_order_sha256'],'input_manifest_sha256':sibling.digest(a.manifest),'queries_sha256':sibling.digest(a.queries),'route_plan_sha256':sibling.digest(a.route_plan/'manifest.json') if a.route_plan else None,'reference_ids':m['reference_ids'],'role':m.get('role'),'split':m['split'],'labels_read':False,'target_source':'frozen_normalized_records_index','raw_targets':False,'query_source':'capture_raw_references_normalize_name_and_address','routed_references':len(wanted),'routed_rows':len(data),'seconds':time.perf_counter()-start,'source_sha256':sibling.digest(__file__)})
  return
 if a.command=='fit':
  import lightgbm as lgb
  train,tm=load_features(a.feature_dir);early,em=load_features(a.early_feature_dir)
  tt=json.loads(a.train_truth.read_text());et=json.loads(a.early_truth.read_text())
  if tm['role']!='residual_train' or set(tt)!=set(tm['reference_ids']) or len(tt)!=20000 or len(et)!=10000 or set(tt)&set(et) or not set(et)<=set(em['reference_ids']): raise ValueError('Require disjoint authorized20kfit+10kearlyonly')
  train=train[train.source1_entity_id.isin(tt)];early=early[early.source1_entity_id.isin(et)]
  def dataset(data,truth):
   labels=np.array([int(c in truth[e]) for e,c in data[sibling.KEYS].itertuples(index=False,name=None)])
   if len(set(labels))!=2:raise ValueError('Both real classes required')
   weights=1/data.groupby('source1_entity_id').source1_entity_id.transform('size').to_numpy()
   return lgb.Dataset(data[FEATURES].to_numpy(dtype=np.float32),label=labels,weight=weights,init_score=base_logit(data.probability.to_numpy()),feature_name=FEATURES)
  fit=dataset(train,tt);validation=dataset(early,et)
  model=lgb.train(dict(objective='binary',metric='binary_logloss',num_leaves=15,min_data_in_leaf=60,lambda_l2=10.,learning_rate=.03,max_bin=127,num_threads=a.threads,deterministic=True,force_col_wise=True,seed=42,verbosity=-1,boost_from_average=False),fit,num_boost_round=400,valid_sets=[validation],valid_names=['early10k'],callbacks=[lgb.early_stopping(40,verbose=False)])
  model.save_model(str(a.output/'adapter.txt'))
  sibling.write_json(a.output/'manifest.json',{'status':'complete','version':VERSION,'features':FEATURES,'model_sha256':sibling.digest(a.output/'adapter.txt'),'training_owner_ids':sorted(tt),'early_owner_ids':sorted(et),'training_routed_pairs':len(train),'early_routed_pairs':len(early),'best_iteration':model.best_iteration,'early_logloss':model.best_score['early10k']['binary_logloss'],'base_probability_clip':CLIP,'inference':'sigmoid(logit(clipped frozenp2)+rawmodelpredict); additivebase exactlyonce','adapter_strength':1.,'train_feature_sha256':tm['features_sha256'],'early_feature_sha256':em['features_sha256'],'train_truth_sha256':sibling.digest(a.train_truth),'early_truth_sha256':sibling.digest(a.early_truth),'source_sha256':sibling.digest(__file__),'seconds':time.perf_counter()-start})
  return
 import lightgbm as lgb
 frame,m=sibling.load_pairs(a.pairs,a.manifest);data,fm=load_features(a.feature_dir)
 mm=json.loads((a.model_dir/'manifest.json').read_text());model_path=a.model_dir/'adapter.txt'
 if mm.get('version')!=VERSION or mm.get('base_probability_clip')!=CLIP or mm.get('source_sha256')!=sibling.digest(__file__) or sibling.digest(model_path)!=mm['model_sha256'] or mm['features']!=FEATURES:raise ValueError('Adapterchanged')
 model=lgb.Booster(model_file=str(model_path))
 if model.feature_name()!=FEATURES:raise ValueError('Adapter featureorder differs')
 if fm['input_pairs_sha256']!=m['pairs_sha256'] or fm['input_pair_order_sha256']!=m['pair_order_sha256']:raise ValueError('Wrong fullfeatureuniverse')
 changed=score_frame(frame,data,model)
 if a.command=='rescore':
  if a.train_truth or a.early_truth:raise ValueError('Inference forbidslabels')
  changed.to_parquet(a.output/'pairs.parquet',index=False)
  sibling.write_json(a.output/'manifest.json',{**m,'status':'scored_requires_global_decision_replay','adapter_model_sha256':mm['model_sha256'],'adapter_strength':1.,'pairs_sha256':sibling.digest(a.output/'pairs.parquet'),'raw_feature_manifest_sha256':sibling.digest(a.feature_dir/'manifest.json'),'seconds':time.perf_counter()-start})
 else:
  from run_frozen_pipeline import decide
  truth=json.loads(a.early_truth.read_text());ids=set(truth)
  if ids!=set(mm['early_owner_ids']) or len(ids)!=10000:raise ValueError('Onlypredeclared early10k maybeevaluated')
  config=json.loads(a.decision_config.read_text());control,_=decide(frame,frame,config);trial,_=decide(changed,changed,config)
  selected=frame.source1_entity_id.isin(ids).to_numpy()
  before=sibling.entity_values(frame[selected],control[selected],truth);after=sibling.entity_values(changed[selected],trial[selected],truth)
  report={'status':'early10k_pilot_only','claimant_references':len(m['reference_ids']),'evaluation_references':len(ids),'baseline_macro_f05':float(np.mean([before[e] for e in ids])),'adapter_macro_f05':float(np.mean([after[e] for e in ids])),'gain':float(np.mean([after[e]-before[e] for e in ids])),'adapter_strength':1.,'selected20k_metric_inspected':False,'evidence_scope':'Early10k stopping/tuning evidence; not fresh confirmation/audit','model_sha256':mm['model_sha256'],'seconds':time.perf_counter()-start}
  sibling.write_json(a.output/'report.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
