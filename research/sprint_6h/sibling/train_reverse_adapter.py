"""One keyed reverse-S1 pilot; fixed36inputs, anchored frozenp2, no rivalIDs."""
import argparse
import hashlib
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
VERSION='reverse-s1-anchored-v1-36'
REVERSE_FEATURES=['reverse_own_name_sort','reverse_own_phonetic_sort','reverse_own_address_sort','reverse_own_number_jaccard','reverse_own_number_contradiction','reverse_own_joint','reverse_other_name_sort','reverse_other_phonetic_sort','reverse_other_address_sort','reverse_other_number_jaccard','reverse_other_number_contradiction','reverse_other_joint','reverse_margin_name','reverse_margin_phonetic','reverse_margin_address','reverse_margin_number','reverse_margin_joint','reverse_other_second_joint','reverse_other_third_joint','reverse_best_second_joint_gap','reverse_name_terms_used','reverse_phonetic_terms_used','reverse_address_terms_used','reverse_query_fields_used','reverse_high_df_terms_skipped','reverse_shortlist_saturated','reverse_own_in_top8','reverse_no_other_returned','reverse_target_name_missing','reverse_target_address_missing','reverse_target_numbers_missing','reverse_target_country_missing','reverse_own_country_conflict']
FEATURES=[*REVERSE_FEATURES,'first_stage','probability','candidate_rank']
assert len(REVERSE_FEATURES)==33 and len(FEATURES)==36
CLIP=1e-5
QUERY_CONFIG={'country_df_fraction':.02,'max_df':20000,'terms_per_field':2,'field_top_k':16,'joint_top_k':32,'rerank_top_k':8,'rivals':3}
EXPECTED_COUNTS={'train':2206821,'test':1732544}

def digest(path):return sibling.digest(path)
def base_logit(probability):
 p=np.clip(np.asarray(probability,dtype=np.float64),CLIP,1-CLIP)
 if not np.isfinite(p).all():raise ValueError('Nonfinite frozenbase')
 return np.log(p)-np.log1p(-p)
def corrected_probability(base,correction):
 correction=np.asarray(correction,dtype=np.float64);margin=base_logit(base)+correction
 if not np.isfinite(correction).all() or not np.isfinite(margin).all():raise ValueError('Nonfinite correction')
 return np.where(correction==0,np.asarray(base,dtype=np.float64),1/(1+np.exp(-np.clip(margin,-50,50))))

def frozen_p2_rank(frame):
 """One-based rank among ALLforward candidates, tie by ascending targetID."""
 ordered=frame.sort_values(['source1_entity_id','probability','candidate_entity_id'],ascending=[True,False,True],kind='stable')
 ranks=ordered.groupby('source1_entity_id',sort=False).cumcount()+1
 return ranks.reindex(frame.index).to_numpy(dtype=np.int32)

def join_reverse(frame,reverse,route):
 """Strict pairkey scatter; only fixed33numeric evidence enters model."""
 if not set([*sibling.KEYS,*REVERSE_FEATURES])<=set(reverse):raise ValueError('Missing fixed reverse features')
 if reverse[sibling.KEYS].isna().any().any() or reverse.duplicated(sibling.KEYS).any():raise ValueError('Duplicate/missing reversekeys')
 if not np.isfinite(reverse[REVERSE_FEATURES].to_numpy(dtype=np.float64)).all():raise ValueError('Nonfinite reversefeatures')
 expected=frame.loc[list(route),sibling.KEYS]
 if set(map(tuple,expected.to_numpy()))!=set(map(tuple,reverse[sibling.KEYS].to_numpy())):raise ValueError('Reversefeatures mustexactlycover routedcandidatekeys')
 keys=pd.MultiIndex.from_frame(frame[sibling.KEYS]);rkeys=pd.MultiIndex.from_frame(reverse[sibling.KEYS]);positions=keys.get_indexer(rkeys)
 if (positions<0).any():raise ValueError('Foreign reversekeys')
 result=reverse[[*sibling.KEYS,*REVERSE_FEATURES]].copy()
 result['first_stage']=frame.first_stage.to_numpy()[positions]
 result['probability']=frame.probability.to_numpy()[positions]
 result['candidate_rank']=frozen_p2_rank(frame)[positions]
 return result.sort_values(sibling.KEYS,kind='stable').reset_index(drop=True)

def validate_reverse_manifest(reverse,path,manifest,source,route_plan):
 """Verify the actual builder schema, completed corpus, and self-exclusion."""
 if manifest.get('status')!='complete' or manifest.get('feature_version')!='reverse-s1-v1' or manifest.get('features')!=REVERSE_FEATURES or digest(path)!=manifest.get('features_sha256') or sibling.pair_order_hash(reverse)!=manifest.get('pair_order_sha256'):raise ValueError('Reversefeatureseal differs')
 if manifest.get('labels_read') is not False or manifest.get('trained_rival_scores_used') is not False or manifest.get('rival_ids_as_features') is not False or manifest.get('own_id_excluded') is not True or manifest.get('audit_labels_read') is not False or manifest.get('test_pseudo_labels') is not False:raise ValueError('Reverse evidence leakage/selfexclusion declaration missing')
 if manifest.get('source_pairs_sha256')!=source['pairs_sha256'] or manifest.get('source_pair_order_sha256')!=source['pair_order_sha256'] or manifest.get('frozen_sha256')!=source.get('frozen_sha256'):raise ValueError('Reverse sourcepairuniverse differs')
 route_path=Path(manifest['route_manifest'])
 if not route_path.exists():route_path=Path(path).parent/'route_manifest.json'
 if route_plan:route_path=route_plan/'manifest.json'
 if digest(route_path)!=manifest.get('route_manifest_sha256'):raise ValueError('Reverse route differs')
 if not route_plan:
  marker=json.loads(route_path.read_text());edge_path=Path(path).parent/'route.parquet'
  if marker.get('status')!='complete' or marker.get('route')!=sibling.asdict(sibling.RouteConfig()) or marker.get('global_pair_order_sha256')!=source['pair_order_sha256'] or digest(edge_path)!=marker.get('route_sha256'):raise ValueError('Residual route seal differs')
 # The copied embedded metadata is an exact canonical copy of index.complete.json.
 index=manifest['index_metadata']
 canonical=(json.dumps(index,indent=2,sort_keys=True)+'\n').encode()
 if hashlib.sha256(canonical).hexdigest()!=manifest.get('index_manifest_sha256'):raise ValueError('Reverseindex metadata seal differs')
 index_path=Path(manifest['index_manifest'])
 if index_path.exists() and digest(index_path)!=manifest['index_manifest_sha256']:raise ValueError('Reverseindex file seal differs')
 split=index.get('fingerprint',{}).get('corpus');expected_split='test' if source['split']=='test' else 'train'
 fp=index.get('fingerprint',{})
 if index.get('complete') is not True or split!=expected_split or index.get('records')!=EXPECTED_COUNTS[expected_split] or index.get('labels_read') is not False or index.get('test_pseudo_labels') is not False or sum(index.get('country_counts',{}).values())!=index['records']:raise ValueError('Incomplete/wrong reverseindex corpus')
 if fp.get('version')!='reverse-s1-v1' or fp.get('normalization')!='shared-normalize/accent-fold/core/phonetic;fts-terms-v1' or not fp.get('source_size') or fp.get('source_sha256')!=manifest.get('source_tsv_sha256',{}).get('S1'):raise ValueError('Reverseindex fingerprint differs')
 if manifest.get('query_config')!=QUERY_CONFIG or index.get('query_config')!=QUERY_CONFIG:raise ValueError('Reverse query/DFpolicy differs')
 expected_df=hashlib.sha256(json.dumps(QUERY_CONFIG,sort_keys=True).encode()).hexdigest()
 if index.get('df_policy_sha256')!=expected_df or any(len(str(index.get(k,'')))!=64 or any(c not in '0123456789abcdef' for c in str(index.get(k,''))) for k in ['index_sha256','build_code_sha256']):raise ValueError('Missing binary/build/DFattestation')
 build_source=ROOT/'scripts/reverse_competition.py'
 if index['build_code_sha256']!=digest(build_source):
  snapshot=ROOT/'research/sprint_6h/reverse_competition/index_build_snapshot.py'
  if not snapshot.exists() or digest(snapshot)!=index['build_code_sha256']:raise ValueError('Missing immutableindexbuildsnapshot')
 sources=manifest.get('source_tsv_sha256',{})
 if set(sources)!={'S1','S2','S3'} or any(len(str(h))!=64 or any(c not in '0123456789abcdef' for c in str(h)) for h in sources.values()):raise ValueError('Missing rawsourceSHA')
 code_root=ROOT/'code/business_entity_resolution/src'
 for field,file in [('feature_code_sha256',ROOT/'scripts/reverse_competition.py'),('normalization_sha256',code_root/'preprocessing/normalize.py'),('phonetic_code_sha256',code_root/'features/evidence.py')]:
  if manifest.get(field)!=digest(file):raise ValueError('Reverse feature/normalizationcode differs')
  if field in ['normalization_sha256','phonetic_code_sha256'] and index.get(field)!=manifest[field]:raise ValueError('Index/feature normalization differs')
 if set(manifest.get('target_index_metadata',{}))!={'S2','S3'} or any(m.get('complete') is not True for m in manifest['target_index_metadata'].values()):raise ValueError('Incomplete targetindex')
 diagnostics=Path(path).parent/'diagnostics.jsonl'
 if digest(diagnostics)!=manifest.get('diagnostics_sha256'):raise ValueError('Reverse diagnostic seal differs')
 expected=set(map(tuple,reverse[sibling.KEYS].to_numpy()));seen=set()
 with diagnostics.open() as stream:
  for line in stream:
   row=json.loads(line);key=tuple(row[n] for n in sibling.KEYS)
   if key in seen or key not in expected or len(row['rival_ids'])>3 or len(row['rival_ids'])!=len(set(row['rival_ids'])) or row[sibling.KEYS[0]] in row['rival_ids']:raise ValueError('Selfrival/duplicate/foreign diagnostics')
   seen.add(key)
 if seen!=expected:raise ValueError('Missing reverse diagnostics')
 return {**index,'split':split}

def load_features(directory):
 m=json.loads((directory/'manifest.json').read_text());path=directory/'features.parquet'
 if m.get('status')!='complete' or m.get('version')!=VERSION or m.get('features')!=FEATURES or m.get('source_sha256')!=digest(__file__) or digest(path)!=m.get('features_sha256'):raise ValueError('Reverseadapterfeatureseal differs')
 data=pd.read_parquet(path)
 if data.duplicated(sibling.KEYS).any() or sibling.pair_order_hash(data)!=m.get('feature_pair_order_sha256') or not np.isfinite(data[FEATURES].to_numpy(dtype=np.float64)).all():raise ValueError('Reverseadapterkeys/nonfinitefeatures')
 return data,m

def score_frame(frame,features,model):
 if not np.isfinite(features[FEATURES].to_numpy(dtype=np.float64)).all():raise ValueError('Nonfinite features')
 keys=pd.MultiIndex.from_frame(frame[sibling.KEYS]);fkeys=pd.MultiIndex.from_frame(features[sibling.KEYS]);positions=keys.get_indexer(fkeys)
 if (positions<0).any() or fkeys.has_duplicates:raise ValueError('Foreign/duplicate adapterkeys')
 base=frame.probability.to_numpy()[positions]
 if not np.array_equal(base,features.probability.to_numpy()) or not np.array_equal(frame.first_stage.to_numpy()[positions],features.first_stage.to_numpy()) or not np.array_equal(frozen_p2_rank(frame)[positions],features.candidate_rank.to_numpy()):raise ValueError('Frozenp1/p2/rank changed')
 delta=model.predict(features[FEATURES].to_numpy(dtype=np.float32),raw_score=True,num_threads=1) if len(features) else np.zeros(0)
 result=frame.copy();result['probability_original']=frame.probability;result['reverse_adapter_correction']=0.
 result.loc[positions,'reverse_adapter_correction']=delta;result.loc[positions,'probability']=corrected_probability(base,delta)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=['join','fit','rescore','early-evaluate'])
 for key in ['pairs','manifest','reverse_features','reverse_manifest','route_plan','output','feature_dir','early_feature_dir','train_truth','early_truth','model_dir','decision_config']:
  p.add_argument('--'+key.replace('_','-'),type=Path)
 p.add_argument('--threads',type=int,default=1)
 a=p.parse_args();start=time.perf_counter()
 if not 1<=a.threads<=16:raise ValueError('CPUallocation1..16')
 pa.set_cpu_count(a.threads)
 if a.output is None or a.output.exists():raise ValueError('Require unused outputdirectory')
 a.output.mkdir(parents=True)
 if a.command=='join':
  if a.train_truth or a.early_truth:raise ValueError('No labels allowedin reversejoin')
  frame,m=sibling.load_pairs(a.pairs,a.manifest)
  route,_=sibling.load_pinned_route(frame,m,a.route_plan) if a.route_plan else (sibling.select_route(frame,universe_ids=m['reference_ids']),None)
  rm=json.loads(a.reverse_manifest.read_text());reverse=pd.read_parquet(a.reverse_features)
  if rm.get('source_manifest_sha256')!=digest(a.manifest):raise ValueError('Reverse sourcecapturemanifest differs')
  index=validate_reverse_manifest(reverse,a.reverse_features,rm,m,a.route_plan)
  if not a.route_plan:
   actual=pd.read_parquet(a.reverse_features.parent/'route.parquet')
   expected=sibling.route_edges(frame,route)
   if set(map(tuple,actual.to_numpy()))!=set(map(tuple,expected.to_numpy())):raise ValueError('Residual route membership differs')
  data=join_reverse(frame,reverse,route);data.to_parquet(a.output/'features.parquet',index=False)
  sibling.write_json(a.output/'manifest.json',{'status':'complete','version':VERSION,'features':FEATURES,'feature_pair_order_sha256':sibling.pair_order_hash(data),'features_sha256':digest(a.output/'features.parquet'),'input_pairs_sha256':m['pairs_sha256'],'input_pair_order_sha256':m['pair_order_sha256'],'input_manifest_sha256':digest(a.manifest),'reverse_manifest_sha256':digest(a.reverse_manifest),'reverse_features_sha256':digest(a.reverse_features),'index_manifest_sha256':rm['index_manifest_sha256'],'index_split':index['split'],'route_plan_sha256':digest(a.route_plan/'manifest.json') if a.route_plan else None,'reference_ids':m['reference_ids'],'role':m.get('role'),'split':m['split'],'labels_read':False,'routed_rows':len(data),'seconds':time.perf_counter()-start,'source_sha256':digest(__file__)})
  return
 if a.command=='fit':
  import lightgbm as lgb
  train,tm=load_features(a.feature_dir);early,em=load_features(a.early_feature_dir)
  tt=json.loads(a.train_truth.read_text());et=json.loads(a.early_truth.read_text())
  if tm['role']!='residual_train' or tm['split']!='development' or em['split']!='development' or set(tt)!=set(tm['reference_ids']) or len(tt)!=20000 or len(et)!=10000 or set(tt)&set(et) or not set(et)<=set(em['reference_ids']):raise ValueError('Require disjoint authorized20kfit+10kearlyonly')
  train=train[train.source1_entity_id.isin(tt)];early=early[early.source1_entity_id.isin(et)]
  def dataset(data,truth):
   labels=np.array([int(c in truth[e]) for e,c in data[sibling.KEYS].itertuples(index=False,name=None)])
   if len(set(labels))!=2:raise ValueError('Bothrealclasses required')
   weights=1/data.groupby('source1_entity_id').source1_entity_id.transform('size').to_numpy()
   return lgb.Dataset(data[FEATURES].to_numpy(dtype=np.float32),label=labels,weight=weights,init_score=base_logit(data.probability.to_numpy()),feature_name=FEATURES)
  fit=dataset(train,tt);validation=dataset(early,et)
  model=lgb.train(dict(objective='binary',metric='binary_logloss',num_leaves=15,min_data_in_leaf=60,lambda_l2=10.,learning_rate=.03,max_bin=127,num_threads=a.threads,deterministic=True,force_col_wise=True,seed=42,verbosity=-1,boost_from_average=False),fit,num_boost_round=400,valid_sets=[validation],valid_names=['early10k'],callbacks=[lgb.early_stopping(40,verbose=False)])
  model.save_model(str(a.output/'adapter.txt'))
  sibling.write_json(a.output/'manifest.json',{'status':'complete','version':VERSION,'features':FEATURES,'model_sha256':digest(a.output/'adapter.txt'),'training_owner_ids':sorted(tt),'early_owner_ids':sorted(et),'training_routed_pairs':len(train),'early_routed_pairs':len(early),'best_iteration':model.best_iteration,'early_logloss':model.best_score['early10k']['binary_logloss'],'base_probability_clip':CLIP,'inference':'sigmoid(logit(clipped frozenp2)+rawmodelpredict); additivebase exactlyonce','adapter_strength':1.,'train_feature_sha256':tm['features_sha256'],'early_feature_sha256':em['features_sha256'],'train_truth_sha256':digest(a.train_truth),'early_truth_sha256':digest(a.early_truth),'source_sha256':digest(__file__),'seconds':time.perf_counter()-start})
  return
 import lightgbm as lgb
 frame,m=sibling.load_pairs(a.pairs,a.manifest);data,fm=load_features(a.feature_dir)
 mm=json.loads((a.model_dir/'manifest.json').read_text());model_path=a.model_dir/'adapter.txt'
 if mm.get('version')!=VERSION or mm.get('base_probability_clip')!=CLIP or mm.get('source_sha256')!=digest(__file__) or digest(model_path)!=mm['model_sha256'] or mm['features']!=FEATURES:raise ValueError('Reverseadapterchanged')
 model=lgb.Booster(model_file=str(model_path))
 if model.feature_name()!=FEATURES:raise ValueError('Adapterfeatureorder differs')
 if fm['input_pairs_sha256']!=m['pairs_sha256'] or fm['input_pair_order_sha256']!=m['pair_order_sha256']:raise ValueError('Wrongfeatureuniverse')
 changed=score_frame(frame,data,model)
 if a.command=='rescore':
  if a.train_truth or a.early_truth:raise ValueError('Inferenceforbidslabels')
  changed.to_parquet(a.output/'pairs.parquet',index=False)
  sibling.write_json(a.output/'manifest.json',{**m,'status':'scored_requires_global_decision_replay','adapter_model_sha256':mm['model_sha256'],'adapter_strength':1.,'pairs_sha256':digest(a.output/'pairs.parquet'),'reverse_feature_manifest_sha256':digest(a.feature_dir/'manifest.json'),'seconds':time.perf_counter()-start})
 else:
  from run_frozen_pipeline import decide
  truth=json.loads(a.early_truth.read_text());ids=set(truth)
  if m['split']!='development' or ids!=set(mm['early_owner_ids']) or len(ids)!=10000:raise ValueError('Onlypredeclared early10k maybeevaluated')
  config=json.loads(a.decision_config.read_text());control,_=decide(frame,frame,config);trial,_=decide(changed,changed,config)
  selected=frame.source1_entity_id.isin(ids).to_numpy()
  before=sibling.entity_values(frame[selected],control[selected],truth);after=sibling.entity_values(changed[selected],trial[selected],truth)
  report={'status':'early10k_reverse_pilot_only','claimant_references':len(m['reference_ids']),'evaluation_references':len(ids),'baseline_macro_f05':float(np.mean([before[e] for e in ids])),'adapter_macro_f05':float(np.mean([after[e] for e in ids])),'gain':float(np.mean([after[e]-before[e] for e in ids])),'adapter_strength':1.,'selected20k_metric_inspected':False,'evidence_scope':'Early10k stopping/tuning evidence; not freshconfirmation/audit','model_sha256':mm['model_sha256'],'seconds':time.perf_counter()-start}
  sibling.write_json(a.output/'report.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
