"""Fixed37-input anchored calibration of independently trained neural evidence."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent/'sibling'))
import train_reverse_adapter as reverse
sibling=reverse.sibling
FEATURES=[*reverse.FEATURES,'neural_probability']
VERSION='neural-reverse-anchored-v1-37'


def load_features(directory):
    marker=json.loads((directory/'manifest.json').read_text())
    path=directory/'features.parquet'
    if marker.get('status')!='complete' or marker.get('version')!=VERSION or marker['features']!=FEATURES or marker['source_sha256']!=sibling.digest(__file__) or sibling.digest(path)!=marker['features_sha256']:
        raise ValueError('Combined feature seal differs')
    frame=pd.read_parquet(path)
    if frame.duplicated(sibling.KEYS).any() or not np.isfinite(frame[FEATURES]).all().all():raise ValueError('Invalid combined features')
    return frame,marker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['join','fit','rescore'])
    for name in ['features','neural','head_manifest','early_features','train_truth','early_truth','model','pairs','manifest','output']:
        p.add_argument('--'+name.replace('_','-'),type=Path)
    a=p.parse_args();started=time.monotonic()
    if a.output is None or a.output.exists():raise ValueError('Require unused output')
    a.output.mkdir(parents=True)
    if a.command=='join':
        frame,marker=reverse.load_features(a.features)
        nm=json.loads((a.neural/'manifest.json').read_text());hm=json.loads(a.head_manifest.read_text())
        if nm.get('status')!='complete' or nm.get('labels_read') is not False or nm['head_manifest_sha256']!=sibling.digest(a.head_manifest) or sibling.digest(a.neural/'scores.jsonl')!=nm['scores_sha256']:
            raise ValueError('Neural score provenance differs')
        if set(frame.source1_entity_id)&set(hm['training_owners']):raise ValueError('Neural fitted owners entered calibration/evaluation')
        inputs=json.loads(Path(a.neural.parent/'neural_inputs'/a.neural.name/'manifest.json').read_text())
        if inputs['feature_manifest_sha256']!=sibling.digest(a.features/'manifest.json') or inputs['input_sha256']!=nm['input_sha256']:
            raise ValueError('Neural input did not use exactly these routed features')
        neural=pd.read_json(a.neural/'scores.jsonl',lines=True)
        if neural.duplicated(sibling.KEYS).any() or not neural.neural_probability.between(0,1).all() or set(map(tuple,frame[sibling.KEYS].to_numpy()))!=set(map(tuple,neural[sibling.KEYS].to_numpy())):
            raise ValueError('Neural keys/score invalid')
        merged=frame.merge(neural,on=sibling.KEYS,how='left',validate='one_to_one',sort=False)
        merged.to_parquet(a.output/'features.parquet',index=False)
        sibling.write_json(a.output/'manifest.json',{**marker,'version':VERSION,'features':FEATURES,'features_sha256':sibling.digest(a.output/'features.parquet'),
            'reverse_feature_manifest_sha256':sibling.digest(a.features/'manifest.json'),'neural_score_manifest_sha256':sibling.digest(a.neural/'manifest.json'),
            'neural_head_manifest_sha256':sibling.digest(a.head_manifest),'source_sha256':sibling.digest(__file__),'labels_read':False})
        return
    import lightgbm as lgb
    if a.command=='fit':
        train,tm=load_features(a.features);early,em=load_features(a.early_features)
        truth=json.loads(a.train_truth.read_text());et=json.loads(a.early_truth.read_text())
        if len(truth)!=20000 or len(et)!=10000 or set(truth)&set(et) or tm['role']!='residual_train' or not set(et)<=set(em['reference_ids']):raise ValueError('Wrong authorized fit/stopping splits')
        train=train[train.source1_entity_id.isin(truth)];early=early[early.source1_entity_id.isin(et)]
        def dataset(frame,labels):
            y=np.array([int(c in labels[e]) for e,c in frame[sibling.KEYS].itertuples(index=False,name=None)])
            return lgb.Dataset(frame[FEATURES].to_numpy(dtype=np.float32),label=y,
                weight=1/frame.groupby('source1_entity_id').source1_entity_id.transform('size').to_numpy(),
                init_score=reverse.base_logit(frame.probability),feature_name=FEATURES)
        model=lgb.train(dict(objective='binary',metric='binary_logloss',num_leaves=15,min_data_in_leaf=60,lambda_l2=10.,
            learning_rate=.03,max_bin=127,num_threads=8,deterministic=True,force_col_wise=True,seed=42,verbosity=-1,boost_from_average=False),
            dataset(train,truth),num_boost_round=400,valid_sets=[dataset(early,et)],valid_names=['early10k'],callbacks=[lgb.early_stopping(40,verbose=False)])
        model.save_model(str(a.output/'adapter.txt'))
        sibling.write_json(a.output/'manifest.json',{'status':'complete','version':VERSION,'features':FEATURES,'model_sha256':sibling.digest(a.output/'adapter.txt'),
            'training_owner_ids':sorted(truth),'early_owner_ids':sorted(et),'training_routed_pairs':len(train),'early_routed_pairs':len(early),
            'best_iteration':model.best_iteration,'early_logloss':model.best_score['early10k']['binary_logloss'],'adapter_strength':1.,'base_probability_clip':reverse.CLIP,
            'train_feature_sha256':tm['features_sha256'],'early_feature_sha256':em['features_sha256'],'neural_head_manifest_sha256':tm['neural_head_manifest_sha256'],
            'train_truth_sha256':sibling.digest(a.train_truth),'early_truth_sha256':sibling.digest(a.early_truth),'source_sha256':sibling.digest(__file__),'seconds':time.monotonic()-started})
        return
    frame,marker=sibling.load_pairs(a.pairs,a.manifest)
    features,fm=load_features(a.features);mm=json.loads((a.model/'manifest.json').read_text())
    if mm['version']!=VERSION or mm['features']!=FEATURES or mm['source_sha256']!=sibling.digest(__file__) or sibling.digest(a.model/'adapter.txt')!=mm['model_sha256'] or fm['input_pairs_sha256']!=marker['pairs_sha256']:
        raise ValueError('Model/feature/claim source differs')
    model=lgb.Booster(model_file=str(a.model/'adapter.txt'))
    if model.feature_name()!=FEATURES:raise ValueError('Native feature order differs')
    positions=pd.MultiIndex.from_frame(frame[sibling.KEYS]).get_indexer(pd.MultiIndex.from_frame(features[sibling.KEYS]))
    if (positions<0).any() or not np.array_equal(frame.probability.to_numpy()[positions],features.probability.to_numpy()) or not np.array_equal(reverse.frozen_p2_rank(frame)[positions],features.candidate_rank.to_numpy()):raise ValueError('Key/base/rank misalignment')
    correction=model.predict(features[FEATURES].to_numpy(dtype=np.float32),raw_score=True,num_threads=1)
    changed=frame.copy();changed['probability_original']=frame.probability;changed['reverse_adapter_correction']=0.
    changed.loc[positions,'reverse_adapter_correction']=correction;changed.loc[positions,'probability']=reverse.corrected_probability(features.probability,correction)
    changed.to_parquet(a.output/'pairs.parquet',index=False)
    sibling.write_json(a.output/'manifest.json',{**marker,'status':'scored_requires_global_decision_replay','adapter_model_sha256':mm['model_sha256'],
        'adapter_strength':1.,'pairs_sha256':sibling.digest(a.output/'pairs.parquet'),'reverse_feature_manifest_sha256':sibling.digest(a.features/'manifest.json'),
        'neural_head_manifest_sha256':mm['neural_head_manifest_sha256'],'seconds':time.monotonic()-started})


if __name__=='__main__':main()
