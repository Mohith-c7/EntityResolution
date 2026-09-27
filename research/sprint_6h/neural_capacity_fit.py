"""One fixed, less regularized calibration; reuses immutable neural features."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import lightgbm as lgb
import neural_adapter as nn


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['features','early_features','train_truth','early_truth','output']:p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Require unused versioned output')
    train,tm=nn.load_features(a.features);early,em=nn.load_features(a.early_features)
    truth=json.loads(a.train_truth.read_text());et=json.loads(a.early_truth.read_text())
    if len(truth)!=20000 or len(et)!=10000 or set(truth)&set(et) or tm['role']!='residual_train' or not set(et)<=set(em['reference_ids']):raise ValueError('Wrong fit/stopping identities')
    train=train[train.source1_entity_id.isin(truth)];early=early[early.source1_entity_id.isin(et)]
    def dataset(f,labels):
        y=np.array([int(c in labels[e]) for e,c in f[nn.sibling.KEYS].itertuples(index=False,name=None)])
        return lgb.Dataset(f[nn.FEATURES].to_numpy(dtype=np.float32),label=y,
            weight=1/f.groupby('source1_entity_id').source1_entity_id.transform('size').to_numpy(),
            init_score=nn.reverse.base_logit(f.probability),feature_name=nn.FEATURES)
    params=dict(objective='binary',metric='binary_logloss',num_leaves=31,min_data_in_leaf=40,lambda_l2=1.,learning_rate=.05,
        max_bin=127,num_threads=8,deterministic=True,force_col_wise=True,seed=42,verbosity=-1,boost_from_average=False)
    start=time.monotonic()
    model=lgb.train(params,dataset(train,truth),num_boost_round=800,valid_sets=[dataset(early,et)],valid_names=['early10k'],callbacks=[lgb.early_stopping(40,verbose=False)])
    a.output.mkdir(parents=True)
    model.save_model(str(a.output/'adapter.txt'))
    nn.sibling.write_json(a.output/'manifest.json',{'status':'complete','version':nn.VERSION,'features':nn.FEATURES,
        'capacity_variant':'v2_single_predeclared_setting','parameters':params,'max_trees':800,'stopping_patience':40,
        'fit_source_sha256':nn.sibling.digest(__file__),'source_sha256':nn.sibling.digest(nn.__file__),
        'model_sha256':nn.sibling.digest(a.output/'adapter.txt'),'training_owner_ids':sorted(truth),'early_owner_ids':sorted(et),
        'training_routed_pairs':len(train),'early_routed_pairs':len(early),'best_iteration':model.best_iteration,
        'early_logloss':model.best_score['early10k']['binary_logloss'],'adapter_strength':1.,'base_probability_clip':nn.reverse.CLIP,
        'train_feature_sha256':tm['features_sha256'],'early_feature_sha256':em['features_sha256'],'neural_head_manifest_sha256':tm['neural_head_manifest_sha256'],
        'train_truth_sha256':nn.sibling.digest(a.train_truth),'early_truth_sha256':nn.sibling.digest(a.early_truth),'seconds':time.monotonic()-start})


if __name__=='__main__':main()
