"""CPU LightGBM baseline trained only on provided challenge records."""

import lightgbm as lgb
import numpy as np

from ..features.registry import FEATURE_NAMES


def train(train_pairs, tune_pairs, *, threads: int = 4, seed: int = 42, feature_names=FEATURE_NAMES, overrides=None):
    if not len(train_pairs) or train_pairs["label"].nunique() < 2:
        raise ValueError("Training candidates must include positive and negative examples")
    if not len(tune_pairs) or tune_pairs["label"].nunique() < 2:
        raise ValueError("Tuning candidates must include positive and negative examples")
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=1200, learning_rate=.04, num_leaves=31,
        min_child_samples=30, reg_lambda=2.0, colsample_bytree=.90,
        subsample=.90, subsample_freq=1, max_bin=127,
        random_state=seed, n_jobs=threads, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    if overrides:
        allowed={"n_estimators","learning_rate","num_leaves","min_child_samples","reg_lambda"}
        if set(overrides)-allowed:
            raise ValueError(f"Unsupported model experiment parameters: {sorted(set(overrides)-allowed)}")
        model.set_params(**overrides)
    train_x = train_pairs.loc[:, list(feature_names)].astype(np.float32)
    tune_x = tune_pairs.loc[:, list(feature_names)].astype(np.float32)
    # Each anchor contributes total weight one, regardless of candidate count.
    weights = 1.0 / train_pairs.groupby("source1_entity_id")["label"].transform("count")
    tune_weights = 1.0 / tune_pairs.groupby("source1_entity_id")["label"].transform("count")
    model.fit(train_x, train_pairs["label"], sample_weight=weights,
        eval_set=[(tune_x, tune_pairs["label"])], eval_sample_weight=[tune_weights],
        eval_metric="binary_logloss", callbacks=[lgb.early_stopping(75, verbose=False)])
    return model
