"""Matching probabilities with a fixed feature registry."""

import numpy as np

from ..features.registry import FEATURE_NAMES


def predict(model, pairs) -> np.ndarray:
    if not len(pairs):
        return np.empty(0, dtype=float)
    return model.predict_proba(pairs.loc[:, list(model.feature_name_)].astype(np.float32))[:, 1]
