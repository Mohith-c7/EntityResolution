"""
Model training, inference, and threshold calibration subpackage.
"""
from src.model.predict import predict, select_matches, write_outputs
from src.model.threshold import sweep_threshold
from src.model.train import (
    build_training_dataset,
    entity_level_split,
    load_feature_schema,
    load_model,
    sample_hard_negatives,
    save_feature_schema,
    save_model,
    train,
)

__all__ = [
    "train",
    "build_training_dataset",
    "sample_hard_negatives",
    "entity_level_split",
    "save_model",
    "load_model",
    "save_feature_schema",
    "load_feature_schema",
    "predict",
    "select_matches",
    "write_outputs",
    "sweep_threshold",
]
