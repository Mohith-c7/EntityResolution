"""
Training pipeline for the business entity resolution model.

Provides:
- entity_level_split()   : entity-aware train/val split (Requirements 2.1)
- sample_hard_negatives(): hard negative sampling at 5:1 ratio (Requirements 2.2)
- build_training_dataset(): assemble feature matrix + labels (Requirements 2.3, 2.4)
- train()                : LightGBM training with seed 42 (Requirements 2.5)
- save_model() / load_model(): model serialization (Requirements 5.1, 5.2)
- save_feature_schema() / load_feature_schema(): schema serialization (Requirements 5.3, 5.4)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.pairwise_features import FEATURE_NAMES, build_feature_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# entity_level_split
# ---------------------------------------------------------------------------

def entity_level_split(
    candidate_pairs: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.2,
    split_manifest_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split candidate_pairs into train and validation partitions such that all
    pairs sharing the same s1_id appear exclusively in one partition.

    If split_manifest_path is provided, loads the official entity-disjoint
    15% validation split from reports/splits/split_manifest.json instead of
    performing a random split.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Must contain an 's1_id' column.
    seed : int
        Random seed for reproducibility (default 42). Used only for fallback split.
    test_size : float
        Fraction of unique S1 IDs to reserve for validation (default 0.2).
        Used only for fallback split when no manifest is provided.
    split_manifest_path : str or None
        Path to the official split manifest JSON. If provided, uses official split.

    Returns
    -------
    (train_df, val_df)

    Requirements: 2.1
    """
    if split_manifest_path is not None:
        import json
        manifest = json.loads(Path(split_manifest_path).read_text())
        train_ids = set(manifest["train_s1_ids"])
        val_ids = set(manifest["val_s1_ids"])
        train_df = candidate_pairs[candidate_pairs["s1_id"].isin(train_ids)].copy()
        val_df = candidate_pairs[candidate_pairs["s1_id"].isin(val_ids)].copy()
        logger.info(
            "Loaded official split manifest: %d train / %d val S1 IDs.",
            len(train_ids), len(val_ids),
        )
        return train_df, val_df

    # Fallback: naive entity-level random split
    unique_ids = candidate_pairs["s1_id"].unique()

    if len(unique_ids) < 2:
        return candidate_pairs.copy(), candidate_pairs.iloc[0:0].copy()

    train_ids_arr, val_ids_arr = train_test_split(
        unique_ids, test_size=test_size, random_state=seed
    )

    train_df = candidate_pairs[candidate_pairs["s1_id"].isin(train_ids_arr)].copy()
    val_df = candidate_pairs[candidate_pairs["s1_id"].isin(val_ids_arr)].copy()

    return train_df, val_df


# ---------------------------------------------------------------------------
# sample_hard_negatives
# ---------------------------------------------------------------------------

def sample_hard_negatives(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    ratio: int = 5,
) -> pd.DataFrame:
    """
    Return hard-negative pairs (non-match candidates) at up to ratio:1 vs positives.

    Hard negatives are candidates NOT in ground_truth, sorted by blocking_score
    descending (highest-confidence near-misses first).

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        All candidate pairs; must contain 's1_id', and either 's2_id' or 's3_id',
        plus 'blocking_score'.
    ground_truth : pd.DataFrame
        True-match pairs; must contain 's1_id' and 'match_id' columns.
    ratio : int
        Hard-negative-to-positive ratio cap (default 5).

    Returns
    -------
    pd.DataFrame of hard-negative rows.

    Requirements: 2.2
    """
    # Build a set of (s1_id, match_id) ground-truth keys
    gt_keys: set[tuple] = set(
        zip(ground_truth["s1_id"], ground_truth["match_id"])
    )

    # Count positives in the candidate set
    def _is_positive(row: pd.Series) -> bool:
        match_id = row.get("s2_id") or row.get("s3_id")
        return (row["s1_id"], match_id) in gt_keys

    mask = candidate_pairs.apply(_is_positive, axis=1)
    positives = candidate_pairs[mask]
    negatives = candidate_pairs[~mask]

    max_negatives = ratio * len(positives)

    hard_negatives = (
        negatives
        .sort_values("blocking_score", ascending=False)
        .head(max_negatives)
    )
    return hard_negatives.copy()


# ---------------------------------------------------------------------------
# build_training_dataset
# ---------------------------------------------------------------------------

def build_training_dataset(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    source_records: dict[str, pd.DataFrame],
    ratio: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine all positive (ground-truth matched) pairs with hard negatives, then
    build the feature matrix and binary label vector.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        All blocked candidate pairs.
    ground_truth : pd.DataFrame
        True-match pairs with 's1_id' and 'match_id'.
    source_records : dict[str, pd.DataFrame]
        Source record DataFrames keyed by "S1", "S2", "S3".
    ratio : int
        Hard-negative-to-positive ratio (default 5).

    Returns
    -------
    (feature_matrix: np.ndarray of shape [N, 36], labels: np.ndarray of shape [N])

    Requirements: 2.3, 2.4
    """
    gt_keys: set[tuple] = set(zip(ground_truth["s1_id"], ground_truth["match_id"]))

    def _is_positive(row: pd.Series) -> bool:
        match_id = row.get("s2_id") or row.get("s3_id")
        return (row["s1_id"], match_id) in gt_keys

    pos_mask = candidate_pairs.apply(_is_positive, axis=1)
    positives = candidate_pairs[pos_mask]
    hard_negs = sample_hard_negatives(candidate_pairs, ground_truth, ratio=ratio)

    combined = pd.concat([positives, hard_negs], ignore_index=True)

    pos_labels = np.ones(len(positives), dtype=np.float64)
    neg_labels = np.zeros(len(hard_negs), dtype=np.float64)
    labels = np.concatenate([pos_labels, neg_labels])

    feature_matrix, _ = build_feature_matrix(combined, source_records)

    return feature_matrix, labels


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    val_matrix: np.ndarray,
    val_labels: np.ndarray,
    params: dict | None = None,
) -> lgb.Booster:
    """
    Train a LightGBM binary classifier with seed 42.

    Parameters
    ----------
    feature_matrix : np.ndarray shape [N, 36]
        Training features.
    labels : np.ndarray shape [N]
        Binary training labels (1 = match, 0 = non-match).
    val_matrix : np.ndarray shape [M, 36]
        Validation features.
    val_labels : np.ndarray shape [M]
        Binary validation labels.
    params : dict or None
        Optional override for LightGBM parameters.

    Returns
    -------
    lgb.Booster

    Requirements: 2.5
    """
    default_params: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "seed": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    train_ds = lgb.Dataset(feature_matrix, label=labels, feature_name=FEATURE_NAMES)
    val_ds = lgb.Dataset(val_matrix, label=val_labels, reference=train_ds)

    callbacks = [lgb.log_evaluation(period=50)]

    booster = lgb.train(
        default_params,
        train_ds,
        num_boost_round=300,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    logger.info("Training complete. Best iteration: %d", booster.best_iteration)
    return booster


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

def save_model(model: lgb.Booster, threshold: float, path: str) -> None:
    """
    Save a trained LightGBM model to disk using its native text format.
    The threshold is saved as a JSON sidecar file alongside the model.

    Requirements: 5.1
    """
    model_path = Path(path)
    model.save_model(str(model_path))

    sidecar_path = model_path.with_suffix(".threshold.json")
    sidecar_path.write_text(json.dumps({"threshold": threshold}))
    logger.info("Model saved to %s, threshold to %s", model_path, sidecar_path)


def load_model(path: str) -> tuple[lgb.Booster, float]:
    """
    Load a LightGBM model and its threshold from disk.

    Returns
    -------
    (booster, threshold)

    Requirements: 5.2
    """
    model_path = Path(path)
    booster = lgb.Booster(model_file=str(model_path))

    sidecar_path = model_path.with_suffix(".threshold.json")
    threshold = json.loads(sidecar_path.read_text())["threshold"]

    logger.info("Model loaded from %s, threshold=%.4f", model_path, threshold)
    return booster, threshold


# ---------------------------------------------------------------------------
# save_feature_schema / load_feature_schema
# ---------------------------------------------------------------------------

def save_feature_schema(feature_names: list[str], defaults: dict, path: str) -> None:
    """
    Write the ordered feature name list and per-feature default values to JSON.

    Requirements: 5.3
    """
    schema = {"feature_names": feature_names, "defaults": defaults}
    Path(path).write_text(json.dumps(schema, indent=2))
    logger.info("Feature schema saved to %s", path)


def load_feature_schema(path: str) -> tuple[list[str], dict]:
    """
    Load the feature schema from a JSON file.

    Returns
    -------
    (feature_names: list[str], defaults: dict)

    Requirements: 5.4
    """
    schema = json.loads(Path(path).read_text())
    return schema["feature_names"], schema["defaults"]
