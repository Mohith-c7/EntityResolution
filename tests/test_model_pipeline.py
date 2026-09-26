"""
Comprehensive unit test suite for pairwise features, model training, prediction,
adapters, threshold sweep, and output formatting.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.legacy.evaluation.metrics import (
    compute_entity_f_beta,
    compute_macro_f05,
    compute_micro_precision_recall,
)
from src.legacy.evaluation.validation import validate_outputs
from src.legacy.features.pairwise_features import (
    FEATURE_NAMES,
    build_feature_matrix,
    build_pair_features,
)
from src.integration.adapters import (
    adapt_candidates,
    adapt_ground_truth,
    adapt_source_records,
)
from src.legacy.model.predict import predict, select_matches, write_outputs
from src.legacy.model.threshold import sweep_threshold
from src.legacy.model.train import (
    build_training_dataset,
    entity_level_split,
    load_feature_schema,
    load_model,
    sample_hard_negatives,
    save_feature_schema,
    save_model,
    train,
)


# ---------------------------------------------------------------------------
# Test Pairwise Features
# ---------------------------------------------------------------------------

def test_feature_vector_dimension_and_types():
    s1 = {"name": "Acme Corp", "address": "123 Main St", "country": "US"}
    cand = {"name": "Acme Corporation", "address": "123 Main Street", "country": "US"}
    
    vec = build_pair_features(s1, cand, blocking_score=0.9, rank=1, source="S1-S2")
    assert len(vec) == 36
    assert len(vec) == len(FEATURE_NAMES)
    assert all(isinstance(v, (int, float)) for v in vec)
    assert not any(np.isnan(v) for v in vec)


def test_digit_features_no_digits_does_not_false_match():
    # If neither record has digits, exact digit match should be 0.0, NOT 1.0!
    s1 = {"name": "Apple Inc", "address": "Infinite Loop", "country": "US"}
    cand = {"name": "Banana Ltd", "address": "Market Street", "country": "US"}
    
    vec = build_pair_features(s1, cand, source="S1-S2")
    # Indices 18: name_digit_exact_match, 20: addr_digit_exact_match, 14: city_postal_exact, 17: postcode_exact
    assert vec[18] == 0.0
    assert vec[20] == 0.0
    assert vec[14] == 0.0
    assert vec[17] == 0.0


def test_digit_features_matching_digits():
    s1 = {"name": "7-Eleven 405", "address": "742 Evergreen Terrace", "country": "US"}
    cand = {"name": "7-Eleven Store 405", "address": "742 Evergreen Terr", "country": "US"}
    
    vec = build_pair_features(s1, cand, source="S1-S2")
    assert vec[18] == 1.0  # name digits match
    assert vec[20] == 1.0  # addr digits match
    assert vec[14] == 1.0  # postal match
    assert vec[22] == 1.0  # name digit jaccard
    assert vec[23] == 1.0  # addr digit jaccard


def test_build_feature_matrix_shapes():
    s1_df = pd.DataFrame([
        {"entity_id": "s1_1", "name": "Alpha LLC", "address": "100 1st Ave", "country": "US"},
        {"entity_id": "s1_2", "name": "Beta Inc", "address": "200 2nd Ave", "country": "US"},
    ]).set_index("entity_id")

    s2_df = pd.DataFrame([
        {"entity_id": "s2_1", "name": "Alpha Co", "address": "100 1st Avenue", "country": "US"},
        {"entity_id": "s2_2", "name": "Beta Corp", "address": "200 2nd St", "country": "US"},
    ]).set_index("entity_id")

    pairs = pd.DataFrame([
        {"s1_id": "s1_1", "s2_id": "s2_1", "source": "S1-S2", "blocking_score": 0.85, "rank": 1},
        {"s1_id": "s1_2", "s2_id": "s2_2", "source": "S1-S2", "blocking_score": 0.70, "rank": 2},
    ])

    source_records = {"S1": s1_df, "S2": s2_df, "S3": pd.DataFrame()}
    mat, names = build_feature_matrix(pairs, source_records)

    assert mat.shape == (2, 36)
    assert names == FEATURE_NAMES


# ---------------------------------------------------------------------------
# Test Adapters
# ---------------------------------------------------------------------------

def test_adapt_candidates():
    harsha_df = pd.DataFrame([
        {"source1_entity_id": "E1", "candidate_entity_id": "E2_A", "candidate_source": "S2", "blocking_score": 0.95},
        {"source1_entity_id": "E1", "candidate_entity_id": "E3_B", "candidate_source": "S3", "blocking_score": 0.80},
    ])
    adapted = adapt_candidates(harsha_df)
    assert list(adapted.columns) == ["s1_id", "s2_id", "s3_id", "blocking_score", "source"]
    assert adapted.iloc[0]["s1_id"] == "E1"
    assert adapted.iloc[0]["s2_id"] == "E2_A"
    assert pd.isna(adapted.iloc[0]["s3_id"]) or adapted.iloc[0]["s3_id"] is None
    assert adapted.iloc[0]["source"] == "S1-S2"

    assert adapted.iloc[1]["s3_id"] == "E3_B"
    assert adapted.iloc[1]["source"] == "S1-S3"


def test_adapt_ground_truth():
    mohit_gt = pd.DataFrame([
        {"source1_entity_id": "E1", "matched_entity_ids": "M1,M2"},
        {"source1_entity_id": "E2", "matched_entity_ids": "M3"},
        {"source1_entity_id": "E3", "matched_entity_ids": ""},  # singleton
    ])
    adapted = adapt_ground_truth(mohit_gt)
    assert list(adapted.columns) == ["s1_id", "match_id"]
    assert len(adapted) == 3  # M1, M2 for E1; M3 for E2; 0 for E3
    assert set(adapted[adapted["s1_id"] == "E1"]["match_id"]) == {"M1", "M2"}


# ---------------------------------------------------------------------------
# Test Evaluation Metrics & Threshold Sweep
# ---------------------------------------------------------------------------

def test_compute_entity_f_beta_singletons():
    # True singleton with no predictions -> 1.0
    assert compute_entity_f_beta(set(), set(), beta=0.5) == 1.0
    # True singleton with false positive prediction -> 0.0
    assert compute_entity_f_beta({"P1"}, set(), beta=0.5) == 0.0


def test_compute_entity_f_beta_matches():
    # Non-singleton with no prediction -> 0.0
    assert compute_entity_f_beta(set(), {"T1"}, beta=0.5) == 0.0
    # Perfect match -> 1.0
    assert compute_entity_f_beta({"T1"}, {"T1"}, beta=0.5) == 1.0
    # Precision 1.0, Recall 0.5 (predicted 1 out of 2 true)
    # F0.5 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = ~0.8333
    score = compute_entity_f_beta({"T1"}, {"T1", "T2"}, beta=0.5)
    assert pytest.approx(score, rel=1e-3) == 5 / 6


def test_sweep_threshold():
    pairs = pd.DataFrame([
        {"s1_id": "E1", "s2_id": "M1", "source": "S1-S2"},
        {"s1_id": "E1", "s2_id": "Wrong1", "source": "S1-S2"},
        {"s1_id": "E2", "s2_id": "M2", "source": "S1-S2"},
    ])
    # M1 and M2 have high probs; Wrong1 has low prob (0.45)
    probs = np.array([0.90, 0.45, 0.85])
    gt = pd.DataFrame([
        {"s1_id": "E1", "match_id": "M1"},
        {"s1_id": "E2", "match_id": "M2"},
    ])

    best_thresh, best_f = sweep_threshold(pairs, probs, gt, min_threshold=0.40, max_threshold=0.80, step=0.05)
    # Threshold >= 0.50 will filter out Wrong1 and keep M1, M2 -> perfect score 1.0
    assert best_thresh >= 0.50
    assert pytest.approx(best_f, rel=1e-3) == 1.0


# ---------------------------------------------------------------------------
# Test Model Training, Serialization & Predictions
# ---------------------------------------------------------------------------

def test_model_train_and_predict_roundtrip():
    np.random.seed(42)
    X_train = np.random.randn(50, 36)
    y_train = np.random.randint(0, 2, size=50).astype(float)
    X_val = np.random.randn(20, 36)
    y_val = np.random.randint(0, 2, size=20).astype(float)

    booster = train(X_train, y_train, X_val, y_val, params={"num_leaves": 7, "min_data_in_leaf": 1})
    preds = predict(booster, X_val)
    assert len(preds) == 20
    assert (preds >= 0.0).all() and (preds <= 1.0).all()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.txt"
        save_model(booster, threshold=0.72, path=str(model_path))

        loaded_booster, loaded_thresh = load_model(str(model_path))
        assert pytest.approx(loaded_thresh) == 0.72

        preds_loaded = predict(loaded_booster, X_val)
        np.testing.assert_allclose(preds, preds_loaded, rtol=1e-5)


def test_select_matches_and_write_outputs():
    pairs = pd.DataFrame([
        {"s1_id": "E1", "s2_id": "M1", "s3_id": None, "source": "S1-S2"},
        {"s1_id": "E1", "s2_id": "M2", "s3_id": None, "source": "S1-S2"},
        {"s1_id": "E2", "s2_id": None, "s3_id": "M3", "source": "S1-S3"},
        {"s1_id": "E2", "s2_id": "Wrong", "s3_id": None, "source": "S1-S2"},
    ])
    probs = np.array([0.90, 0.85, 0.75, 0.40])
    all_s1_ids = ["E1", "E2", "E_singleton"]

    matches = select_matches(pairs, probs, threshold=0.70, all_s1_ids=all_s1_ids)
    assert len(matches) == 3
    
    # E1 should match M1,M2
    e1_row = matches[matches["s1_id"] == "E1"].iloc[0]
    assert set(e1_row["s2_id"].split(",")) == {"M1", "M2"}
    
    # E2 should match M3 only (Wrong was below threshold)
    e2_row = matches[matches["s1_id"] == "E2"].iloc[0]
    assert e2_row["s3_id"] == "M3"
    assert e2_row["s2_id"] == ""

    # E_singleton should have empty matches
    es_row = matches[matches["s1_id"] == "E_singleton"].iloc[0]
    assert es_row["s2_id"] == "" and es_row["s3_id"] == ""

    # Validate output writing
    with tempfile.TemporaryDirectory() as tmpdir:
        write_outputs(matches, pairs, output_dir=tmpdir)
        m_file = Path(tmpdir) / "matching_results.tsv"
        c_file = Path(tmpdir) / "candidate_pairs.tsv"
        assert m_file.is_file()
        assert c_file.is_file()

        m_df = pd.read_csv(m_file, sep="\t", dtype=str)
        c_df = pd.read_csv(c_file, sep="\t", dtype=str)
        assert list(m_df.columns) == ["source1_entity_id", "matched_entity_ids"]
        assert list(c_df.columns) == ["source1_entity_id", "candidate_entity_ids"]
        assert len(m_df) == 3
        assert len(c_df) == 3

        # Validate subset invariant
        assert validate_outputs(matches, pairs)
