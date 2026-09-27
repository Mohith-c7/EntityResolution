"""The frozen full claim graph must survive the fine-neural correction."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SPEC = importlib.util.spec_from_file_location(
    "score_neural_last4_heldout",
    Path(__file__).resolve().parents[1] / "research/final_2h/score_neural_last4_heldout.py",
)


def scorer():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def test_scatter_keeps_original_pair_order_and_unrouted_scores():
    frame = pd.DataFrame({
        "source1_entity_id": ["A", "A", "B"],
        "candidate_entity_id": ["X", "Y", "Z"],
        "candidate_order": [0, 1, 0],
        "probability": [.8, .6, .3],
        "first_stage": [.9, .7, .4],
    })
    features = pd.DataFrame({
        "source1_entity_id": ["B", "A"],
        "candidate_entity_id": ["Z", "X"],
        "probability": [.3, .8],
        "first_stage": [.4, .9],
        "candidate_rank": [1, 1],
    })
    result = scorer().scatter_corrections(frame, features, np.array([.5, -.5]))
    assert result[["source1_entity_id", "candidate_entity_id", "candidate_order"]].equals(
        frame[["source1_entity_id", "candidate_entity_id", "candidate_order"]]
    )
    assert result.probability_original.tolist() == [.8, .6, .3]
    assert result.probability.iat[1] == .6
    assert result.probability.iat[0] == pytest.approx(1 / (1 + np.exp(-(
        np.log(.8 / .2) - .5
    ))))
    assert result.probability.iat[2] == pytest.approx(1 / (1 + np.exp(-(
        np.log(.3 / .7) + .5
    ))))


def test_scatter_rejects_wrong_frozen_p2_anchor():
    frame = pd.DataFrame({
        "source1_entity_id": ["A"], "candidate_entity_id": ["X"],
        "probability": [.8], "first_stage": [.9],
    })
    features = pd.DataFrame({
        "source1_entity_id": ["A"], "candidate_entity_id": ["X"],
        "probability": [.7], "first_stage": [.9], "candidate_rank": [1],
    })
    with pytest.raises(ValueError, match="anchor"):
        scorer().scatter_corrections(frame, features, np.array([0.]))
