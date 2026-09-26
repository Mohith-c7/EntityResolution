import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_scale_model", ROOT / "scripts/train_scale_model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_vector_metric_counts_unretrieved_truth_and_zero_candidate_entities():
    # Four references: partial recovery, singleton error, correct empty, missing all true links.
    probabilities = np.array([.9, .8, .1, .85])
    groups = np.array([0, 0, 0, 1])
    labels = np.array([1, 0, 1, 0], dtype=np.uint8)
    expected = np.array([3, 0, 0, 2])
    report, scores = module.score(probabilities, .75, labels, groups, expected)
    np.testing.assert_allclose(scores, [1.25 / 2.75, 0., 1., 0.])
    assert report["macro_f05"] == pytest.approx((1.25 / 2.75 + 1) / 4)
    assert report["micro_precision"] == pytest.approx(1 / 3)
    assert report["micro_recall"] == .2
    assert report["singleton_false_positives"] == 1
    assert report["non_singleton_empty_predictions"] == 1


def test_empty_pair_array_still_scores_all_references():
    report, scores = module.score(np.array([]), .75, np.array([], dtype=np.uint8),
                                 np.array([], dtype=np.int32), np.array([0, 2]))
    np.testing.assert_array_equal(scores, [1., 0.])
    assert report["macro_f05"] == .5
    assert report["predicted_links"] == 0
