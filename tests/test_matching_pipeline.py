"""Metric, leakage, feature and retrieval contracts for the training baseline."""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))

from src.blocking.contracts import Candidate
from src.blocking.disk_index import DiskSearchConfig, DiskSourceIndex, build_disk_index, normalize_record
from src.evaluation.metrics import entity_f05, score_matches
from src.evaluation.validation import entity_fold, sample_references
from src.features.pairwise_features import build_pair_features
from src.features.registry import FEATURE_NAMES
from src.model.threshold import select_matches, tune_threshold
from src.pipeline.training import build_pairs


def record(eid, name="Cafe Lumiere LLC", address="00123 Rue Mozart 75001", country="France"):
    return normalize_record(dict(entity_id=eid, business_name=name, business_address=address, country=country))


@pytest.mark.parametrize("expected,predicted,score", [
    ([], [], 1), ([], ["S2-A"], 0), (["S2-A"], [], 0),
    (["S2-A", "S3-A"], ["S2-A", "S2-B", "S3-A"], 5/7),
    (["S2-A", "S3-A"], ["S2-A"], 5/6),
])
def test_official_entity_metric(expected, predicted, score):
    assert entity_f05(expected, predicted) == pytest.approx(score)


def test_macro_includes_singletons_and_missing_predictions():
    result = score_matches({"S1-A": set(), "S1-B": {"S2-B"}}, {})
    assert result["macro_f05"] == .5
    with pytest.raises(ValueError):
        score_matches({"S1-A": set()}, {"S1-unknown": []})


def test_features_are_ordered_finite_and_never_use_empty_agreement():
    candidate = Candidate("S1-A", "S2-A", "S2", .5, ("rare_name",))
    features = build_pair_features(record("S1-A"), record("S2-A"), candidate)
    assert tuple(features) == FEATURE_NAMES
    assert len(features) == 36
    assert features["exact_numeric_match"] == 1
    assert features["street_number_match"] == 1
    assert features["country_exact_match"] == 1
    assert features["source_is_s2"] == 1
    assert all(np.isfinite(list(features.values())))
    empty = build_pair_features(record("S1-A", "", "", ""), record("S2-A", "", "", ""), candidate)
    for name in ("exact_name", "exact_name_core", "exact_numeric_match", "country_exact_match", "address_token_set"):
        assert empty[name] == 0
    assert empty["candidate_address_missing"] == 1
    assert empty["postcode_missing"] == 1


def test_threshold_tuning_includes_singleton_cost_and_retains_zero_candidate_entities():
    pairs = pd.DataFrame({"source1_entity_id": ["S1-A", "S1-B"], "candidate_entity_id": ["S2-A", "S2-B"]})
    truth = {"S1-A": {"S2-A"}, "S1-B": set(), "S1-C": set()}
    probabilities = np.array([.9, .7])
    threshold, sweep = tune_threshold(pairs, probabilities, truth)
    predictions = select_matches(pairs, probabilities, threshold, truth)
    assert score_matches(truth, predictions)["macro_f05"] == 1
    assert predictions["S1-C"] == []
    assert len(sweep) >= 66
    with pytest.raises(ValueError):
        select_matches(pairs, probabilities[:1], .5, truth)


def test_sampling_is_independent_of_source_row_order(tmp_path):
    rows = [dict(entity_id=f"S1-{n}",business_name="Business",business_address="00123",country="Newlandia") for n in range(200)]
    def sample(values, name):
        path = tmp_path / f"{name}.tsv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0], delimiter="\t")
            writer.writeheader()
            writer.writerows(values)
        selected, meta = sample_references(path, {"train": 5, "tune": 5, "holdout": 5}, tmp_path/f"{name}_manifest.tsv")
        return {fold: [row["entity_id"] for row in values] for fold,values in selected.items()}
    assert sample(rows, "first") == sample(list(reversed(rows)), "second")


def test_disk_search_preserves_numeric_strings_and_accent_recall(tmp_path):
    source = tmp_path / "targets.tsv"
    source.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-A\tCafé Lumière LLC\t00123 Rue Mozart 75001\tFrance\nS2-B\tOther\t42 Hill Road\tIndia\n")
    path = tmp_path / "targets.sqlite"
    build_disk_index(source, path, "S2", batch_size=1)
    index = DiskSourceIndex(path, DiskSearchConfig(top_k=2,path_top_k=5))
    try:
        results = index.query(record("S1-A"))
        assert results[0][0].candidate_entity_id == "S2-A"
        assert "name_character" in results[0][0].blocking_paths
        assert results[0][1].address_digits == {"00123", "75001"}
        assert build_disk_index(source,path,"S2")["complete"]
        with pytest.raises(ValueError):
            build_disk_index(source,path,"S3")
    finally:
        index.close()


def test_training_excludes_heldout_target_identities_even_as_negative_pairs():
    ref = record("S1-A")
    own = record("S2-own")
    unseen = record("S2-heldout")
    class Index:
        record_count = 2
        def query(self, _):
            return [(Candidate(ref.entity_id,t.entity_id,"S2",.9,("rare_name",),n),t) for n,t in enumerate((own,unseen),1)]
    raw = {"entity_id":ref.entity_id,"business_name":ref.name,"business_address":ref.address,"country":ref.country}
    truth = {ref.entity_id: {own.entity_id}}
    folds = {own.entity_id:"train",unseen.entity_id:"holdout"}
    frame, countries, summary = build_pairs([raw],[Index()],truth,"train",folds)
    assert list(frame["candidate_entity_id"]) == [own.entity_id]
    assert summary["training_pairs_excluded_for_entity_isolation"] == 1
    validation, _, _ = build_pairs([raw],[Index()],truth,"holdout",folds)
    assert set(validation["candidate_entity_id"]) == {own.entity_id,unseen.entity_id}
