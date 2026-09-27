import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("prepare_cpu_scale", HERE / "prepare_cpu_scale.py")
scale = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scale
SPEC.loader.exec_module(scale)


def record(entity):
    return {"entity_id": entity, "business_name": entity, "business_address": "1 Road", "country": "US"}


def test_fixed_reference_windows_and_hashed_training_are_order_independent():
    reservation = {
        "train": [record(f"S1-t{number:02d}") for number in range(8)],
        "tune": [record(f"S1-v{number:05d}") for number in range(50_000)],
    }
    first = scale.choose_reference_folds(reservation, 5)
    reversed_input = {key: list(reversed(value)) for key, value in reservation.items()}
    second = scale.choose_reference_folds(reversed_input, 5)
    assert [row["entity_id"] for row in first["train"]] == [row["entity_id"] for row in second["train"]]
    assert len(first["early_stop"]) == 10_000
    assert len(first["selection"]) == 20_000
    assert not ({row["entity_id"] for row in first["early_stop"]} &
                {row["entity_id"] for row in first["selection"]})


def test_forbidden_audit_holdout_and_test_paths_are_rejected():
    for value in ("models/run/audit/cache", "cache_holdout", "dataset/test/source.tsv"):
        with pytest.raises(ValueError, match="Forbidden"):
            scale.reject_forbidden_input(Path(value))
    scale.reject_forbidden_input(Path("models/scale_v1_run/cache_tune"))


def test_chunk_keeps_all_pairs_and_assigns_equal_entity_weight():
    base = {name: np.zeros(3, dtype=np.float32) for name in scale.FEATURE_NAMES_65}
    frame = pd.DataFrame({
        **base,
        "source1_entity_id": ["S1-a", "S1-a", "S1-b"],
        "candidate_entity_id": ["S2-x", "S2-z", "S2-y"],
        "label": [1, 0, 1],
    })
    frame["blocking_score"] = [.9, .4, .8]
    metadata = [
        {"entity_id": "S1-a", "retained_pairs": 2, "true_ids": ["S2-x"]},
        {"entity_id": "S1-b", "retained_pairs": 1, "true_ids": ["S2-y"]},
    ]
    references = {
        "S1-a": scale.RawRecord("Blue LLC", "12A Road", "US"),
        "S1-b": scale.RawRecord("Red Inc.", "9 Road", "US"),
    }
    targets = {
        "S2-x": scale.RawRecord("Blue", "12A Road", "US"),
        "S2-z": scale.RawRecord("Other", "44 Road", "US"),
        "S2-y": scale.RawRecord("Red", "9 Road", "US"),
    }
    result = scale.prepare_chunk(frame, metadata, references, targets, "synthetic")
    assert len(result) == 3
    assert result.sample_weight.tolist() == pytest.approx([.5, .5, 1.0])
    assert list(result.columns) == scale.OUTPUT_COLUMNS
    assert np.isfinite(result[list(scale.NEW_FEATURE_NAMES)].to_numpy()).all()


def test_chunk_rejects_label_and_pair_count_mismatches():
    frame = pd.DataFrame({**{name: [0.0] for name in scale.FEATURE_NAMES_65},
                          "source1_entity_id": ["S1-a"], "candidate_entity_id": ["S2-x"],
                          "label": [0]})
    metadata = [{"entity_id": "S1-a", "retained_pairs": 1, "true_ids": ["S2-x"]}]
    refs = {"S1-a": scale.RawRecord("A", "1", "US")}
    targets = {"S2-x": scale.RawRecord("A", "1", "US")}
    with pytest.raises(ValueError, match="labels disagree"):
        scale.prepare_chunk(frame, metadata, refs, targets, "synthetic")
    metadata[0]["retained_pairs"] = 2
    with pytest.raises(ValueError, match="Pair-count mismatch"):
        scale.prepare_chunk(frame, metadata, refs, targets, "synthetic")


def test_streaming_parquet_combination_preserves_order_and_rows(tmp_path):
    paths = []
    for number, values in enumerate(([1, 2], [3])):
        path = tmp_path / f"{number}.parquet"
        pd.DataFrame({"value": values}).to_parquet(path, index=False)
        paths.append(path)
    output = tmp_path / "combined.parquet"
    assert scale.combine_parquet_chunks(paths, output) == 3
    assert pd.read_parquet(output).value.tolist() == [1, 2, 3]
