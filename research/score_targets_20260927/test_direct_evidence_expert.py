import json

import numpy as np
import pandas as pd
import pytest

import train_direct_evidence_expert as expert


def test_split_includes_empty_entities_and_is_order_independent():
    truth = {f"S1-{i}": [] if i % 2 else [f"S2-{i}"] for i in range(20)}
    left, right = expert.split_entities(truth)
    assert (left, right) == expert.split_entities(dict(reversed(list(truth.items()))))
    assert not left & right
    assert left | right == set(truth)
    frame = pd.DataFrame({"source1_entity_id": list(left)[:2]})
    _, _, kept = expert.subset(frame, truth, left)
    assert set(kept) == left  # References with no candidates remain in the metric.


def test_direct_schema_removes_alias_and_retrieval_channels():
    assert len(expert.ALL_COLUMNS) == 77
    assert len(expert.DIRECT_COLUMNS) == 60
    assert set(expert.DIRECT_COLUMNS).isdisjoint(expert.EXCLUDED)
    assert "candidate_alias_confidence" not in expert.DIRECT_COLUMNS
    assert "blocking_score" not in expert.DIRECT_COLUMNS
    assert set(expert.NEW_FEATURE_NAMES) <= set(expert.DIRECT_COLUMNS)


def test_blend_can_reject_harmful_expert_without_forcing_matches():
    frame = pd.DataFrame({"source1_entity_id": ["A", "A", "B"], "label": [1, 0, 0]})
    truth = {"A": ["x"], "B": []}
    base = np.array([.9, .1, .1])
    bad = np.array([.1, .9, .9])
    selected, grid = expert.choose_blend(base, bad, frame, truth)
    assert selected["expert_weight"] == 0
    assert selected["macro_f05"] == 1
    assert len(grid) == 5
    with pytest.raises(ValueError):
        expert.blend(base, bad[:2], .5)


def test_baseline_requires_same_data_and_predeclared_seed(tmp_path):
    path = tmp_path / "baseline65_plus_cpu12.txt"
    path.write_text("dummy model")
    protocol = {"prepared_manifest_sha256": "same", "parameters": {"seed": 70007}}
    report = {"arms": {expert.BASELINE_ARM: {"model_sha256": expert.digest(path)}}}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert expert.wait_for_baseline(tmp_path, "same") == path
    with pytest.raises(ValueError, match="different prepared dataset"):
        expert.wait_for_baseline(tmp_path, "different")
    protocol["parameters"]["seed"] = 70009
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="predeclared"):
        expert.wait_for_baseline(tmp_path, "same")
