import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODULE = Path(__file__).with_name("cpu_feature_experiment.py")
SPEC = importlib.util.spec_from_file_location("cpu_feature_experiment", MODULE)
cpu = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cpu
SPEC.loader.exec_module(cpu)


def test_legal_core_preserves_accents_and_understands_dotted_endings():
    left = cpu.legal_name_parts("Société Élégance S.A.")
    right = cpu.legal_name_parts("Société Élégance SA")
    assert left == (("société", "élégance"), "sa", True)
    assert right == (("société", "élégance"), "sa", False)
    features = cpu.pair_features(
        cpu.RawRecord("Société Élégance S.A.", "12 Rue A", "FR"),
        cpu.RawRecord("Société Élégance SA", "12 Rue A", "FR"),
    )
    assert features["unicode_legal_core_exact"] == 1.0
    assert features["dotted_legal_ending_equivalent"] == 1.0


def test_cross_field_recovery_and_numeric_boundaries_are_distinct():
    swapped = cpu.pair_features(
        cpu.RawRecord("Blue Heron", "44 Cedar Road", "US"),
        cpu.RawRecord("Blue", "Heron 44 Cedar Road", "US"),
    )
    assert swapped["cross_name_recovery"] > 0

    same = cpu.pair_features(cpu.RawRecord("Depot 12A", "", "US"),
                             cpu.RawRecord("Depot 12A", "", "US"))
    split = cpu.pair_features(cpu.RawRecord("Depot 12A", "", "US"),
                              cpu.RawRecord("Depot 12 A", "", "US"))
    assert same["numeric_boundary_jaccard"] == 1.0
    assert same["numeric_boundary_sequence_exact"] == 1.0
    assert split["numeric_boundary_jaccard"] < same["numeric_boundary_jaccard"]
    assert split["numeric_boundary_conflict"] == 1.0


def test_alignment_rejects_missing_and_changed_labels():
    pairs = pd.DataFrame({"source1_entity_id": ["S1-a"], "candidate_entity_id": ["S2-a"],
                          "label": [1], "blocking_score": [.8]})
    missing = pd.DataFrame({"source1_entity_id": ["S1-a"], "candidate_entity_id": ["S2-b"],
                            "label": [1], "blocking_score": [.8]})
    with pytest.raises(ValueError, match="Missing cached pair"):
        cpu.align_cache(pairs, missing, "synthetic")
    changed = pairs.copy()
    changed["label"] = 0
    with pytest.raises(ValueError, match="Label mismatch"):
        cpu.align_cache(pairs, changed, "synthetic")


def test_reference_split_is_deterministic_and_disjoint():
    ids = [f"S1-{number}" for number in range(10)]
    first = cpu.deterministic_reference_split(ids, 4, 6, seed=9)
    second = cpu.deterministic_reference_split(reversed(ids), 4, 6, seed=9)
    assert first == second
    assert not set(first[0]) & set(first[1])
    assert set(first[0]) | set(first[1]) == set(ids)


def test_macro_f05_includes_singletons_and_unretrieved_truth():
    frame = pd.DataFrame({"source1_entity_id": ["A", "A"],
                          "candidate_entity_id": ["x", "z"], "label": [1, 0]})
    truth = {"A": ["x", "y"], "B": [], "C": ["never-retrieved"]}
    result, values = cpu.entity_metrics(np.array([.9, .1]), .5, frame, truth)
    assert values == pytest.approx([1.25 / 1.5, 1.0, 0.0])
    assert result["macro_f05"] == pytest.approx((1.25 / 1.5 + 1.0) / 3)
    assert result["entities"] == 3
    assert result["non_singleton_empty_predictions"] == 1


def test_exact_threshold_sweep_finds_selection_optimum():
    frame = pd.DataFrame({"source1_entity_id": ["A", "A", "B"],
                          "candidate_entity_id": ["x", "z", "q"], "label": [1, 0, 0]})
    truth = {"A": ["x"], "B": []}
    result, values = cpu.tune_threshold(np.array([.61, .60, .59]), frame, truth)
    assert result["threshold"] == pytest.approx(.61)
    assert result["macro_f05"] == 1.0
    assert values.tolist() == [1.0, 1.0]
