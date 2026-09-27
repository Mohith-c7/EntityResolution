import importlib.util
from pathlib import Path
import pytest


P = Path(__file__).resolve().parents[1] / "research/final_2h/hybrid_audit/preflight.py"
spec = importlib.util.spec_from_file_location("hybrid_preflight", P)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def evidence():
    return {"status": "complete_graph_predictions_frozen_no_labels_read", "labels_read": False,
            "candidate_frozen_sha256": "hybrid", "baseline_frozen_sha256": "blue",
            "references_sha256": "refs", "full_claim_graph": {"references": 331012, "pairs": 13240480},
            "hybrid_policy": module.POLICY, "original_p2_anchored_once": True,
            "original_candidate_order_preserved": True, "main_rescue_pair_overlap": 0}


def test_complete_graph_requires_all_rivals_before_slicing():
    e = evidence()
    module.verify_graph_metadata(e, {"references_sha256": "refs"}, "hybrid", "blue")
    e["full_claim_graph"] = {"references": 10000, "pairs": 400000}
    with pytest.raises(ValueError):
        module.verify_graph_metadata(e, {"references_sha256": "refs"}, "hybrid", "blue")


@pytest.mark.parametrize("key,value", [("main_rescue_pair_overlap", 1),
    ("original_p2_anchored_once", False), ("baseline_frozen_sha256", "old04"),
    ("labels_read", True)])
def test_overlap_double_anchor_wrong_control_or_read_labels_fail(key, value):
    e = evidence(); e[key] = value
    with pytest.raises(ValueError):
        module.verify_graph_metadata(e, {"references_sha256": "refs"}, "hybrid", "blue")


def test_explicit_root_promotion_is_bound_to_candidate_and_reservation():
    good = {"status": "root_promoted_hybrid_to_extension_audit", "authorized_by": "root",
            "candidate_frozen_sha256": "hybrid", "reservation_plan_sha256": "plan",
            "extension_labels_read": False}
    module.verify_promotion(good, "hybrid", "plan")
    with pytest.raises(ValueError): module.verify_promotion(good, "another", "plan")
    with pytest.raises(ValueError): module.verify_promotion({}, "hybrid", "plan")


def test_blue_comparator_rules_cannot_follow_candidate_calibration(tmp_path):
    import json
    path = tmp_path / "blue.json"
    original = {"threshold": .83, "t_first": .7, "t_rest": .83}
    path.write_text(json.dumps({"decision_config": original}))
    frozen = {"baseline_release_freeze_path": str(path),
              "baseline_release_freeze_sha256": module.sha(path),
              "baseline_decision_config": original,
              "decision_config": {**original, "t_rest": .8}}
    module.verify_blue_control(tmp_path, frozen)
    frozen["baseline_decision_config"] = frozen["decision_config"]
    with pytest.raises(ValueError): module.verify_blue_control(tmp_path, frozen)
