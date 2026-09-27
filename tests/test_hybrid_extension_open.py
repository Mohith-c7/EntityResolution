import importlib.util
from pathlib import Path
import sys
import pytest

DIR = Path(__file__).resolve().parents[1] / "research/final_2h/hybrid_audit"
sys.path.insert(0, str(DIR))
spec = importlib.util.spec_from_file_location("hybrid_extension_evaluator", DIR / "evaluate_extension.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_development_promotion_cannot_open_extension_labels():
    proof = {"candidate_frozen_sha256": "candidate", "reservation_plan_sha256": "reservation",
             "prediction_evidence_sha256": "predictions", "runtime_sha256": "runtime"}
    with pytest.raises(ValueError):
        module.verify_open_authorization({"status": "root_promoted_hybrid_to_extension_audit"}, proof)
    record = {"status": "root_authorized_extension_labels_open", "authorized_by": "root",
              "labels_read_before_authorization": False, **proof}
    module.verify_open_authorization(record, proof)
    record["prediction_evidence_sha256"] = "another"
    with pytest.raises(ValueError): module.verify_open_authorization(record, proof)


def test_frozen_prospective_macro_rule_does_not_silently_reuse_old_precision_gate():
    criteria = {"minimum_point_macro_gain": .0003, "paired_ci_lower_bound_strictly_above": 0.,
                "country_macro_delta_floor": -.0005, "precision_singletons_diagnostic_only": True}
    report = {"paired_macro_f05_delta": .0004, "paired_delta_95pct_ci": [.00001, .0008],
              "country_deltas": {"US": .0002, "India": .0005}, "runtime": {"passed": True},
              "acceptance": {"ownership_scope_eligible": True, "promotion_pass": False,
                             "failures": ["point_precision_fell"]}}
    assert module.prospective_acceptance(report, criteria)["passed"]
    report["paired_delta_95pct_ci"][0] = 0.
    assert not module.prospective_acceptance(report, criteria)["passed"]
    with pytest.raises(ValueError): module.prospective_acceptance(report, {**criteria, "minimum_point_macro_gain": 0.})
