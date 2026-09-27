"""Bind the locked hybrid and label-free heldout inputs before root OPEN."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from preflight import GROUPS, POLICY, POST_POLICY, read, sha, verify_reservation

ROOT = Path(__file__).resolve().parents[3]
BASE = Path("reports/final_2h/neural_last4_release_freeze_v3.json")
LANE = Path("research/final_2h/hybrid_audit")
CRITERIA = {"minimum_point_macro_gain": .0003,
            "paired_ci_lower_bound_strictly_above": 0.0,
            "country_macro_delta_floor": -.0005,
            "precision_singletons_diagnostic_only": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exporter-sha256", required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    reservation = LANE.parent / "extension_audit_last8_v1"
    proof = verify_reservation(ROOT, reservation)
    source = read("models/sprint_6h/heldout_cohort/manifest.json")
    owners = set(source["reference_ids"])
    fitted = set()
    model_files = ["research/final_2h/neural_last8_fine_v1/adapter.txt",
                   "research/final_2h/neural_last8_fine_v1/report.json",
                   "models/final_2h/neural_last8_continue_v1/manifest.json"]
    for name in ("neural_last4_fine_v1", "neural_last8_fine_v1"):
        fitted.update(read(f"research/final_2h/{name}/report.json")["fit_owners"])
    for path in ("models/sprint_6h/neural/frozen_head120k_v1/manifest.json",
                 "models/final_2h/neural_last4_continue_v1/manifest.json",
                 "models/final_2h/neural_last8_continue_v1/manifest.json"):
        fitted.update(read(path)["training_owners"])
    if owners & fitted: raise ValueError("Fitted owner entered complete heldout graph")
    directory = LANE / "heldout_scored_v1"
    universe = directory / (a.output.stem + "_universe.json")
    universe.write_text(json.dumps({"kind": "complete_heldout", "complete": True,
        "supervised_training_excluded": True, "scoring_complete": True,
        "declared_references": 331012, "scored_references": 331012,
        "declared_pairs": 13240480, "scored_pairs": 13240480, "pairs": 13240480,
        "training_owner_intersection": 0, "labels_read": False}, indent=2) + "\n")
    frozen = read(BASE)
    # Preserve inherited immutable runtime dependencies, then bind new helpers.
    for group in GROUPS:
        for filename, expected in frozen[group].items():
            if sha(filename) != expected: raise ValueError("Inherited artifact changed: " + filename)
    exporter = "scripts/build_hybrid_submission.py"
    if sha(exporter) != a.exporter_sha256: raise ValueError("Wrong final exporter bytes")
    for filename in [*map(str, LANE.glob("*.py")), exporter,
                     "research/final_2h/prepare_hybrid_test_rescue.py",
                     "research/final_2h/verify_hybrid_test_rescue.py",
                     "research/final_2h/post_selection_exclusivity.py",
                     "research/final_2h/evaluate_fixed_hybrid.py",
                     "research/final_2h/hybrid_decision_probe.py",
                     "research/final_2h/prepare_hybrid_decision_release.py",
                     "research/final_2h/score_full_test_neural.py",
                     "research/final_2h/neural_layers8.py"]:
        frozen["code_sha256"][filename] = sha(filename)
    for filename in model_files: frozen["model_sha256"][filename] = sha(filename)
    for filename in ("models/index_train_S2.sqlite", "models/index_train_S3.sqlite"):
        frozen["index_sha256"][filename] = sha(filename)
    schemas = [str(reservation / "plan.json"), str(directory / "manifest.json"), str(universe),
               "research/final_2h/hybrid_decision_release_v1/protocol.json",
               "research/final_2h/hybrid_decision_release_v1/report.json",
               "research/final_2h/hybrid_decision_probe_v1/protocol.json",
               "research/final_2h/hybrid_decision_probe_v1/chosen.json",
               "models/sprint_6h/neural/last8_heldout331k/manifest.json",
               "research/sprint_6h/sibling/neural_v2_features_heldout331k/manifest.json"]
    extra = LANE / "heldout_empty_extra_v1"
    schemas += [str(extra / f) for f in ("manifest.json", "route_manifest.json", "neural_inputs/manifest.json",
                                         "neural_old/manifest.json", "neural_fine/manifest.json")]
    for filename in schemas: frozen["schema_sha256"][filename] = sha(filename)
    actual_scorers = [("models/sprint_6h/neural/last8_heldout331k/manifest.json", "research/final_2h/neural_layers.py"),
                      (extra / "neural_fine/manifest.json", "research/final_2h/neural_layers.py"),
                      (extra / "neural_old/manifest.json", "research/sprint_6h/neural_frozen_head.py")]
    for manifest, code in actual_scorers:
        if read(manifest)["code_sha256"] != sha(code): raise ValueError("Actual neural scorer source mismatch")
    files = {"original": Path("models/sprint_6h/heldout_cohort/pairs.parquet"),
             "original_manifest": Path("models/sprint_6h/heldout_cohort/manifest.json"),
             "baseline": Path("research/final_2h/neural_last4_fine_scored_heldout331k/pairs.parquet"),
             "candidate": directory / "pairs.parquet", "main_route": directory / "main_route.parquet",
             "rescue_route": directory / "rescue_route.parquet", "universe": universe}
    frozen.update({"frozen_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "hybrid8_empty4_calibrated_exclusive_v1", "policy": "hybrid_original_p2_anchored",
        "hybrid_policy": POLICY, "candidate_post_selection_policy": POST_POLICY,
        "baseline_release_freeze_sha256": sha(BASE), "baseline_frozen_sha256": sha(BASE),
        "baseline_release_freeze_path": str(BASE), "original_frozen_sha256": source["frozen_sha256"],
        "parent_frozen_sha256": source["frozen_sha256"],
        "baseline_decision_config": read(BASE)["decision_config"],
        "decision_config": {"threshold": .83, "t_rest": .79, "t_first": .6999999999999998},
        "split_ledger_sha256": proof["reservation_plan_sha256"],
        "reservation_plan_sha256": proof["reservation_plan_sha256"],
        "audit_sampled_ids_sha256": proof["references_sha256"],
        "audit_reference_path": str(reservation / "references.json"),
        "extension_acceptance_criteria": CRITERIA,
        "prediction_input_sha256": {k: sha(v) for k, v in files.items()},
        "prediction_input_paths": {k: str(v) for k, v in files.items()},
        "fresh_confirmation_labels_opened": False, "fresh_audit_labels_opened": False,
        "extension_audit_labels_opened": False, "labels_read_by_freeze": False,
        "full_test_submission_started": False, "runtime_not_yet_verified": True,
        "head8_checkpoint_declared_sha256": read(model_files[2])["checkpoint_files"],
        "confirmation_policy": "Fresh extension audit only; no new confirmation cohort.",
        "selected_evidence": {"path": "research/final_2h/hybrid_decision_release_v1/report.json",
                              "sha256": sha("research/final_2h/hybrid_decision_release_v1/report.json")}})
    # Remove claims pertaining only to the old validation-only amendment.
    for key in ("audit_freeze", "audit_freeze_sha256", "audit_report", "runtime_evidence", "amendment",
                "release_validation_proof", "supersedes_freeze_sha256", "audit_completed_under_original_frozen_scoring",
                "scoring_function_ast_sha256", "scoring_ast_unchanged", "release_version"):
        frozen.pop(key, None)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(frozen, indent=2) + "\n")
    promotion = a.output.parent / (a.output.stem + "_promotion.json")
    if promotion.exists(): raise FileExistsError(promotion)
    promotion.write_text(json.dumps({"status": "root_promoted_hybrid_to_extension_audit",
        "authorized_by": "root", "candidate_frozen_sha256": sha(a.output),
        "reservation_plan_sha256": proof["reservation_plan_sha256"], "extension_labels_read": False,
        "authorization": "Root directed exact locked freeze/preflight/sealing; OPEN remains separate."}, indent=2) + "\n")
    print(json.dumps({"freeze": str(a.output), "sha256": sha(a.output), "promotion": str(promotion),
                      "universe_sha256": sha(universe), "labels_read": False}, indent=2))


if __name__ == "__main__": main()
