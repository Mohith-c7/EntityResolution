"""Label-free gates for the separately reserved hybrid extension audit.

This helper deliberately has no truth argument and never exports labels.
Reservation checks may precede promotion; final readiness requires an explicit
root promotion record, a new freeze, complete-graph predictions and runtime.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


GROUPS = ("code_sha256", "model_sha256", "schema_sha256", "index_sha256", "native_sha256")
POLICY = {
    "main": "nn8_original_route",
    "rescue": "fixed4_disjoint_extra",
    "anchor": "original_p2_once",
    "empty_owner_source": "selected4_full_graph",
}
POST_POLICY = "selected_only_probability_then_source1_id"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound_file(root, item):
    path = Path(item["path"])
    path = path if path.is_absolute() else root / path
    if sha(path) != item["sha256"]:
        raise ValueError("Bound artifact changed: " + str(path))
    return path


def verify_reservation(root, directory):
    """Read only IDs, reservation metadata and exposure ledgers; hash raw refs."""
    root, directory = Path(root), Path(directory)
    plan_path = directory / "plan.json"
    plan = read(plan_path)
    if (plan.get("status") != "extension_audit_reserved_no_labels_read"
            or plan.get("kind") != "fresh_extension_audit_only"
            or plan.get("scope_reviewed") is not True
            or plan.get("exactly_one_new_cohort") is not True
            or plan.get("confirmation_cohort_created") is not False
            or plan.get("labels_read") is not False
            or plan.get("ground_truth_read") is not False):
        raise ValueError("Not the separately reserved unopened extension audit")
    for name in ("audit_evaluation_started.json", "labels_opening.json", "truth.json"):
        if (directory / name).exists():
            raise ValueError("Extension audit already opened/started")
    ids = read(directory / "entity_ids.json")
    cohort = plan["cohort_fingerprints"]["audit"]
    if (len(ids) != 10000 or len(set(ids)) != 10000
            or cohort["entities"] != len(ids)
            or hashlib.sha256("\n".join(ids).encode()).hexdigest() != cohort["reference_ids_sha256"]
            or sha(directory / "references.json") != cohort["references_sha256"]):
        raise ValueError("Extension owner IDs or reference bytes changed")
    for ledger in plan["exposure_ledgers"]:
        path = bound_file(root, ledger)
        exposed = read(path)
        if set(ids).intersection(exposed):
            raise ValueError("Extension owners were previously exposed")
    for key in ("original_plan", "original_heldout_manifest", "source1_records"):
        bound_file(root, plan[key])
    if plan["source1_records"]["owners"] != 331012:
        raise ValueError("Wrong complete heldout owner universe")
    return {"reservation_verified": True, "role": "extension_final_audit",
            "reservation_plan_sha256": sha(plan_path),
            "references_sha256": sha(directory / "references.json"),
            "reference_ids_sha256": cohort["reference_ids_sha256"],
            "reserved_at": plan["reserved_at"], "owners": len(ids), "labels_read": False}


def verify_promotion(promotion, freeze_sha, plan_sha):
    if (promotion.get("status") != "root_promoted_hybrid_to_extension_audit"
            or promotion.get("authorized_by") != "root"
            or promotion.get("candidate_frozen_sha256") != freeze_sha
            or promotion.get("reservation_plan_sha256") != plan_sha
            or promotion.get("extension_labels_read") is not False):
        raise ValueError("Explicit root promotion of this frozen hybrid is required")


def verify_graph_metadata(evidence, reservation, freeze_sha, blue_sha):
    if (evidence.get("status") != "complete_graph_predictions_frozen_no_labels_read"
            or evidence.get("labels_read") is not False
            or evidence.get("candidate_frozen_sha256") != freeze_sha
            or evidence.get("baseline_frozen_sha256") != blue_sha
            or evidence.get("references_sha256") != reservation["references_sha256"]
            or evidence.get("full_claim_graph") != {"references": 331012, "pairs": 13240480}
            or evidence.get("hybrid_policy") != POLICY
            or evidence.get("original_p2_anchored_once") is not True
            or evidence.get("original_candidate_order_preserved") is not True
            or evidence.get("main_rescue_pair_overlap") != 0):
        raise ValueError("Incomplete/mismatched complete-claim hybrid prediction evidence")


def verify_blue_control(root, frozen):
    path = Path(frozen.get("baseline_release_freeze_path",
                          "reports/final_2h/neural_last4_release_freeze_v3.json"))
    path = path if path.is_absolute() else Path(root) / path
    if sha(path) != frozen["baseline_release_freeze_sha256"]:
        raise ValueError("Blue05 comparator release descriptor changed")
    if frozen["baseline_decision_config"] != read(path)["decision_config"]:
        raise ValueError("Blue05 comparator thresholds changed")


def verify_own_source(root, frozen, source):
    relative = Path(source).resolve().relative_to(Path(root).resolve()).as_posix()
    if frozen["code_sha256"].get(relative) != sha(source):
        raise ValueError("Hybrid helper source is not frozen: " + relative)


def ready(root, directory, promotion_path, freeze_path, evidence_path, runtime_path):
    root = Path(root)
    reservation = verify_reservation(root, directory)
    frozen = read(freeze_path)
    freeze_sha = sha(freeze_path)
    verify_promotion(read(promotion_path), freeze_sha, reservation["reservation_plan_sha256"])
    if (frozen.get("status") != "candidate_frozen"
            or frozen.get("hybrid_policy") != POLICY
            or frozen.get("candidate_post_selection_policy") != POST_POLICY
            or frozen.get("split_ledger_sha256") != reservation["reservation_plan_sha256"]
            or frozen.get("audit_sampled_ids_sha256") != reservation["references_sha256"]):
        raise ValueError("Hybrid candidate was not frozen against the extension reservation")
    when = datetime.fromisoformat(frozen["frozen_at"].replace("Z", "+00:00"))
    if when.tzinfo is None or when > datetime.now(timezone.utc):
        raise ValueError("Invalid hybrid freeze time")
    for group in GROUPS:
        if not frozen.get(group):
            raise ValueError("Missing hybrid freeze group: " + group)
        for name, expected in frozen[group].items():
            bound_file(root, {"path": name, "sha256": expected})
    verify_own_source(root, frozen, __file__)
    blue_sha = frozen["baseline_release_freeze_sha256"]
    verify_blue_control(root, frozen)
    evidence = read(evidence_path)
    verify_graph_metadata(evidence, reservation, freeze_sha, blue_sha)
    for role in ("baseline", "candidate", "universe"):
        bound_file(root, {"path": evidence[role + "_path"], "sha256": evidence[role + "_sha256"]})
    import sys
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_sprint import check_runtime, canonical_sha
    if evidence["decision_config_sha256"] != canonical_sha(frozen["decision_config"]):
        raise ValueError("Hybrid prediction decisions differ from frozen original rules")
    if evidence.get("baseline_decision_config_sha256") != canonical_sha(frozen["baseline_decision_config"]):
        raise ValueError("Blue05 comparator decision rules changed")
    if evidence.get("candidate_post_selection_policy") != frozen.get("candidate_post_selection_policy"):
        raise ValueError("Candidate post-selection policy differs from freeze")
    runtime = check_runtime(read(runtime_path), freeze_sha)
    if not runtime["passed"]:
        raise ValueError("Hybrid runtime gate failed: " + ",".join(runtime["failures"]))
    return {"status": "hybrid_label_free_gates_passed_labels_still_unopened",
            **reservation, "candidate_frozen_sha256": freeze_sha,
            "baseline_release_freeze_sha256": blue_sha,
            "prediction_evidence_sha256": sha(evidence_path),
            "runtime_sha256": sha(runtime_path), "promotion_sha256": sha(promotion_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("reservation", "ready"))
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument("--reservation", type=Path, required=True)
    for name in ("promotion", "freeze", "evidence", "runtime"):
        p.add_argument("--" + name, type=Path)
    a = p.parse_args()
    if a.mode == "reservation":
        result = verify_reservation(a.root, a.reservation)
    else:
        if any(getattr(a, n) is None for n in ("promotion", "freeze", "evidence", "runtime")):
            p.error("ready requires --promotion --freeze --evidence --runtime")
        result = ready(a.root, a.reservation, a.promotion, a.freeze, a.evidence, a.runtime)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
