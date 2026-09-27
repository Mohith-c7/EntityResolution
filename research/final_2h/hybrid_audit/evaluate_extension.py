"""Root-only, once-only extension label opening after all hybrid gates pass.

Never run without a separate explicit root OPEN authorization. Development
promotion is insufficient. Existing consumed audit files are not used.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(Path(__file__).resolve().parent)]
from preflight import read, ready, sha


def verify_open_authorization(record, proof):
    if (record.get("status") != "root_authorized_extension_labels_open"
            or record.get("authorized_by") != "root"
            or record.get("labels_read_before_authorization") is not False
            or any(record.get(k) != proof[k] for k in (
                "candidate_frozen_sha256", "reservation_plan_sha256",
                "prediction_evidence_sha256", "runtime_sha256"))):
        raise ValueError("A separate explicit root OPEN authorization is required")


def prospective_acceptance(report, criteria):
    required = {"minimum_point_macro_gain": .0003,
                "paired_ci_lower_bound_strictly_above": 0.0,
                "country_macro_delta_floor": -.0005,
                "precision_singletons_diagnostic_only": True}
    if criteria != required:
        raise ValueError("Final extension criteria differ from prospectively declared hybrid rule")
    failures = []
    if report["paired_macro_f05_delta"] < criteria["minimum_point_macro_gain"]:
        failures.append("minimum_macro_gain_not_met")
    if report["paired_delta_95pct_ci"][0] <= criteria["paired_ci_lower_bound_strictly_above"]:
        failures.append("paired_ci_lower_not_positive")
    if min(report["country_deltas"].values()) < criteria["country_macro_delta_floor"]:
        failures.append("country_guard_failed")
    if report["runtime"]["passed"] is not True:
        failures.append("runtime_gate_failed")
    if report["acceptance"]["ownership_scope_eligible"] is not True:
        failures.append("incomplete_heldout_claim_scope")
    return {"passed": not failures, "failures": failures, "criteria": criteria,
            "legacy_precision_singleton_acceptance_is_diagnostic": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("promotion", "open_authorization", "freeze", "reservation", "evidence", "runtime", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--truth-database", type=Path, default=Path("models/scale_v1_plan/aliases/counts.sqlite"))
    p.add_argument("--official-truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    p.add_argument("--iterations", type=int, default=2000)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError("Single-use extension evaluation output already exists")
    proof = ready(ROOT, a.reservation, a.promotion, a.freeze, a.evidence, a.runtime)
    verify_open_authorization(read(a.open_authorization), proof)
    frozen, evidence = read(a.freeze), read(a.evidence)
    own = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if frozen["code_sha256"].get(own) != sha(__file__):
        raise ValueError("Extension evaluator source is not frozen")
    criteria = frozen["extension_acceptance_criteria"]
    # Validate exact criteria before the once-only lock and before any truth.
    prospective_acceptance({"paired_macro_f05_delta": 1, "paired_delta_95pct_ci": [1, 1],
        "country_deltas": {"US": 1, "India": 1}, "runtime": {"passed": True},
        "acceptance": {"ownership_scope_eligible": True}}, criteria)
    ids = read(a.reservation / "entity_ids.json")
    from export_sprint_workbench import reference_rows
    from evaluate_sprint import paired_evaluate
    refs = reference_rows(a.reservation / "references.json")
    if [r["entity_id"] for r in refs] != ids: raise ValueError("Extension raw-reference ID order changed")
    countries = {r["entity_id"]: r["country"] for r in refs}
    baseline, candidate, universe = [read(evidence[name + "_path"]) for name in ("baseline", "candidate", "universe")]
    if set(baseline) != set(ids) or set(candidate) != set(ids):
        raise ValueError("Incomplete reserved predictions, including empty owners")
    lock = a.reservation / "audit_evaluation_started.json"
    with lock.open("x") as stream:
        json.dump({"candidate_frozen_sha256": proof["candidate_frozen_sha256"],
            "references_sha256": proof["references_sha256"], "output": str(a.output.resolve()),
            "root_open_authorization_sha256": sha(a.open_authorization),
            "truth_not_opened_at_start": True, "started_at": datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        stream.write("\n")
    a.output.mkdir(parents=True)
    truth = {e: [] for e in ids}
    with sqlite3.connect(a.truth_database.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        for begin in range(0, len(ids), 500):
            group = ids[begin:begin + 500]; placeholders = ",".join("?" for _ in group)
            folds = conn.execute(f"SELECT id,outer_fold FROM refs WHERE id IN ({placeholders})", group).fetchall()
            if len(folds) != len(group) or any(f != "holdout" for _, f in folds):
                raise ValueError("Reserved audit owner outside outer holdout")
            for target, owner in conn.execute(f"SELECT id,owner FROM owners WHERE owner IN ({placeholders})", group):
                truth[owner].append(target)
    before, seen = sha(a.official_truth), set()
    with a.official_truth.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            owner = row["source1_entity_id"]
            if owner not in truth: continue
            if owner in seen: raise ValueError("Duplicate official reserved truth owner")
            seen.add(owner)
            targets = [t.strip() for t in row["matched_entity_ids"].split(",") if t.strip()]
            if len(targets) != len(set(targets)) or set(targets) != set(truth[owner]):
                raise ValueError("Alias truth differs from official reserved truth")
    if seen != set(ids) or sha(a.official_truth) != before:
        raise ValueError("Official truth coverage/integrity changed")
    truth_path = a.output / "truth.json"
    truth_path.write_text(json.dumps({e: sorted(truth[e]) for e in ids}) + "\n")
    freshness = {**proof, "candidate_preselected": True, "frozen_rules_verified": True,
                 "root_open_authorization_sha256": sha(a.open_authorization)}
    report = paired_evaluate(truth, baseline, candidate, countries, iterations=a.iterations,
        universe=universe, ownership=True, role="final_audit", freshness=freshness, runtime=read(a.runtime))
    report["status"] = "paired_extension_audit_complete"
    report["hybrid_prospective_acceptance"] = prospective_acceptance(report, criteria)
    report["hybrid_prospective_acceptance"]["prediction_evidence_sha256"] = sha(a.evidence)
    report.update({"candidate_frozen_sha256": proof["candidate_frozen_sha256"],
        "baseline_frozen_sha256": proof["baseline_release_freeze_sha256"],
        "reservation_plan_sha256": proof["reservation_plan_sha256"],
        "full_claim_graph": evidence["full_claim_graph"], "source_sha256": sha(__file__),
        "decision_config_sha256": evidence["decision_config_sha256"],
        "baseline_decision_config_sha256": evidence["baseline_decision_config_sha256"],
        "candidate_post_selection_policy": evidence["candidate_post_selection_policy"],
        "input_sha256": {"truth": sha(truth_path), "official_truth": before,
            "candidate": evidence["candidate_sha256"], "baseline": evidence["baseline_sha256"],
            "references": proof["references_sha256"], "universe": evidence["universe_sha256"],
            "prediction_evidence": sha(a.evidence), "runtime": sha(a.runtime), "freeze": sha(a.freeze)}})
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
