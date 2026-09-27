"""Paired entity-level sprint evaluation against a freshly scored control.

Audits require a hash-verified frozen candidate before truth is opened. Global
ownership cannot be promoted on a cheap exact-key rival sample. No historical
score or historical singleton cap enters acceptance.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from decision_policy_v2 import decide_v2, validate_claims
from export_sprint_workbench import sha

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.evaluation.metrics import entity_f05, score_matches


def decide_control(frame, claims, config):
    """Unchanged frozen decide semantics, retained here to avoid scoring imports.

Parity with run_frozen_pipeline.decide is required before using this replay.
"""
    validate_claims(frame); validate_claims(claims)
    threshold, t_first, t_rest = config["threshold"], config["t_first"], config["t_rest"]
    strong = claims[claims.probability >= threshold].sort_values("probability", ascending=False, kind="stable")
    top = strong.groupby("candidate_entity_id", sort=False).head(2)
    best = {}
    for c, s, p in zip(top.candidate_entity_id, top.source1_entity_id, top.probability): best.setdefault(c, []).append((s, p))
    probability = frame.probability.to_numpy(); lost = np.zeros(len(frame), dtype=bool)
    for i in np.flatnonzero(probability >= threshold):
        s, c = frame.source1_entity_id.iat[i], frame.candidate_entity_id.iat[i]
        rival = max((p for owner, p in best.get(c, ()) if owner != s), default=-1.)
        lost[i] = rival > probability[i]
    adjusted = np.where(lost, 0., probability)
    codes, _ = pd.factorize(frame.source1_entity_id); order = np.lexsort((-adjusted, codes))
    first = np.zeros(len(frame), dtype=bool)
    if len(frame): first[order[np.r_[True, codes[order][1:] != codes[order][:-1]]]] = True
    return (adjusted >= t_rest) | (first & (adjusted >= t_first)), lost


def predictions_for(frame, chosen, ids):
    predictions = {e: set() for e in ids}
    for e, target, keep in zip(frame.source1_entity_id, frame.candidate_entity_id, chosen):
        if keep:
            if e not in predictions: raise ValueError("Predictions outside evaluation IDs")
            predictions[e].add(target)
    return predictions


def paired_interval(delta, seed=42, iterations=2000):
    if iterations < 100: raise ValueError("Bootstrap iterations must be at least 100")
    delta = np.asarray(delta, dtype=float)
    if not len(delta): raise ValueError("Empty evaluation")
    rng = np.random.default_rng(seed)
    values = [float(rng.choice(delta, size=len(delta), replace=True).mean()) for _ in range(iterations)]
    return np.quantile(values, [.025, .975]).tolist()


def paired_evaluate(truth, baseline, candidate, countries, seed=42, iterations=2000, universe=None, ownership=False, *, role="development", freshness=None, runtime=None):
    ids = sorted(truth)
    if not ids: raise ValueError("No evaluation entities")
    if set(baseline) != set(ids) or set(candidate) != set(ids) or set(countries) != set(ids):
        raise ValueError("Control, candidate, truth and countries must contain identical entity IDs, including empty predictions")
    for name, mapping in (("truth", truth), ("baseline", baseline), ("candidate", candidate)):
        for e, values in mapping.items():
            if not isinstance(values, (list, tuple, set, frozenset, np.ndarray)) or any(not isinstance(t, str) for t in values):
                raise ValueError(f"{name} values must be target-ID lists/sets, not nested records or strings: {e}")
    truth = {e: set(truth[e]) for e in ids}; baseline = {e: set(baseline[e]) for e in ids}; candidate = {e: set(candidate[e]) for e in ids}
    base = score_matches(truth, baseline, countries); trial = score_matches(truth, candidate, countries)
    delta = np.array([entity_f05(truth[e], candidate[e]) - entity_f05(truth[e], baseline[e]) for e in ids])
    ci = paired_interval(delta, seed, iterations)
    country_delta = {c: float(np.mean([v for e, v in zip(ids, delta) if countries[e] == c])) for c in sorted(set(countries.values()))}
    true_removed = false_removed = true_added = false_added = newly_empty = 0
    for e in ids:
        removed, added = baseline[e] - candidate[e], candidate[e] - baseline[e]
        true_removed += len(removed & truth[e]); false_removed += len(removed - truth[e])
        true_added += len(added & truth[e]); false_added += len(added - truth[e])
        newly_empty += bool(truth[e] and baseline[e] and not candidate[e])
    def owners(predictions):
        result = defaultdict(set)
        for e, targets in predictions.items():
            for target in targets: result[target].add(e)
        return result
    bo, co = owners(baseline), owners(candidate)
    complete = bool(universe and universe.get("kind") == "complete_heldout" and universe.get("complete") is True
                    and universe.get("supervised_training_excluded") is True and universe.get("scoring_complete") is True
                    and universe.get("declared_references", 0) > 0 and universe.get("declared_references") == universe.get("scored_references"))
    failures = []
    if ci[0] <= -.0005: failures.append("paired_lower_bound_not_above_minus_0.0005")
    if trial["micro_precision"] < base["micro_precision"]: failures.append("point_precision_fell")
    if trial["singleton_false_positives"] > base["singleton_false_positives"]: failures.append("singleton_false_predictions_rose")
    if any(v < -.0005 for v in country_delta.values()): failures.append("known_country_point_loss_below_minus_0.0005")
    quality_failures = list(failures)
    if ownership and not complete: failures.append("global_ownership_universe_not_complete_and_heldout")
    if role not in ("development", "confirmation", "final_audit"): raise ValueError("Unknown evaluation role")
    fresh = bool(freshness and freshness.get("reservation_verified") is True and freshness.get("candidate_preselected") is True
                 and (role != "final_audit" or freshness.get("frozen_rules_verified") is True and bool(freshness.get("candidate_frozen_sha256"))))
    runtime_check = check_runtime(runtime, freshness.get("candidate_frozen_sha256") if role == "final_audit" and freshness else None)
    quality_pass = not quality_failures and ci[0] > 0
    return {"status": "paired_evaluation_complete", "role": role, "baseline": base, "candidate": trial,
            "paired_macro_f05_delta": float(delta.mean()), "paired_delta_95pct_ci": ci, "country_deltas": country_delta,
            "links": {"true_removed": true_removed, "false_removed": false_removed, "true_added": true_added, "false_added": false_added,
                      "newly_empty_non_singletons": newly_empty,
                      "ownership_targets_changed": sum(bo.get(t, set()) != co.get(t, set()) for t in set(bo) | set(co)),
                      "baseline_collision_targets": sum(len(v) > 1 for v in bo.values()), "candidate_collision_targets": sum(len(v) > 1 for v in co.values())},
            "acceptance": {"non_inferiority_pass": not failures, "quality_non_inferiority_pass": not quality_failures,
                           "quality_improvement_pass": quality_pass, "ownership_scope_eligible": not ownership or complete,
                           "promotion_pass": quality_pass and not failures and role in ("confirmation", "final_audit") and fresh and runtime_check["passed"],
                           "evidence_scope": "fresh_evaluation" if role != "development" and fresh else "quality_only_development_or_unverified_freshness",
                           "failures": failures, "rule": "Paired LCB > -0.0005, point precision does not fall, singleton false predictions do not rise, no country delta < -0.0005. Improvement additionally requires LCB > 0."},
            "freshness": freshness or {"reservation_verified": False}, "runtime": runtime_check,
            "claimant_universe": universe or {"kind": "unspecified", "complete": False},
            "ownership_limitation": "Complete within a declared heldout universe does not reproduce full official-test claimant density." if complete else "Cheap exact-key rivals or incomplete/in-sample claims cannot validate a global ownership gain.",
            "bootstrap": {"seed": seed, "iterations": iterations, "unit": "Source 1 entity", "paired": True}}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_runtime(evidence, expected_frozen_sha256=None):
    failures = []
    if not evidence: return {"passed": False, "failures": ["runtime_not_provided"], "evidence": None}
    if expected_frozen_sha256 and evidence.get("candidate_frozen_sha256") != expected_frozen_sha256: failures.append("runtime_frozen_candidate_mismatch")
    if evidence.get("measured") is not True: failures.append("runtime_not_measured")
    if evidence.get("passed") is not True: failures.append("runtime_report_did_not_pass")
    if evidence.get("synthetic_fixture_runtime") is True or evidence.get("full_test_eta_supported") is not True:
        failures.append("representative_full_test_eta_not_supported")
    if evidence.get("mode") not in ("R1", "R2", "R3"): failures.append("unknown_release_mode")
    eta, budget = evidence.get("estimated_processing_seconds", 0), evidence.get("processing_budget_seconds", 0)
    if not isinstance(eta, (int, float)) or not isinstance(budget, (int, float)) or not np.isfinite([eta, budget]).all() or not 0 < eta <= budget:
        failures.append("estimated_processing_exceeds_or_lacks_release_budget")
    safety = evidence.get("safety_margin_fraction", 0)
    if not isinstance(safety, (int, float)) or not np.isfinite(safety) or safety < .15: failures.append("runtime_safety_margin_below_0.15")
    blocks = evidence.get("blocks", [])
    if not isinstance(blocks, list) or len(blocks) < 2: failures.append("fewer_than_two_measured_blocks")
    else:
        stages = {"reads", "lookups", "normalization", "gate_or_features", "predict", "global_decision", "diagnostic_write", "tsv_export", "strict_validator", "official_validator"}
        for block in blocks:
            if block.get("country_stratified") is not True or block.get("references", 0) <= 0: failures.append("unrepresentative_runtime_block")
            seconds = block.get("seconds", {})
            if not stages <= set(seconds) or any(not isinstance(v,(int,float)) or not np.isfinite(v) or v < 0 for v in seconds.values()): failures.append("runtime_block_missing_end_to_end_stages")
    return {"passed": not failures, "failures": sorted(set(failures)), "evidence": evidence}


def verify_reservation(plan_path, references, role):
    from export_sprint_workbench import reference_rows
    plan = json.loads(plan_path.read_text())
    name = "audit" if role == "final_audit" else "confirmation"
    if plan.get("status") != "confirmation_and_audit_reserved_no_labels_read" or plan.get("scope_reviewed") is not True:
        raise ValueError("Freshness reservation scope is incomplete")
    rows = reference_rows(references); ids = [r["entity_id"] for r in rows]
    cohort = plan["cohort_fingerprints"][name]
    if cohort["entities"] != len(ids) or cohort["reference_ids_sha256"] != hashlib.sha256("\n".join(ids).encode()).hexdigest():
        raise ValueError("Evaluation references differ from reserved fresh cohort")
    allowed = [cohort.get("references_sha256"), cohort.get("sampled_references_sha256")]
    if not any(allowed) or sha(references) not in allowed: raise ValueError("Reserved raw reference records changed")
    exposed = set()
    for ledger in plan["exposure_ledgers"]:
        path = Path(ledger["path"])
        if sha(path) != ledger["sha256"]: raise ValueError("Reservation exposure ledger changed")
        exposed.update(json.loads(path.read_text()))
    if set(ids) & exposed: raise ValueError("Fresh cohort contains previously exposed references")
    return {"reservation_verified": True, "reservation_plan_sha256": sha(plan_path), "reserved_at": plan["reserved_at"], "role": role}


def verify_prediction_evidence(path, args, frozen=None):
    report = json.loads(path.read_text())
    for name, file in (("candidate", args.candidate), ("baseline", args.baseline), ("references", args.references), ("universe", args.universe)):
        if report.get(name + "_sha256") != sha(file): raise ValueError(f"Prediction evidence {name} hash mismatch")
    if frozen:
        if report.get("candidate_frozen_sha256") != sha(args.freeze) or report.get("baseline_frozen_sha256") != frozen["parent_frozen_sha256"]:
            raise ValueError("Predictions do not belong to the frozen candidate/control")
        if "decision_config" not in frozen or report.get("decision_config_sha256") != canonical_sha(frozen["decision_config"]):
            raise ValueError("Predictions do not use the exact frozen decision rules")
    else:
        if report.get("status") != "candidate_selected_before_confirmation": raise ValueError("Confirmation candidate was not preselected")
        when = datetime.fromisoformat(report["selected_at"].replace("Z", "+00:00"))
        if when.tzinfo is None or when > datetime.now(timezone.utc): raise ValueError("Invalid preselection time")
    return {"candidate_preselected": True, "frozen_rules_verified": bool(frozen), "candidate_frozen_sha256": sha(args.freeze) if frozen else None, "prediction_evidence_sha256": sha(path)}

def verify_audit_freeze(path, references):
    frozen = json.loads(path.read_text())
    if frozen.get("status") != "candidate_frozen": raise ValueError("Candidate not frozen before audit")
    when = datetime.fromisoformat(frozen["frozen_at"].replace("Z", "+00:00"))
    if when.tzinfo is None or when > datetime.now(timezone.utc): raise ValueError("Invalid freeze time")
    required = ("parent_frozen_sha256", "split_ledger_sha256", "audit_sampled_ids_sha256")
    if any(not frozen.get(k) for k in required): raise ValueError("Incomplete candidate freeze proof")
    if frozen["audit_sampled_ids_sha256"] != sha(references): raise ValueError("Audit reference manifest differs from freeze")
    groups = ("code_sha256", "model_sha256", "schema_sha256", "index_sha256", "native_sha256")
    for group in groups:
        if not frozen.get(group): raise ValueError(f"Missing freeze hash group {group}")
        for filename, expected in frozen[group].items():
            if sha(filename) != expected: raise ValueError(f"Frozen artifact changed: {filename}")
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True); parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True); parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True); parser.add_argument("--ownership", action="store_true")
    roles = parser.add_mutually_exclusive_group(); roles.add_argument("--audit", action="store_true"); roles.add_argument("--confirmation", action="store_true")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--runtime", type=Path); parser.add_argument("--reservation-plan", type=Path); parser.add_argument("--prediction-evidence", type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Evaluation output already exists; audit must be evaluated once")
    proof, freshness = None, None
    role = "final_audit" if args.audit else ("confirmation" if args.confirmation else "development")
    runtime = json.loads(args.runtime.read_text()) if args.runtime else None
    if args.audit:
        if not args.freeze: raise ValueError("Audit requires --freeze")
        proof = verify_audit_freeze(args.freeze, args.references)
    if args.audit or args.confirmation:
        if not args.reservation_plan or not args.prediction_evidence: raise ValueError("Fresh evaluation requires reservation plan and prediction evidence before opening labels")
        freshness = {**verify_reservation(args.reservation_plan, args.references, role), **verify_prediction_evidence(args.prediction_evidence, args, proof)}
    if args.audit:
        if not check_runtime(runtime)["passed"]: raise ValueError("Final audit requires a sufficient measured release budget")
        if runtime.get("candidate_frozen_sha256") != sha(args.freeze): raise ValueError("Runtime evidence belongs to another frozen candidate")
    # Anchor the audit-once lock to the reserved cohort, not a caller-selected
    # report directory, so another output path cannot silently repeat it.
    if args.audit or args.confirmation:
        lock = args.references.parent / ("audit_evaluation_started.json" if args.audit else "confirmation_evaluation_started.json")
        with lock.open("x") as stream:
            json.dump({"freeze_sha256": sha(args.freeze) if args.audit else None, "output": str(args.output.resolve()),
                       "references_sha256": sha(args.references)}, stream, indent=2)
            stream.write("\n")
    # Create a durable audit-start marker before opening the truth file.
    args.output.mkdir(parents=True)
    (args.output / "evaluation_started.json").write_text(json.dumps({"audit": args.audit, "truth_not_opened_at_start": True,
        "freeze_sha256": sha(args.freeze) if proof else None, "references_sha256": sha(args.references)}, indent=2) + "\n")
    from export_sprint_workbench import reference_rows
    rows = reference_rows(args.references); countries = {r["entity_id"]: r["country"] for r in rows}
    truth, baseline, candidate = [json.loads(p.read_text()) for p in (args.truth, args.baseline, args.candidate)]
    report = paired_evaluate(truth, baseline, candidate, countries, iterations=args.iterations,
                             universe=json.loads(args.universe.read_text()), ownership=args.ownership, role=role, freshness=freshness, runtime=runtime)
    report["input_sha256"] = {name: sha(path) for name, path in (("truth", args.truth), ("baseline", args.baseline), ("candidate", args.candidate), ("references", args.references), ("universe", args.universe))}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
