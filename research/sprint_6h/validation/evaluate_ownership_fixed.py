"""One predeclared margin-zero ownership ablation on a sealed development graph.

Only already exposed development truth is accepted. This is not a fresh audit
and cannot certify ownership at the density of the complete heldout graph.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from decision_policy_v2 import decide_v2
from evaluate_sprint import decide_control, entity_f05, paired_evaluate, predictions_for
from export_sprint_workbench import reference_rows, sha


def verify_development(workbench, frozen):
    manifest = json.loads((workbench / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("verified_current_pipeline") is not True:
        raise ValueError("Development workbench is not sealed and verified")
    if manifest.get("split") != "development" or manifest.get("frozen_sha256") != sha(frozen):
        raise ValueError("Require the pinned development control")
    if manifest.get("audit_labels_read") is not False or manifest.get("second_stage_owner_excluded") is not True:
        raise ValueError("Development workbench label/training exclusions are missing")
    for name in ("parity", "asset"):
        proof = Path(manifest[name + "_proof"])
        if sha(proof) != manifest[name + "_proof_sha256"]:
            raise ValueError("Workbench verification proof changed")
    if sha(workbench / "pairs.parquet") != manifest["pairs_sha256"]:
        raise ValueError("Workbench pairs changed after sealing")
    rows = reference_rows(workbench / "references.json")
    ids = [r["entity_id"] for r in rows]
    if ids != manifest["reference_ids"] or len(ids) != manifest["entities"]:
        raise ValueError("Workbench reference coverage changed")
    if sum(c["entities"] for c in manifest["chunks"]) != len(ids):
        raise ValueError("Incomplete processed reference universe")
    claims = pd.read_parquet(workbench / "pairs.parquet", columns=[
        "source1_entity_id", "candidate_entity_id", "probability", "candidate_order"])
    digest = hashlib.sha256()
    for s, c in claims[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None):
        digest.update(f"{s}\t{c}\n".encode())
    if digest.hexdigest() != manifest["pair_order_sha256"] or len(claims) != manifest["pairs"]:
        raise ValueError("Workbench pair IDs/order differ from the seal")
    if not set(claims.source1_entity_id) <= set(ids):
        raise ValueError("Foreign reference in development claim graph")
    return manifest, rows, claims


def run(args):
    start = time.perf_counter()
    manifest, rows, claims = verify_development(args.workbench, args.frozen)
    print(json.dumps({"stage": "sealed_workbench_verified", "pairs": len(claims)}), flush=True)
    selected = reference_rows(args.references)
    ids = [r["entity_id"] for r in selected]
    if not set(ids) <= set(manifest["reference_ids"]):
        raise ValueError("Selected evaluation references outside development graph")
    label_manifest = json.loads((args.truth.parent / "manifest.json").read_text())
    if label_manifest.get("status") != "already_exposed_development_truth_exported":
        raise ValueError("Only previously exposed development truth is authorized")
    if label_manifest["outputs"][args.truth.name] != sha(args.truth):
        raise ValueError("Consumed development truth changed")
    truth = json.loads(args.truth.read_text())
    if set(truth) != set(ids):
        raise ValueError("Selected references and development truth differ")
    config = json.loads(args.frozen.read_text())
    config = {k: config[k] for k in ("threshold", "t_first", "t_rest")}
    candidate_config = {**config, "ownership_margin": 0.0, "tie_tolerance": 1e-12, "ownership_min_score": 0.0}
    frame = claims[claims.source1_entity_id.isin(set(ids))].reset_index(drop=True)
    before = time.perf_counter()
    original, original_lost = decide_control(frame, claims, config)
    control_seconds = time.perf_counter() - before
    print(json.dumps({"stage": "unchanged_control_complete", "seconds": control_seconds}), flush=True)
    before = time.perf_counter()
    candidate, _, diagnostics = decide_v2(frame, claims, candidate_config)
    candidate_seconds = time.perf_counter() - before
    baseline = predictions_for(frame, original, ids)
    trial = predictions_for(frame, candidate, ids)
    countries = {r["entity_id"]: r["country"] for r in selected}
    universe = {"kind": "complete_development", "complete": True, "scoring_complete": True,
                "declared_references": len(rows), "scored_references": len(rows),
                "pairs": len(claims), "selection_references": len(ids),
                "supervised_training_excluded": manifest["second_stage_owner_excluded"],
                "second_stage_owner_excluded": True,
                "first_stage_owner_excluded_oof": manifest["first_stage_owner_excluded_oof"],
                "scope": "All incumbent claims in the declared development50k graph; both incumbent fit-owner sets exclude this cohort (capture verifies the shared fitted-owner ledger), first-stage full model, outer-training second stage. No learned residual claims; no heldout density certification."}
    report = paired_evaluate(truth, baseline, trial, countries, universe=universe,
                             ownership=True, role="development", iterations=args.iterations)
    report.update(candidate_policy="ownership_v2_margin0_fixed", evaluated_at=datetime.now(timezone.utc).isoformat(),
                  decision_config=candidate_config, global_decision=diagnostics["summary"],
                  selected_decision_reasons=dict(Counter(diagnostics["rows"].ownership_reason)),
                  changed_prediction_entities=sum(baseline[e] != trial[e] for e in ids),
                  input_hashes={"workbench_manifest": sha(args.workbench / "manifest.json"),
                                "pairs": manifest["pairs_sha256"], "frozen_control": sha(args.frozen),
                                "references": sha(args.references), "truth": sha(args.truth)},
                  code_hashes={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / "scripts/decision_policy_v2.py", ROOT / "scripts/evaluate_sprint.py"]},
                  timing={"control_seconds": control_seconds, "candidate_seconds": candidate_seconds,
                          "elapsed_seconds": time.perf_counter() - start,
                          "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "full_test_eta_supported": False})
    args.output.mkdir(parents=True, exist_ok=False)
    for name, values in (("baseline_predictions", baseline), ("candidate_predictions", trial)):
        (args.output / (name + ".json")).write_text(json.dumps({e: sorted(v) for e, v in values.items()}) + "\n")
    (args.output / "universe.json").write_text(json.dumps(universe, indent=2) + "\n")
    per_entity = []
    for e in ids:
        removed, added = baseline[e] - trial[e], trial[e] - baseline[e]
        correct = set(truth[e])
        per_entity.append({"entity_id": e, "country": countries[e], "baseline_f05": entity_f05(correct, baseline[e]),
                           "candidate_f05": entity_f05(correct, trial[e]),
                           "true_removed": sorted(removed & correct), "false_removed": sorted(removed - correct),
                           "true_added": sorted(added & correct), "false_added": sorted(added - correct),
                           "changed": baseline[e] != trial[e]})
    pd.DataFrame(per_entity).to_parquet(args.output / "per_entity.parquet", index=False)
    diagnostic = diagnostics["rows"].copy()
    diagnostic["baseline_accepted"] = original
    diagnostic["baseline_ownership_lost"] = original_lost
    diagnostic.loc[original != candidate].to_parquet(args.output / "changed_pairs.parquet", index=False)
    report["output_hashes"] = {p.name: sha(p) for p in sorted(args.output.iterdir())}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "paired_evaluation_complete", "output": str(args.output),
                      "delta": report["paired_macro_f05_delta"], "ci": report["paired_delta_95pct_ci"],
                      "links": report["links"], "acceptance": report["acceptance"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbench", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
