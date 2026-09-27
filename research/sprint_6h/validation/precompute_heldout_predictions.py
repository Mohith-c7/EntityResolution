"""Unlabeled complete-heldout incumbent predictions and optional v2 mechanics."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from decision_policy_v2 import decide_v2
from evaluate_sprint import canonical_sha, predictions_for
from export_sprint_workbench import reference_rows, sha


def original_decide(runner, frozen):
    if sha(runner) != frozen["code_sha256"]["scripts/run_frozen_pipeline.py"]:
        raise ValueError("Original runner differs from pinned incumbent code")
    source = runner.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "decide")
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(runner), "exec"), namespace)
    return namespace["decide"], hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()


def run(args):
    start = time.perf_counter()
    evidence = json.loads(args.evidence.read_text())
    if not all(evidence.get(k) is True for k in ("complete", "scoring_complete", "supervised_training_excluded")) or evidence.get("kind") != "complete_heldout" or evidence.get("labels_read") is not False:
        raise ValueError("Complete unlabeled heldout scoring evidence required")
    manifest_path = args.workbench / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha(manifest_path) != evidence["score_manifest_sha256"] or manifest.get("verified_current_pipeline") is not True:
        raise ValueError("Scoring seal differs from complete graph evidence")
    frozen = json.loads(args.frozen.read_text())
    if sha(args.frozen) != evidence["frozen_control_sha256"]:
        raise ValueError("Incumbent configuration differs from scored control")
    path = args.workbench / "pairs.parquet"
    if sha(path) != evidence["pairs_sha256"]:
        raise ValueError("Heldout scores changed")
    rows = reference_rows(args.workbench / "references.json")
    ids = [r["entity_id"] for r in rows]
    if ids != manifest["reference_ids"] or len(ids) != evidence["scored_references"]:
        raise ValueError("Missing processed references including empty candidates")
    config = {k: frozen[k] for k in ("threshold", "t_first", "t_rest")}
    decide, function_sha = original_decide(args.runner, frozen)
    frame = pd.read_parquet(path, columns=["source1_entity_id", "candidate_entity_id", "probability", "candidate_order"])
    args.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"stage": "inputs_verified", "references": len(ids), "pairs": len(frame)}), flush=True)
    before = time.perf_counter()
    chosen, lost = decide(frame, frame, config)
    original_seconds = time.perf_counter() - before
    predictions = predictions_for(frame, chosen, ids)
    prediction_path = args.output / "baseline_predictions.json"
    prediction_path.write_text(json.dumps({e: sorted(predictions[e]) for e in ids}) + "\n")
    accepted = frame.loc[chosen].copy()
    accepted.to_parquet(args.output / "baseline_accepted_pairs.parquet", index=False)
    owners = accepted.groupby("candidate_entity_id").source1_entity_id.nunique()
    cohorts = {}
    for role in ("confirmation", "audit"):
        references = args.reservation_plan.parent / role / "references.json"
        selected = reference_rows(references)
        selected_ids = [r["entity_id"] for r in selected]
        if not set(selected_ids) <= set(ids): raise ValueError("Fresh cohort absent from complete predictions")
        output = args.output / (role + "_baseline_predictions.json")
        output.write_text(json.dumps({e: sorted(predictions[e]) for e in selected_ids}) + "\n")
        cohorts[role] = {"entities": len(selected_ids), "references_sha256": sha(references),
                         "baseline_sha256": sha(output), "predicted_links": sum(len(predictions[e]) for e in selected_ids),
                         "empty_prediction_entities": sum(not predictions[e] for e in selected_ids)}
    report = {"status": "incumbent_complete_heldout_predictions_no_labels", "predicted_at": datetime.now(timezone.utc).isoformat(),
              "claimants": len(ids), "pairs": len(frame), "predicted_links": int(chosen.sum()),
              "empty_prediction_references": sum(not v for v in predictions.values()), "ownership_lost_pairs": int(lost.sum()),
              "final_collision_targets": int((owners > 1).sum()), "decision_config": config,
              "decision_config_sha256": canonical_sha(config), "original_runner_sha256": sha(args.runner),
              "original_decide_function_sha256": function_sha, "baseline_predictions_sha256": sha(prediction_path),
              "baseline_accepted_pairs_sha256": sha(args.output / "baseline_accepted_pairs.parquet"),
              "source_hashes": {"scoring_evidence": sha(args.evidence), "score_manifest": sha(manifest_path),
                                "pairs": evidence["pairs_sha256"], "pair_order": evidence["pair_order_sha256"],
                                "frozen": sha(args.frozen), "reservation_plan": sha(args.reservation_plan)},
              "cohorts": cohorts, "labels_read": False, "timing": {"original_decision_seconds": original_seconds},
              "limitation": evidence["limitation"]}
    (args.output / "baseline_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "incumbent_predictions_complete", "links": report["predicted_links"], "seconds": original_seconds}), flush=True)
    if args.v2:
        before = time.perf_counter()
        candidate, _, diagnostics = decide_v2(frame, frame, {**config, "ownership_margin": 0.})
        v2_predictions = predictions_for(frame, candidate, ids)
        changed = [e for e in ids if predictions[e] != v2_predictions[e]]
        v2 = {"status": "unlabeled_ownership_v2_mechanics_only", "changed_prediction_references": len(changed),
              "changed_accepted_pairs": int((chosen != candidate).sum()), "summary": diagnostics["summary"],
              "decision_seconds": time.perf_counter() - before, "quality_or_promotion_evidence": False,
              "changed_references": changed}
        (args.output / "ownership_v2_mechanics.json").write_text(json.dumps(v2, indent=2) + "\n")
        print(json.dumps({k: v for k, v in v2.items() if k != "changed_references"}), flush=True)
    report["timing"].update(elapsed_seconds=time.perf_counter() - start, peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (args.output / "baseline_manifest.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("workbench", "evidence", "frozen", "runner", "reservation_plan", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--v2", action="store_true")
    run(p.parse_args())


if __name__ == "__main__": main()
