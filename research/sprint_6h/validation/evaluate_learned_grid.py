"""Independent paired checks for the three predeclared learned-weight outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from decision_policy_v2 import PAIR_KEYS
from evaluate_sprint import decide_control, entity_f05, paired_evaluate, predictions_for
from export_sprint_workbench import sha


def pair_hash(frame):
    h = hashlib.sha256()
    for s, c in frame[PAIR_KEYS].itertuples(index=False, name=None):
        h.update(f"{s}\t{c}\n".encode())
    return h.hexdigest()


def run(args):
    started = time.perf_counter()
    manifest = json.loads((args.source / "manifest.json").read_text())
    if manifest.get("status") != "complete" or not manifest.get("verified_current_pipeline") or not manifest.get("new_component_fit_owners_excluded"):
        raise ValueError("Learned development cohort is not verified and fit-owner excluded")
    if manifest.get("audit_labels_read") is not False or sha(args.frozen) != manifest["frozen_sha256"]:
        raise ValueError("Fresh labels or mismatched incumbent freeze")
    path = args.source / "pairs.parquet"
    if sha(path) != manifest["pairs_sha256"]:
        raise ValueError("Control pair table checksum differs")
    claims = pd.read_parquet(path)
    ids = json.loads((args.source / "reference_ids.json").read_text())
    evaluation_ids = json.loads((args.source / "evaluation_reference_ids.json").read_text())
    fit_ids = json.loads(args.fit_ids.read_text())
    if len(set(ids)) != len(ids) or len(ids) != manifest["entities"] or set(ids) & set(fit_ids):
        raise ValueError("Claim graph includes residual-fit owners or inconsistent coverage")
    if len(set(evaluation_ids)) != len(evaluation_ids) or not set(evaluation_ids) <= set(ids):
        raise ValueError("Invalid evaluation subset")
    if not set(claims.source1_entity_id) <= set(ids) or len(claims) != manifest["pairs"] or pair_hash(claims) != manifest["pair_order_sha256"]:
        raise ValueError("Incomplete or foreign keyed control claims")
    if sha(args.source / "truth.json") != manifest["truth_sha256"]:
        raise ValueError("Consumed development truth checksum differs")
    raw_truth = json.loads((args.source / "truth.json").read_text())
    raw_countries = json.loads((args.source / "countries.json").read_text())
    if set(raw_truth) != set(ids) or set(raw_countries) != set(ids):
        raise ValueError("Development labels and country coverage differ")
    truth = {e: raw_truth[e] for e in evaluation_ids}
    countries = {e: raw_countries[e] for e in evaluation_ids}
    frozen = json.loads(args.frozen.read_text())
    config = {k: frozen[k] for k in ("threshold", "t_first", "t_rest")}
    evaluation_mask = claims.source1_entity_id.isin(set(evaluation_ids)).to_numpy()
    control_frame = claims.loc[evaluation_mask].reset_index(drop=True)
    baseline_chosen, _ = decide_control(control_frame, claims, config)
    baseline = predictions_for(control_frame, baseline_chosen, evaluation_ids)
    universe = {"kind": "complete_development", "complete": True, "scoring_complete": True,
                "declared_references": len(ids), "scored_references": len(ids), "pairs": len(claims),
                "supervised_training_excluded": True, "new_component_fit_owners_excluded": True,
                "selection_references": len(evaluation_ids), "excluded_residual_training_references": len(fit_ids),
                "scope": "Exactly early10k plus selection20k; residual-fit20k excluded from both baseline and transformed claimant graphs. Full incumbent first stage is outer-training owner-excluded. External heldout/test claimant density not certified."}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "baseline_predictions.json").write_text(json.dumps({e: sorted(v) for e, v in baseline.items()}) + "\n")
    reports = []
    for weight, directory in ((.5, "learned_weight_0p5"), (.75, "learned_weight_0p75"), (1., "learned_weight_1p0")):
        trial_dir = args.variants / directory
        trial_manifest = json.loads((trial_dir / "manifest.json").read_text())
        if trial_manifest.get("weight") != weight or trial_manifest.get("learned_comparison_weights") != [.5, .75, 1.] or trial_manifest.get("labels_used_for_route") is not False:
            raise ValueError("Unplanned learned weight or label-dependent route")
        if trial_manifest["input_manifest_sha256"] != sha(args.source / "manifest.json") or sha(trial_dir / "pairs.parquet") != trial_manifest["pairs_sha256"]:
            raise ValueError("Transformed-score lineage differs from control")
        trial_claims = pd.read_parquet(trial_dir / "pairs.parquet")
        if not trial_claims[PAIR_KEYS + ["candidate_order"]].equals(claims[PAIR_KEYS + ["candidate_order"]]):
            raise ValueError("Learned output pair IDs/order differ")
        if pair_hash(trial_claims) != trial_manifest["pair_order_sha256"]:
            raise ValueError("Learned output key fingerprint differs")
        if not np.array_equal(trial_claims.probability_original.to_numpy(), claims.probability.to_numpy()):
            raise ValueError("Learned output original probabilities differ")
        trial_frame = trial_claims.loc[evaluation_mask].reset_index(drop=True)
        chosen, _ = decide_control(trial_frame, trial_claims, config)
        predictions = predictions_for(trial_frame, chosen, evaluation_ids)
        report = paired_evaluate(truth, baseline, predictions, countries, universe=universe, role="development", iterations=2000)
        report.update(weight=weight, decision_policy="unchanged_frozen_original", decision_config=config,
                      replayed_control_macro_matches_source=abs(report["baseline"]["macro_f05"] - manifest["evaluation_expected_macro_f05"]) < 1e-12,
                      independent_rejection="point_macro_regression" if report["paired_macro_f05_delta"] < 0 else None,
                      input_hashes={"control_manifest": sha(args.source / "manifest.json"), "control_pairs": manifest["pairs_sha256"],
                                    "candidate_manifest": sha(trial_dir / "manifest.json"), "candidate_pairs": trial_manifest["pairs_sha256"],
                                    "fit_ids": sha(args.fit_ids), "evaluation_ids": sha(args.source / "evaluation_reference_ids.json"),
                                    "truth": manifest["truth_sha256"], "countries": sha(args.source / "countries.json"), "frozen": sha(args.frozen)},
                      validation_code_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / "scripts/evaluate_sprint.py"]})
        output = args.output / directory; output.mkdir()
        (output / "candidate_predictions.json").write_text(json.dumps({e: sorted(v) for e, v in predictions.items()}) + "\n")
        rows = [{"entity_id": e, "country": countries[e], "delta_f05": entity_f05(set(truth[e]), predictions[e]) - entity_f05(set(truth[e]), baseline[e]),
                 "removed": sorted(baseline[e] - predictions[e]), "added": sorted(predictions[e] - baseline[e])} for e in evaluation_ids]
        pd.DataFrame(rows).to_parquet(output / "per_entity.parquet", index=False)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append({k: report[k] for k in ["weight", "paired_macro_f05_delta", "paired_delta_95pct_ci", "country_deltas", "baseline", "candidate", "links", "acceptance", "independent_rejection"]})
        print(json.dumps({"weight": weight, "delta": report["paired_macro_f05_delta"], "ci": report["paired_delta_95pct_ci"], "acceptance": report["acceptance"]}), flush=True)
    summary = {"status": "independent_paired_grid_complete", "evaluated_at": datetime.now(timezone.utc).isoformat(),
               "trials": reports, "claimant_universe": universe, "fresh_labels_read": False,
               "seconds": time.perf_counter() - started}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True); p.add_argument("--variants", type=Path, required=True)
    p.add_argument("--fit-ids", type=Path, required=True); p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__": main()
