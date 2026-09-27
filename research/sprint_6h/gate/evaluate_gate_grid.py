"""Predeclared development gate grid on an unchanged complete claimant table.

All label files are explicitly supplied development exports. Gate discovery
and pair-table construction never receive these labels. This script evaluates
the original frozen decision rule with vetoed proposals removed before its
unchanged claim arbitration and rank-one rescue.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_sprint import decide_control, paired_evaluate, predictions_for
from build_raw_lookup import sha256


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def gated_predictions(frame, gate, config, ids):
    keys = ["source1_entity_id", "candidate_entity_id"]
    joined = frame.merge(gate, on=keys, how="left", validate="one_to_one", sort=False)
    if len(joined) != len(frame) or joined.gate_veto.isna().any():
        raise ValueError("Incomplete gate pair join")
    if not joined[keys].equals(frame[keys]):
        raise ValueError("Gate merge reordered the original scored pairs")
    eligible = joined.loc[~joined.gate_veto, frame.columns].reset_index(drop=True)
    chosen, _ = decide_control(eligible, eligible, config)
    return predictions_for(eligible, chosen, ids)


def main():
    p = argparse.ArgumentParser()
    for key in ("scores", "chunks", "score_manifest", "raw_lookup", "stats", "grid",
                "decision_config", "truth", "countries", "selection_ids", "output"):
        p.add_argument("--" + key.replace("_", "-"), required=True, type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gate-rule", choices=("v2", "v3-house", "v4-distinctive"), default="v2")
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        raise ValueError("Gate allocation is at most eight workers")
    sealed = json.loads(a.score_manifest.read_text())
    if sealed.get("status") != "complete":
        raise ValueError("Development score universe is not complete")
    truth, countries = json.loads(a.truth.read_text()), json.loads(a.countries.read_text())
    ids = set(truth)
    if set(countries) != ids or len(ids) != 50000:
        raise ValueError("Expected complete supplied development50k exports")
    selection = json.loads(a.selection_ids.read_text())
    selected = {r["entity_id"] for r in selection}
    if len(selected) != 20000 or not selected <= ids:
        raise ValueError("Expected the previously exposed selected20k subset")
    frame = pd.read_parquet(a.scores, columns=["source1_entity_id", "candidate_entity_id", "probability"])
    if set(frame.source1_entity_id) != ids:
        raise ValueError("Scores and development labels cover different owner universes")
    config = json.loads(a.decision_config.read_text())
    chosen, _ = decide_control(frame, frame, config)
    baseline = predictions_for(frame, chosen, ids)
    dump(a.output / "baseline_predictions.json", {e: sorted(v) for e, v in baseline.items()})
    grid = json.loads(a.grid.read_text())
    if len(grid) > 6:
        raise ValueError("At most six predeclared settings")
    scope = {"kind": "complete_development", "complete": True,
             "declared_references": len(ids), "scored_references": len(ids),
             "scoring_complete": True, "supervised_training_excluded": False}
    provenance = {"scores_sha256": sha256(a.scores), "score_manifest_sha256": sha256(a.score_manifest),
                  "truth_sha256": sha256(a.truth), "countries_sha256": sha256(a.countries),
                  "grid_sha256": sha256(a.grid), "stats_sha256": sha256(a.stats),
                  "decision_config_sha256": sha256(a.decision_config),
                  "evaluator_sha256": sha256(ROOT / "scripts/evaluate_sprint.py"),
                  "runner_sha256": sha256(__file__), "label_role": "development_only",
                  "policy": "original frozen decision; veto removes proposals before arbitration/rescue",
                  "gate_rule": a.gate_rule}
    results = []
    for i, setting in enumerate(grid):
        directory = a.output / f"setting_{i}"
        config_path = directory / "gate_config.json"
        dump(config_path, setting)
        table = directory / "gate_table.parquet"
        marker = table.with_suffix(table.suffix + ".complete.json")
        if not marker.exists():
            command = [sys.executable, str(Path(__file__).with_name("build_gate_table.py")),
                       "--input-chunks", str(a.chunks), "--raw-lookup", str(a.raw_lookup),
                       "--stats", str(a.stats), "--config", str(config_path),
                       "--gate-rule", a.gate_rule,
                       "--execution-mode", "R1", "--decision-config", str(a.decision_config),
                       "--output-chunks", str(directory / "chunks"), "--merge-output", str(table),
                       "--expected-pair-order-sha256", sealed["pair_order_sha256"],
                       "--workers", str(a.workers)]
            with (directory / "builder.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        evidence = json.loads(marker.read_text())
        if evidence["sha256"] != sha256(table) or evidence["pair_order_sha256"] != sealed["pair_order_sha256"]:
            raise ValueError("Completed gate table hash/order differs")
        candidate = gated_predictions(frame, pd.read_parquet(table), config, ids)
        dump(directory / "candidate_predictions.json", {e: sorted(v) for e, v in candidate.items()})
        reports = {}
        for label, subset in (("complete50k", ids), ("selection20k", selected)):
            maps = [{e: mapping[e] for e in subset} for mapping in (truth, baseline, candidate, countries)]
            reports[label] = paired_evaluate(*maps, universe=scope, ownership=False, role="development")
        result = {"setting": setting, "table": evidence, "evaluations": reports, "provenance": provenance}
        dump(directory / "evaluation.json", result)
        results.append(result)
        dump(a.output / "grid_report.json", {"status": "running", "completed": len(results), "results": results})
        print(json.dumps({"setting": i, "config": setting,
                          "selection20k": reports["selection20k"], "complete50k": reports["complete50k"]}), flush=True)
    dump(a.output / "grid_report.json", {"status": "complete", "completed": len(results), "results": results,
                                         "provenance": provenance, "promotion": "none; development only"})


if __name__ == "__main__":
    main()
