"""Evaluate a frozen tuning selection once on the reserved entity-level audit."""
import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa

from train_scale_model import digest, materialize, predict, score, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    selection = json.loads(args.selection.read_text())
    if selection["status"] != "frozen_before_audit": raise ValueError("Selection was not frozen")
    for prefix in ("model", "baseline_model"):
        if digest(selection[f"{prefix}_path"]) != selection[f"{prefix}_sha256"]:
            raise ValueError("A frozen model changed")
    if args.output.exists(): raise FileExistsError("Audit already started; do not tune against it")
    args.output.mkdir(parents=True); os.nice(10); pa.set_cpu_count(1)
    write_json(args.output / "audit_started.json", {"selection_sha256": digest(args.selection)})
    x, y, w, groups, expected, countries, ids, manifest = materialize(args.cache, "holdout", args.output)
    if manifest["audit_selection"] != selection: raise ValueError("Cache belongs to another audit selection")
    new_model = lgb.Booster(model_file=selection["model_path"])
    baseline = lgb.Booster(model_file=selection["baseline_model_path"])
    if new_model.feature_name() != baseline.feature_name(): raise ValueError("Feature schema mismatch")
    new_probability = predict(new_model, x, args.threads)
    old_probability = predict(baseline, x, args.threads)
    new, new_values = score(new_probability, selection["threshold"], y, groups, expected, countries)
    old, old_values = score(old_probability, selection["baseline_threshold"], y, groups, expected, countries)
    delta = new_values - old_values
    rng = np.random.default_rng(42)
    interval = np.quantile([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)], [.025, .975]).tolist()
    accepted = interval[0] > 0 and new["micro_precision"] >= old["micro_precision"] - .002 and new["singleton_false_positives"] <= old["singleton_false_positives"]
    np.save(args.output / "probabilities.npy", new_probability)
    np.save(args.output / "baseline_probabilities.npy", old_probability)
    np.save(args.output / "entity_scores.npy", new_values)
    np.save(args.output / "baseline_entity_scores.npy", old_values)
    report = {"status": "fresh_audit_complete", "accepted": accepted, "model": new, "baseline": old,
        "paired_macro_gain": float(delta.mean()), "paired_gain_95pct_ci": interval,
        "selection_sha256": digest(args.selection), "audit_entities": len(ids), "leaderboard_score": None,
        "france": "No labeled France records; country generalization remains unmeasured.",
        "acceptance": "Paired gain lower bound > 0, precision drop <= .002, and no increase in singleton false positives."}
    write_json(args.output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
