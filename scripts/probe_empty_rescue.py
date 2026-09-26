"""Tune a conservative empty-prediction rescue; report singleton tradeoffs."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa


def main():
    os.nice(10); pa.set_cpu_count(1)
    artifact = Path("models/features_v3_audit01")
    cache = Path("models/redesign/wide_tune_cache")
    manifest = json.loads((cache / "manifest.json").read_text())
    if not manifest["complete"] or manifest["test"]: raise ValueError("Use a complete tuning cache")
    truth = {e: set(ids) for e, ids in json.loads((artifact / "sampled_truth.json").read_text()).items()}
    frame = pd.concat([pd.read_parquet(cache / p, columns=["source1_entity_id", "candidate_entity_id", "probability"])
        for p in manifest["files"]], ignore_index=True)
    ordered = frame.sort_values(["source1_entity_id", "probability", "candidate_entity_id"], ascending=[True, False, True])
    top = ordered.groupby("source1_entity_id", sort=False).head(2)
    empty = []
    for eid, rows in top.groupby("source1_entity_id", sort=False):
        first = rows.iloc[0]
        if first.probability >= .75: continue
        second = rows.iloc[1].probability if len(rows) > 1 else 0.
        correct = first.candidate_entity_id in truth[eid]
        gain = 1.25 / (.25 * len(truth[eid]) + 1) if correct else (-1. if not truth[eid] else 0.)
        empty.append((eid, first.probability, first.probability-second, correct, not truth[eid], gain))
    values = pd.DataFrame(empty, columns=["entity_id", "probability", "margin", "top_is_true", "singleton", "gain"])
    trials = []
    for threshold in (.3, .4, .45, .5, .55, .6, .65, .7, .72, .74):
        for margin in (0., .05, .1, .2, .3):
            chosen = values[(values.probability >= threshold) & (values.margin >= margin)]
            trials.append({"minimum_probability": threshold, "minimum_margin": margin,
                "rescued_references": len(chosen), "true_links_added": int(chosen.top_is_true.sum()),
                "new_singleton_false_positives": int(chosen.singleton.sum()),
                "tuning_macro_gain": float(chosen.gain.sum() / len(manifest["reference_ids"]))})
    safe = [r for r in trials if not r["new_singleton_false_positives"]]
    result = {"scope": "Exploratory tuning selection; requires independent confirmation",
        "empty_predictions": len(values), "correct_empty_singletons": int(values.singleton.sum()),
        "incorrect_empty_non_singletons": int((~values.singleton).sum()),
        "top_candidate_true_among_empty": int(values.top_is_true.sum()),
        "correct_empty_fraction": float(values.singleton.mean()),
        "best_unconstrained": max(trials, key=lambda r: r["tuning_macro_gain"]),
        "best_without_new_singleton_errors": max(safe, key=lambda r: r["tuning_macro_gain"]) if safe else None,
        "trials": trials}
    Path("reports/experiments/redesign/empty_rescue_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}))


if __name__ == "__main__": main()
