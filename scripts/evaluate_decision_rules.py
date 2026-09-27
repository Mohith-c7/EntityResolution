"""Compare link decision rules on the development selection references.

A rank-one threshold and a threshold for additional links are selected jointly
under the protocol constraints (precision >= first-stage baseline - .002 and no
more singleton false positives than the baseline).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_competition_context import paired_interval
from train_scale_model import write_json


def entity_scores(chosen, labels, groups, expected):
    predicted = np.bincount(groups[chosen], minlength=len(expected))
    tp = np.bincount(groups[chosen & (labels == 1)], minlength=len(expected))
    values = np.zeros(len(expected))
    denominator = .25 * expected + predicted
    np.divide(1.25 * tp, denominator, out=values, where=denominator > 0)
    values[(expected == 0) & (predicted == 0)] = 1.
    return values, predicted, tp


def summary(chosen, labels, groups, expected):
    values, predicted, tp = entity_scores(chosen, labels, groups, expected)
    return values, {"macro_f05": float(values.mean()), "micro_precision": float(tp.sum() / max(1, predicted.sum())),
        "micro_recall": float(tp.sum() / max(1, expected.sum())), "predicted_links": int(predicted.sum()),
        "singleton_false_positives": int(((expected == 0) & (predicted > 0)).sum()),
        "non_singleton_empty_predictions": int(((expected > 0) & (predicted == 0)).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("models/next_round/extended_context_cache"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--probabilities", type=Path, help="Final selection scores to use instead of the model blend, e.g. after exclusivity")
    p.add_argument("--report-output", type=Path, required=True)
    p.add_argument("--bridge", action="store_true", help="Use the saved bridge candidates of the same selection references")
    args = p.parse_args()
    report = json.loads((args.model / "report.json").read_text())
    chosen = report["protocol_selection"]; limits = report["first_stage_baseline"]
    if args.bridge:
        adapter = Path("models/next_round/context_selection_adapter")
        truth = {e: set(v) for e, v in json.loads((adapter / "sampled_truth.json").read_text()).items()}
        frame = pd.read_parquet("models/next_round/bridge_selection_stacked_features.parquet",
                                columns=["source1_entity_id", "candidate_entity_id", "ctx_probability"])
        frame["label"] = [c in truth[e] for e, c in zip(frame.source1_entity_id, frame.candidate_entity_id)]
        ids = sorted(truth)
        refs = {e: {"true_ids": sorted(truth[e])} for e in ids}
        stacked = np.load(args.model / "bridge_selection_probabilities.npy")
    else:
        manifest = json.loads((args.cache / "manifest.json").read_text())
        source = Path(manifest["source_cache"])
        refs, frames = {}, []
        for n in range(manifest["chunks"]):
            for row in json.loads((source / f"{n:06d}.json").read_text())["entity_rows"]: refs[row["entity_id"]] = row
            frames.append(pd.read_parquet(args.cache / f"{n:06d}.parquet", columns=["source1_entity_id", "label", "ctx_probability"]))
        frame = pd.concat(frames, ignore_index=True)
        ids = sorted(refs, key=lambda e: hashlib.sha256(("context-split-v1:" + e).encode()).digest())[30000:]
        frame = frame[frame.source1_entity_id.isin(set(ids))].reset_index(drop=True)
        stacked = np.load(args.model / "selection_probabilities.npy")
    probability = (1 - chosen["context_weight"]) * frame.ctx_probability.to_numpy() + chosen["context_weight"] * stacked
    if args.probabilities:
        probability = np.load(args.probabilities)
        if probability.shape != stacked.shape: raise ValueError("Override scores do not align with the selection rows")
    positions = {e: i for i, e in enumerate(ids)}
    groups = frame.source1_entity_id.map(positions).to_numpy(dtype=np.int32)
    labels = frame.label.to_numpy(dtype=np.uint8)
    expected = np.array([len(refs[e]["true_ids"]) for e in ids])
    order = np.lexsort((-probability, groups))
    first = np.zeros(len(frame), dtype=bool)
    first[order[np.r_[True, groups[order][1:] != groups[order][:-1]]]] = True

    def eligible(result):
        return (result["micro_precision"] >= limits["micro_precision"] - .002
                and result["singleton_false_positives"] <= limits["singleton_false_positives"])

    base_values, base = summary(probability >= chosen["threshold"], labels, groups, expected)
    best = (base["macro_f05"], chosen["threshold"], chosen["threshold"], base, base_values)
    grid = []
    for t_first in np.r_[np.arange(.10, .951, .025)]:
        for t_rest in np.r_[np.arange(.40, .951, .01), .96, .97, .98, .99]:
            mask = (probability >= t_rest) | (first & (probability >= t_first))
            values, result = summary(mask, labels, groups, expected)
            grid.append({"t_first": round(float(t_first), 4), "t_rest": round(float(t_rest), 4), **result})
            if eligible(result) and result["macro_f05"] > best[0]:
                best = (result["macro_f05"], float(t_first), float(t_rest), result, values)
    out = {"status": "exploratory_selection_only", "model": str(args.model), "single_threshold": {"threshold": chosen["threshold"], **base},
           "rank_one_rule": {"t_first": best[1], "t_rest": best[2], **best[3],
                             "gain": float((best[4] - base_values).mean()), "gain_95pct_ci": paired_interval(best[4] - base_values)},
           "caveat": "Thresholds chosen on the same selection references; confirm on a fresh audit."}
    write_json(args.report_output, out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
