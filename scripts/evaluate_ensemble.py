"""Average second-stage models on the development selection references.

Each model's selection scores are averaged with a small grid of mixing weights,
blended with the first stage, and thresholded under the protocol constraints;
the rank-one rule is then selected on the best mixture.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_decision_rules import summary
from train_competition_context import paired_interval
from train_scale_model import write_json

THRESHOLDS = np.r_[np.arange(.30, .951, .01), .96, .97, .98, .99]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("models/next_round/extended_context_cache"))
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--incumbent", type=Path, default=Path("models/next_round/competition_context_model"))
    p.add_argument("--baseline-threshold", type=float, default=.76)
    p.add_argument("--report-output", type=Path, required=True)
    p.add_argument("--bridge", action="store_true", help="Score the saved bridge candidates of the same selection references")
    args = p.parse_args()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    source = Path(manifest["source_cache"])
    if not source.exists() and "models" in source.parts:
        source = Path(*source.parts[source.parts.index("models"):])
    refs, frames = {}, []
    for n in range(manifest["chunks"]):
        for row in json.loads((source / f"{n:06d}.json").read_text())["entity_rows"]: refs[row["entity_id"]] = row
        frames.append(pd.read_parquet(args.cache / f"{n:06d}.parquet", columns=["source1_entity_id", "label", "ctx_probability"]))
    frame = pd.concat(frames, ignore_index=True)
    ids = sorted(refs, key=lambda e: hashlib.sha256(("context-split-v1:" + e).encode()).digest())[30000:]
    frame = frame[frame.source1_entity_id.isin(set(ids))].reset_index(drop=True)
    positions = {e: i for i, e in enumerate(ids)}
    groups = frame.source1_entity_id.map(positions).to_numpy(dtype=np.int32)
    labels = frame.label.to_numpy(dtype=np.uint8)
    expected = np.array([len(refs[e]["true_ids"]) for e in ids])
    base = frame.ctx_probability.to_numpy()
    _, limits = summary(base >= args.baseline_threshold, labels, groups, expected)

    def eligible(result):
        return (result["micro_precision"] >= limits["micro_precision"] - .002
                and result["singleton_false_positives"] <= limits["singleton_false_positives"])

    members = {str(m): np.load(m / "selection_probabilities.npy") for m in args.models}
    incumbent = incumbent_values = None
    if args.incumbent.name:
        incumbent_report = json.loads((args.incumbent / "report.json").read_text())["protocol_selection"]
        members[str(args.incumbent)] = np.load(args.incumbent / "selection_probabilities.npy")
        incumbent_values, incumbent = summary((1 - incumbent_report["context_weight"]) * base + incumbent_report["context_weight"]
            * members[str(args.incumbent)] >= incumbent_report["threshold"], labels, groups, expected)
    if args.bridge:
        by_entity = dict(zip(ids, incumbent_values)) if incumbent_values is not None else None
        truth = {e: set(v) for e, v in json.loads(Path("models/next_round/context_selection_adapter/sampled_truth.json").read_text()).items()}
        frame = pd.read_parquet("models/next_round/bridge_selection_stacked_features.parquet",
                                columns=["source1_entity_id", "candidate_entity_id", "ctx_probability"])
        if set(truth) != set(ids): raise ValueError("Bridge references differ from the selection references")
        labels = np.array([c in truth[e] for e, c in zip(frame.source1_entity_id, frame.candidate_entity_id)], dtype=np.uint8)
        ids = sorted(truth); positions = {e: i for i, e in enumerate(ids)}
        groups = frame.source1_entity_id.map(positions).to_numpy(dtype=np.int32)
        expected = np.array([len(truth[e]) for e in ids]); base = frame.ctx_probability.to_numpy()
        incumbent_values = np.array([by_entity[e] for e in ids]) if by_entity else None
        members = {str(m): np.load(m / "bridge_selection_probabilities.npy") for m in args.models}
    if any(v.shape != base.shape for v in members.values()): raise ValueError("Selection scores do not align")
    names = list(members)
    grid = [w for w in itertools.product((0., .25, .5, .75, 1.), repeat=len(names)) if abs(sum(w) - 1) < 1e-9]
    best = None
    for mix in grid:
        stacked = sum(w * members[n] for w, n in zip(mix, names) if w)
        for weight in (.5, .75, 1.):
            blended = (1 - weight) * base + weight * stacked
            for threshold in THRESHOLDS:
                values, result = summary(blended >= threshold, labels, groups, expected)
                if eligible(result) and (best is None or result["macro_f05"] > best["macro_f05"]):
                    best = {"mix": dict(zip(names, mix)), "context_weight": weight, "threshold": round(float(threshold), 4),
                            **result, "_values": values, "_probability": blended}
    probability, values = best.pop("_probability"), best.pop("_values")
    order = np.lexsort((-probability, groups))
    first = np.zeros(len(frame), dtype=bool)
    first[order[np.r_[True, groups[order][1:] != groups[order][:-1]]]] = True
    rank_one = {"macro_f05": best["macro_f05"], "t_first": best["threshold"], "t_rest": best["threshold"]}
    rank_values = values
    for t_first in np.arange(.10, .951, .025):
        for t_rest in THRESHOLDS:
            mask = (probability >= t_rest) | (first & (probability >= t_first))
            candidate_values, result = summary(mask, labels, groups, expected)
            if eligible(result) and result["macro_f05"] > rank_one["macro_f05"]:
                rank_one = {"t_first": round(float(t_first), 4), "t_rest": round(float(t_rest), 4), **result}
                rank_values = candidate_values
    def versus_incumbent(result, entity_values):
        if incumbent_values is None: return result
        return {**result, "gain_vs_incumbent": float((entity_values - incumbent_values).mean()),
                "gain_vs_incumbent_95pct_ci": paired_interval(entity_values - incumbent_values)}

    out = {"status": "exploratory_selection_only", "first_stage_limits": limits, "incumbent": incumbent,
           "best_mixture": versus_incumbent(best, values), "best_mixture_rank_one": versus_incumbent(rank_one, rank_values),
           "caveat": "Mixture, blend and thresholds chosen on the same selection references; confirm on a fresh audit."}
    write_json(args.report_output, out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
