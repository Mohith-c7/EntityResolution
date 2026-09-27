"""Fit the second stage on out-of-fold training references at scale.

Training rows come from the outer-training references, whose context features
use out-of-fold first-stage scores. Early stopping, blend and threshold use the
same fixed development split as train_competition_context.py, so selection
numbers are comparable with the incumbent on identical rows.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

from build_context_features import CONTEXT_NAMES
from train_competition_context import paired_interval
from train_scale_model import FEATURE_NAMES_V3, digest, predict, score, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.features.competition import COMPETITION_NAMES

NAMES = [*FEATURE_NAMES_V3, *CONTEXT_NAMES, *COMPETITION_NAMES]


def load(context, competition):
    manifest = json.loads((context / "manifest.json").read_text())
    extra_manifest = json.loads((competition / "manifest.json").read_text())
    if manifest["status"] != "complete" or extra_manifest["status"] != "complete" or extra_manifest["labels_read"]:
        raise ValueError("Incomplete or label-reading cache")
    source = Path(manifest["source_cache"])
    if not source.exists() and "models" in source.parts:
        source = Path(*source.parts[source.parts.index("models"):])
    refs, frames = {}, []
    for number in range(manifest["chunks"]):
        stem = f"{number:06d}"
        if digest(context / (stem + ".parquet")) != json.loads((context / (stem + ".json")).read_text())["sha256"]:
            raise ValueError("Corrupt context cache")
        for row in json.loads((source / (stem + ".json")).read_text())["entity_rows"]: refs[row["entity_id"]] = row
        base = pd.read_parquet(context / (stem + ".parquet"),
            columns=[*FEATURE_NAMES_V3, *CONTEXT_NAMES, "label", "sample_weight", "source1_entity_id", "candidate_entity_id"])
        extra = pd.read_parquet(competition / (stem + ".parquet"))
        if not (base.source1_entity_id.equals(extra.source1_entity_id) and base.candidate_entity_id.equals(extra.candidate_entity_id)):
            raise ValueError(f"Competition rows do not align with chunk {stem}")
        frames.append(pd.concat([base, extra[list(COMPETITION_NAMES)]], axis=1))
    return pd.concat(frames, ignore_index=True), refs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-context", type=Path, default=Path("models/next_round/context_train_oof"))
    p.add_argument("--train-competition", type=Path, default=Path("models/next_round/competition_train"))
    p.add_argument("--tune-context", type=Path, default=Path("models/next_round/extended_context_cache"))
    p.add_argument("--tune-competition", type=Path, default=Path("models/next_round/competition_v1"))
    p.add_argument("--incumbent", type=Path, default=Path("models/next_round/competition_context_model"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report-output", type=Path)
    p.add_argument("--baseline-threshold", type=float, default=.76)
    p.add_argument("--add-development-training", action="store_true",
                   help="Also fit on the 20k development references the incumbent trained on")
    p.add_argument("--trees", type=int, default=4000)
    p.add_argument("--leaves", type=int, default=31)
    p.add_argument("--learning-rate", type=float, default=.035)
    p.add_argument("--min-child", type=int, default=100)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-file", type=Path, help="Evaluate a saved second-stage model instead of fitting one")
    args = p.parse_args()
    if args.output.exists() and not args.model_file: raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True); os.nice(5); pa.set_cpu_count(4)
    start = time.perf_counter()
    tune, refs = load(args.tune_context, args.tune_competition)
    ids = sorted(refs, key=lambda e: hashlib.sha256(("context-split-v1:" + e).encode()).digest())
    if len(ids) != 50000: raise ValueError("Expected the fixed 50k development set")
    development_ids, early_ids, select_ids = ids[:20000], ids[20000:30000], ids[30000:]
    params = dict(objective="binary", metric="binary_logloss", learning_rate=args.learning_rate, num_leaves=args.leaves,
        min_data_in_leaf=args.min_child, lambda_l2=5., feature_fraction=.9, bagging_fraction=.9, bagging_freq=1,
        max_bin=127, seed=args.seed, num_threads=args.threads, deterministic=True, force_col_wise=True, verbosity=-1)
    if args.model_file:
        model = lgb.Booster(model_file=str(args.model_file)); training_rows = training_references = None
        if model.feature_name() != NAMES: raise ValueError("Saved model feature order differs")
    else:
        train, _ = load(args.train_context, args.train_competition)
        if set(train.source1_entity_id) & set(refs): raise ValueError("Training and development references overlap")
        if args.add_development_training:
            train = pd.concat([train, tune[tune.source1_entity_id.isin(development_ids)]], ignore_index=True)
        early = tune[tune.source1_entity_id.isin(early_ids)]
        print(json.dumps({"stage": "loaded", "training_rows": len(train), "training_references": int(train.source1_entity_id.nunique()),
                          "seconds": round(time.perf_counter() - start, 1)}), flush=True)
        training = lgb.Dataset(train[NAMES].to_numpy(dtype=np.float32), label=train.label.to_numpy(), weight=train.sample_weight.to_numpy(),
                               feature_name=NAMES, free_raw_data=True)
        validation = lgb.Dataset(early[NAMES].to_numpy(dtype=np.float32), label=early.label.to_numpy(),
                                 weight=early.sample_weight.to_numpy(), reference=training, feature_name=NAMES, free_raw_data=True)
        training_rows, training_references = len(train), int(train.source1_entity_id.nunique()); del train
        model = lgb.train(params, training, num_boost_round=args.trees, valid_sets=[validation], valid_names=["early"],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(250)])
        model.save_model(str(args.output / "model.txt"))
    selection = tune[tune.source1_entity_id.isin(select_ids)]; del tune
    probabilities = predict(model, selection[NAMES].to_numpy(dtype=np.float32), args.threads)

    base = selection.ctx_probability.to_numpy(); label = selection.label.to_numpy(dtype=np.uint8)
    positions = {e: i for i, e in enumerate(select_ids)}
    groups = selection.source1_entity_id.map(positions).to_numpy(dtype=np.int32)
    expected = np.array([len(refs[e]["true_ids"]) for e in select_ids])
    countries = np.array([refs[e]["country"] for e in select_ids])
    baseline, baseline_values = score(base, args.baseline_threshold, label, groups, expected, countries)
    incumbent = incumbent_values = None
    incumbent_scores = np.load(args.incumbent / "selection_probabilities.npy") if args.incumbent.name else None
    if incumbent_scores is not None and incumbent_scores.shape == base.shape:
        incumbent_report = json.loads((args.incumbent / "report.json").read_text())["protocol_selection"]
        w, t = incumbent_report["context_weight"], incumbent_report["threshold"]
        incumbent, incumbent_values = score((1 - w) * base + w * incumbent_scores, t, label, groups, expected, countries)

    trials = []
    for weight in (0., .25, .5, .75, 1.):
        blended = (1 - weight) * base + weight * probabilities
        for threshold in np.r_[np.arange(.30, .951, .01), .96, .97, .98, .99, .995]:
            result, _ = score(blended, round(float(threshold), 6), label, groups, expected)
            trials.append({"context_weight": weight, **result})
    eligible = [r for r in trials if r["micro_precision"] >= baseline["micro_precision"] - .002
                and r["singleton_false_positives"] <= baseline["singleton_false_positives"]]
    chosen = max(eligible, key=lambda r: r["macro_f05"]) if eligible else None
    selection_report = None
    if chosen:
        detail, values = score((1 - chosen["context_weight"]) * base + chosen["context_weight"] * probabilities,
                               chosen["threshold"], label, groups, expected, countries)
        selection_report = {"context_weight": chosen["context_weight"], **detail,
            "gain_vs_first_stage": float((values - baseline_values).mean()),
            "gain_vs_first_stage_95pct_ci": paired_interval(values - baseline_values)}
        if incumbent_values is not None:
            selection_report.update({"gain_vs_incumbent": float((values - incumbent_values).mean()),
                                     "gain_vs_incumbent_95pct_ci": paired_interval(values - incumbent_values)})
    importance = dict(zip(NAMES, model.feature_importance("gain")))
    total = sum(importance.values())
    report = {"status": "exploratory_selection_only", "first_stage_baseline": baseline, "incumbent": incumbent,
        "protocol_selection": selection_report, "best_unconstrained": max(trials, key=lambda r: r["macro_f05"]),
        "training_rows": training_rows, "training_references": training_references,
        "development_training_added": args.add_development_training, "parameters": params,
        "best_iteration": model.current_iteration() if args.model_file else model.best_iteration, "trees_allowed": args.trees,
        "top_features": sorted(((n, v / total) for n, v in importance.items()), key=lambda x: -x[1])[:25],
        "train_context_manifest_sha256": digest(args.train_context / "manifest.json"),
        "tune_context_manifest_sha256": digest(args.tune_context / "manifest.json"),
        "retained_candidate_oracle": score(label, .5, label, groups, expected, countries)[0],
        "caveat": "Previously inspected development references; the selection set also chose the blend and threshold.",
        "seconds": time.perf_counter() - start, "fresh_audit_evaluated": False, "submission_generated": False}
    np.save(args.output / "selection_probabilities.npy", probabilities)
    write_json(args.output / "threshold_sweep.json", trials); write_json(args.output / "report.json", report)
    if args.report_output: write_json(args.report_output, report)
    print(json.dumps({k: report[k] for k in ("protocol_selection", "best_iteration", "training_references")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
