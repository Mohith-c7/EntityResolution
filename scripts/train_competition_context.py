"""Retrain the candidate-context stage with Source 1 competition counts added.

Uses the same development split, model settings and selection rule as
train_context_model.py, and compares against the saved incumbent context model
on identical selection references and pairs.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

from build_context_features import CONTEXT_NAMES
from train_scale_model import FEATURE_NAMES_V3, digest, write_json, score, predict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.features.competition import COMPETITION_NAMES


def paired_interval(delta, seed=42):
    rng = np.random.default_rng(seed)
    return np.quantile([rng.choice(delta, len(delta), replace=True).mean() for _ in range(1000)], [.025, .975]).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("models/next_round/extended_context_cache"))
    p.add_argument("--competition", type=Path, default=Path("models/next_round/competition_v1"))
    p.add_argument("--incumbent", type=Path, default=Path("models/next_round/extended_context_model"))
    p.add_argument("--output", type=Path, default=Path("models/next_round/competition_context_model"))
    p.add_argument("--report-output", type=Path, default=Path("reports/experiments/round2/competition_context_model.json"))
    p.add_argument("--baseline-threshold", type=float, default=.76)
    p.add_argument("--trees", type=int, default=1200)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--drop", nargs="*", default=[], help="Feature names to leave out for ablations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--without-competition", action="store_true", help="Control run on the incumbent feature set")
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); os.nice(5); pa.set_cpu_count(1)
    start = time.perf_counter()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    competition = json.loads((args.competition / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["split"] != "outer_tune_only": raise ValueError("Wrong context cache")
    if competition["status"] != "complete" or competition["labels_read"]: raise ValueError("Wrong competition cache")
    if competition["source_manifest_sha256"] != digest(args.cache / "manifest.json"): raise ValueError("Competition cache is stale")
    added = [] if args.without_competition else list(COMPETITION_NAMES)
    names = [*FEATURE_NAMES_V3, *CONTEXT_NAMES, *added]
    if not set(args.drop) <= set(names): raise ValueError("Unknown feature in --drop")
    names = [n for n in names if n not in args.drop]
    source = Path(manifest["source_cache"]); refs = {}; frames = []
    for number in range(manifest["chunks"]):
        stem = f"{number:06d}"
        if digest(args.cache / (stem + ".parquet")) != json.loads((args.cache / (stem + ".json")).read_text())["sha256"]:
            raise ValueError("Corrupt context cache")
        if digest(args.competition / (stem + ".parquet")) != json.loads((args.competition / (stem + ".json")).read_text())["sha256"]:
            raise ValueError("Corrupt competition cache")
        for row in json.loads((source / (stem + ".json")).read_text())["entity_rows"]: refs[row["entity_id"]] = row
        base = pd.read_parquet(args.cache / (stem + ".parquet"),
            columns=[*FEATURE_NAMES_V3, *CONTEXT_NAMES, "label", "sample_weight", "source1_entity_id", "candidate_entity_id"])
        extra = pd.read_parquet(args.competition / (stem + ".parquet"))
        if not (base.source1_entity_id.equals(extra.source1_entity_id) and base.candidate_entity_id.equals(extra.candidate_entity_id)):
            raise ValueError(f"Competition rows do not align with chunk {stem}")
        frames.append(pd.concat([base, extra[list(COMPETITION_NAMES)]], axis=1))
    frame = pd.concat(frames, ignore_index=True); del frames
    ids = sorted(refs, key=lambda e: hashlib.sha256(("context-split-v1:" + e).encode()).digest())
    if len(ids) != 50000: raise ValueError("Expected the fixed 50k development set")
    train_ids, early_ids, select_ids = ids[:20000], ids[20000:30000], ids[30000:]
    training = frame[frame.source1_entity_id.isin(train_ids)]
    early = frame[frame.source1_entity_id.isin(early_ids)]
    selection = frame[frame.source1_entity_id.isin(select_ids)]; del frame
    model = lgb.LGBMClassifier(objective="binary", n_estimators=args.trees, learning_rate=.035, num_leaves=31,
        min_child_samples=100, reg_lambda=5., colsample_bytree=.9, subsample=.9, subsample_freq=1,
        max_bin=127, random_state=args.seed, n_jobs=args.threads, deterministic=True, force_col_wise=True, verbosity=-1)
    model.fit(training[names].to_numpy(dtype=np.float32), training.label, sample_weight=training.sample_weight,
        feature_name=names, eval_set=[(early[names].to_numpy(dtype=np.float32), early.label)],
        eval_sample_weight=[early.sample_weight], callbacks=[lgb.early_stopping(75), lgb.log_evaluation(200)])
    model.booster_.save_model(str(args.output / "model.txt"))
    probabilities = predict(model.booster_, selection[names].to_numpy(dtype=np.float32), args.threads)

    base = selection.ctx_probability.to_numpy(); label = selection.label.to_numpy(dtype=np.uint8)
    positions = {e: i for i, e in enumerate(select_ids)}
    groups = selection.source1_entity_id.map(positions).to_numpy(dtype=np.int32)
    expected = np.array([len(refs[e]["true_ids"]) for e in select_ids])
    countries = np.array([refs[e]["country"] for e in select_ids])
    baseline, baseline_values = score(base, args.baseline_threshold, label, groups, expected, countries)

    incumbent_report = json.loads((args.incumbent / "report.json").read_text())
    incumbent_base = np.load(args.incumbent / "baseline_selection_probabilities.npy")
    if incumbent_base.shape != base.shape or not np.allclose(incumbent_base, base):
        raise ValueError("Incumbent selection rows do not align")
    w, t = incumbent_report["selection"]["context_weight"], incumbent_report["selection"]["threshold"]
    incumbent_probability = (1 - w) * base + w * np.load(args.incumbent / "selection_probabilities.npy")
    incumbent, incumbent_values = score(incumbent_probability, t, label, groups, expected, countries)

    trials = []
    for weight in (0., .25, .5, .75, 1.):
        blended = (1 - weight) * base + weight * probabilities
        for threshold in np.r_[np.arange(.30, .951, .01), .96, .97, .98, .99, .995]:
            result, _ = score(blended, round(float(threshold), 6), label, groups, expected)
            trials.append({"context_weight": weight, **result})

    def choose(max_singletons):
        eligible = [r for r in trials if r["micro_precision"] >= baseline["micro_precision"] - .002
                    and r["singleton_false_positives"] <= max_singletons]
        if not eligible: return None
        chosen = max(eligible, key=lambda r: r["macro_f05"])
        detail, values = score((1 - chosen["context_weight"]) * base + chosen["context_weight"] * probabilities,
                               chosen["threshold"], label, groups, expected, countries)
        return {"context_weight": chosen["context_weight"], **detail,
                "gain_vs_first_stage": float((values - baseline_values).mean()),
                "gain_vs_first_stage_95pct_ci": paired_interval(values - baseline_values),
                "gain_vs_incumbent": float((values - incumbent_values).mean()),
                "gain_vs_incumbent_95pct_ci": paired_interval(values - incumbent_values)}

    importance = dict(zip(names, model.booster_.feature_importance("gain")))
    total = sum(importance.values())
    report = {"status": "exploratory_selection_only", "first_stage_baseline": baseline, "incumbent_context": incumbent,
        "protocol_selection": choose(baseline["singleton_false_positives"]),
        "selection_with_incumbent_singletons": choose(incumbent["singleton_false_positives"]),
        "selection_with_earlier_pipeline_singletons": choose(28),
        "competition_gain_share": {n: importance[n] / total for n in added if n in importance},
        "features": len(names), "seed": args.seed, "best_iteration": model.best_iteration_, "trees_allowed": args.trees,
        "competition_manifest_sha256": digest(args.competition / "manifest.json"),
        "cache_manifest_sha256": digest(args.cache / "manifest.json"),
        "retained_candidate_oracle": score(label, .5, label, groups, expected, countries)[0],
        "caveat": "Previously inspected development references; the selection set also chose the blend and threshold.",
        "seconds": time.perf_counter() - start, "fresh_audit_evaluated": False, "submission_generated": False}
    np.save(args.output / "selection_probabilities.npy", probabilities)
    write_json(args.output / "threshold_sweep.json", trials); write_json(args.output / "report.json", report)
    write_json(args.report_output, report)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
