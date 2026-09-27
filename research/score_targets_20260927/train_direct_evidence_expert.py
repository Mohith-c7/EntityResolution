"""One complementary CPU expert; no test inference or fresh-audit access.

Removes retrieval, alias-derived and corpus-frequency features from the matcher.
The reference matcher is a predeclared seed, not the winner of the parallel trials.
Blend/threshold selection and confirmation use disjoint S1 subsets of an OLD
development pool; confirmation is not a fresh audit or France evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from cpu_feature_experiment import (
    FEATURE_NAMES_65, NEW_FEATURE_NAMES, PAIR_KEYS, digest, entity_metrics,
    load_prepared, paired_interval, tune_threshold, write_json,
)

EXCLUDED = frozenset({
    "blocking_score", "canonical_name_sort", "canonical_name_set",
    "canonical_core_exact", "canonical_name_cosine", "canonical_name_length_ratio",
    "query_family_log_frequency", "candidate_alias_confidence", "candidate_alias_log_support",
    "name_weighted_jaccard", "address_weighted_jaccard", "strongest_shared_address_idf",
    "cand_name_min_log_df", "ref_name_min_log_df", "addr_left_unmatched_idf",
    "addr_right_unmatched_idf", "addr_distinct_mismatch",
})
ALL_COLUMNS = [*FEATURE_NAMES_65, *NEW_FEATURE_NAMES]
DIRECT_COLUMNS = [name for name in ALL_COLUMNS if name not in EXCLUDED]
BLEND_WEIGHTS = (0.0, 0.10, 0.25, 0.50, 1.0)
BASELINE_ARM = "baseline65_plus_cpu12"


def split_entities(truth, seed=70010):
    ids = sorted(truth, key=lambda key: hashlib.sha256(
        f"direct-expert-selection-v1:{seed}:{key}".encode()).digest())
    midpoint = len(ids) // 2
    if not midpoint or midpoint == len(ids):
        raise ValueError("Need at least two selection entities")
    return set(ids[:midpoint]), set(ids[midpoint:])


def subset(frame, truth, ids):
    mask = frame.source1_entity_id.isin(ids).to_numpy()
    return mask, frame.loc[mask].reset_index(drop=True), {key: truth[key] for key in sorted(ids)}


def blend(base, expert, weight):
    if np.shape(base) != np.shape(expert) or not 0 <= weight <= 1:
        raise ValueError("Invalid blend inputs")
    result = (1 - weight) * np.asarray(base) + weight * np.asarray(expert)
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("Invalid probabilities")
    return result


def choose_blend(base, expert, frame, truth):
    outcomes = []
    for weight in BLEND_WEIGHTS:
        result, _ = tune_threshold(blend(base, expert, weight), frame, truth)
        outcomes.append({"expert_weight": weight, **result})
    # Prefer the original matcher if scores and precision tie.
    selected = max(outcomes, key=lambda row: (row["macro_f05"], row["micro_precision"],
                                               -row["expert_weight"]))
    return selected, outcomes


def wait_for_baseline(directory, expected_manifest, timeout=3600):
    deadline = time.monotonic() + timeout
    report_path = directory / "report.json"
    while not report_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Baseline not complete; expert saved, no comparison produced")
        time.sleep(5)
    protocol = json.loads((directory / "protocol.json").read_text())
    report = json.loads(report_path.read_text())
    if protocol["prepared_manifest_sha256"] != expected_manifest:
        raise ValueError("Baseline was trained on a different prepared dataset")
    if protocol["parameters"]["seed"] != 70007:
        raise ValueError("Baseline seed must be the predeclared 70007")
    path = directory / f"{BASELINE_ARM}.txt"
    if digest(path) != report["arms"][BASELINE_ARM]["model_sha256"]:
        raise ValueError("Baseline model hash mismatch")
    return path


def train(args):
    if not 1 <= args.threads <= 8 or args.trees < 1:
        raise ValueError("Expected 1..8 threads and positive tree budget")
    if args.output.exists():
        raise FileExistsError("Use a new experiment output directory")
    if args.data.resolve().name != "cpu_scale_stage2_20260927_e70007":
        raise ValueError("Only the prepared development dataset is allowed")
    args.output.mkdir(parents=True)
    os.nice(10)
    import lightgbm as lgb
    started = time.perf_counter()
    manifest_hash = digest(args.data / "manifest.json")
    _, frames, truths = load_prepared(args.data)
    tune_ids, confirmation_ids = split_entities(truths["selection"])
    params = dict(objective="binary", metric="binary_logloss", learning_rate=.04,
                  num_leaves=31, min_data_in_leaf=30, lambda_l2=2., feature_fraction=.9,
                  bagging_fraction=.9, bagging_freq=1, max_bin=127, seed=70007,
                  num_threads=args.threads, deterministic=True, force_col_wise=True, verbosity=-1)
    write_json(args.output / "protocol.json", {
        "hypothesis": "Direct evidence expert can correct errors caused by overreliance on retrieval/alias signals",
        "parameters": params, "max_trees": args.trees, "early_stopping_rounds": 75,
        "features": DIRECT_COLUMNS, "excluded_features": sorted(EXCLUDED),
        "baseline_directory": str(args.baseline), "baseline_seed": 70007,
        "blend_weights": BLEND_WEIGHTS, "selection_ids": sorted(tune_ids),
        "confirmation_ids": sorted(confirmation_ids),
        "acceptance": "Confirmation paired CI lower bound > 0; precision loss <= .002; singleton errors do not rise",
        "manifest_sha256": manifest_hash, "script_sha256": digest(Path(__file__)),
        "helper_sha256": digest(Path(__file__).with_name("cpu_feature_experiment.py")),
        "limitations": ["Old development pool, not a fresh audit", "Pre-bridge candidates remain unchanged",
            "Not the complete frozen two-stage model", "No labeled France data; no France accuracy claim"],
    })
    datasets = {}
    for fold in ("train", "early_stop"):
        frame = frames[fold]
        datasets[fold] = lgb.Dataset(frame[DIRECT_COLUMNS].to_numpy(np.float32),
            label=frame.label, weight=frame.sample_weight, feature_name=DIRECT_COLUMNS,
            reference=datasets.get("train"))
    print(json.dumps({"stage": "training", "features": len(DIRECT_COLUMNS),
                      "train_pairs": len(frames["train"]), "threads": args.threads}), flush=True)
    model = lgb.train(params, datasets["train"], num_boost_round=args.trees,
        valid_sets=[datasets["early_stop"]], valid_names=["early_stop"],
        callbacks=[lgb.early_stopping(75), lgb.log_evaluation(100)])
    model_path = args.output / "direct_evidence.txt"
    model.save_model(str(model_path))
    frame, truth = frames["selection"], truths["selection"]
    expert = model.predict(frame[DIRECT_COLUMNS].to_numpy(np.float32), num_threads=args.threads)
    scored = frame[PAIR_KEYS + ["label", "country"]].copy()
    scored["expert_probability"] = expert
    scored.to_parquet(args.output / "expert_scores.parquet", index=False)
    print(json.dumps({"stage": "expert_saved", "best_iteration": model.best_iteration}), flush=True)
    baseline_path = wait_for_baseline(args.baseline, manifest_hash)
    baseline = lgb.Booster(model_file=str(baseline_path))
    if baseline.feature_name() != ALL_COLUMNS:
        raise ValueError("Baseline feature order mismatch")
    base = baseline.predict(frame[ALL_COLUMNS].to_numpy(np.float32), num_threads=args.threads)
    tune_mask, tune_frame, tune_truth = subset(frame, truth, tune_ids)
    confirm_mask, confirm_frame, confirm_truth = subset(frame, truth, confirmation_ids)
    selected, grid = choose_blend(base[tune_mask], expert[tune_mask], tune_frame, tune_truth)
    baseline_choice = next(row for row in grid if row["expert_weight"] == 0)
    mixed = blend(base, expert, selected["expert_weight"])
    write_json(args.output / "frozen_decision.json", {
        "selected": selected, "baseline": baseline_choice, "selection_grid": grid,
        "expert_model_sha256": digest(model_path), "baseline_model_sha256": digest(baseline_path),
    })
    base_result, base_values = entity_metrics(base[confirm_mask], baseline_choice["threshold"],
                                             confirm_frame, confirm_truth)
    result, values = entity_metrics(mixed[confirm_mask], selected["threshold"], confirm_frame, confirm_truth)
    interval = paired_interval(values - base_values, seed=70010)
    # Slices are entity-level, retaining every candidate and full truth for each entity.
    countries = frame.groupby("source1_entity_id", sort=False).country.first().to_dict()
    alias = frame.groupby("source1_entity_id", sort=False).candidate_alias_confidence.max().to_dict()
    block = frame.groupby("source1_entity_id", sort=False).blocking_score.max().to_dict()
    slice_ids = {f"country:{country}": {key for key in confirmation_ids if countries.get(key) == country}
                 for country in sorted(set(countries.values()))}
    slice_ids["no_alias_evidence"] = {key for key in confirmation_ids if alias.get(key, 0) == 0}
    slice_ids["weak_retrieval_evidence"] = {key for key in confirmation_ids if block.get(key, 0) < .65}
    slices = {}
    for name, ids in slice_ids.items():
        if not ids:
            continue
        mask, part, part_truth = subset(frame, truth, ids)
        before, _ = entity_metrics(base[mask], baseline_choice["threshold"], part, part_truth)
        after, _ = entity_metrics(mixed[mask], selected["threshold"], part, part_truth)
        slices[name] = {"baseline": before, "selected": after,
                        "gain": after["macro_f05"] - before["macro_f05"]}
    accepted = (selected["expert_weight"] > 0 and interval["paired_bootstrap_95_low"] > 0
        and result["micro_precision"] >= base_result["micro_precision"] - .002
        and result["singleton_false_positives"] <= base_result["singleton_false_positives"])
    scored["baseline_probability"] = base
    scored["selected_probability"] = mixed
    scored["split"] = np.where(tune_mask, "blend_selection", "development_confirmation")
    scored.to_parquet(args.output / "comparison_scores.parquet", index=False)
    report = {"status": "exploratory_development_confirmation_not_fresh_audit",
        "baseline": base_result, "selected": result, "selected_expert_weight": selected["expert_weight"],
        "paired_confirmation": interval, "slices": slices, "pilot_acceptance_passed": bool(accepted),
        "best_iteration": model.best_iteration, "seconds": time.perf_counter() - started,
        "fresh_audit_evaluated": False, "submission_modified": False,
        "feature_importance": dict(zip(DIRECT_COLUMNS, map(float, model.feature_importance("gain"))))}
    write_json(args.output / "report.json", report)
    print(json.dumps({"stage": "complete", "paired_confirmation": interval,
                      "accepted": bool(accepted), "weight": selected["expert_weight"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", default=6, type=int)
    parser.add_argument("--trees", default=2400, type=int)
    train(parser.parse_args())
