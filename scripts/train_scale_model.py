"""Fit a larger matcher from immutable, cross-fitted caches; select on tuning only."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.features.registry import FEATURE_NAMES_V3, FEATURE_VERSION_V3


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inventory(cache, split):
    manifest = json.loads((cache / "manifest.json").read_text())
    progress = json.loads((cache / "progress.json").read_text())
    if progress["status"] != "complete" or manifest["split"] != split:
        raise ValueError("Cache must be complete and in the requested fold")
    if manifest["feature_version"] != FEATURE_VERSION_V3:
        raise ValueError("Incompatible cache feature schema")
    chunks, entities = [], {}
    for number in range(progress["chunks"]):
        marker = cache / f"{number:06d}.json"
        data = json.loads(marker.read_text())
        path = cache / f"{number:06d}.parquet"
        if digest(path) != data["parquet_sha256"]:
            raise ValueError(f"Corrupt cache: {path}")
        chunks.append((path, data))
        for row in data["entity_rows"]:
            if row["entity_id"] in entities: raise ValueError("Duplicate cached reference")
            entities[row["entity_id"]] = row
    if len(entities) != manifest["references"] or len(entities) != progress["entities"]:
        raise ValueError("Incomplete reference universe")
    return manifest, chunks, entities


def materialize(cache, split, output, limit=0):
    manifest, chunks, entities = inventory(cache, split)
    ids = sorted(entities, key=lambda e: hashlib.sha256(e.encode()).digest())
    if limit: ids = ids[:limit]
    positions = {eid: i for i, eid in enumerate(ids)}
    rows = sum(entities[e]["retained_pairs"] for e in ids)
    if not rows: raise ValueError("No training/evaluation pairs")
    prefix = output / split
    features = np.lib.format.open_memmap(str(prefix) + "_features.npy", mode="w+", dtype=np.float32,
                                        shape=(rows, len(FEATURE_NAMES_V3)))
    labels = np.lib.format.open_memmap(str(prefix) + "_labels.npy", mode="w+", dtype=np.uint8, shape=(rows,))
    weights = np.lib.format.open_memmap(str(prefix) + "_weights.npy", mode="w+", dtype=np.float32, shape=(rows,))
    groups = np.lib.format.open_memmap(str(prefix) + "_groups.npy", mode="w+", dtype=np.int32, shape=(rows,))
    columns = [*FEATURE_NAMES_V3, "label", "sample_weight", "source1_entity_id"]
    offset = 0
    for path, data in chunks:
        if not any(r["entity_id"] in positions for r in data["entity_rows"]): continue
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame.source1_entity_id.isin(positions)]
        end = offset + len(frame)
        matrix = frame[list(FEATURE_NAMES_V3)].to_numpy(dtype=np.float32)
        if not np.isfinite(matrix).all(): raise ValueError("Nonfinite model features")
        features[offset:end] = matrix
        labels[offset:end] = frame.label.to_numpy()
        weights[offset:end] = frame.sample_weight.to_numpy()
        groups[offset:end] = frame.source1_entity_id.map(positions).to_numpy()
        offset = end
    if offset != rows: raise ValueError("Pair-count mismatch")
    for array in (features, labels, weights, groups): array.flush()
    expected = np.array([len(entities[e]["true_ids"]) for e in ids], dtype=np.int32)
    countries = np.array([entities[e]["country"] for e in ids])
    weight_sums = np.bincount(groups, weights=weights, minlength=len(ids))
    has_pairs = np.array([entities[e]["retained_pairs"] > 0 for e in ids])
    if not np.allclose(weight_sums[has_pairs], 1., atol=1e-6): raise ValueError("Reference weights do not sum to one")
    write_json(output / f"{split}_references.json", ids)
    print(json.dumps({"stage": "numeric_cache_ready", "split": split, "entities": len(ids), "pairs": rows}), flush=True)
    return features, labels, weights, groups, expected, countries, ids, manifest


def score(probabilities, threshold, labels, groups, expected, countries=None):
    chosen = probabilities >= threshold
    predicted = np.bincount(groups[chosen], minlength=len(expected))
    tp = np.bincount(groups[chosen & (labels == 1)], minlength=len(expected))
    values = np.zeros(len(expected))
    denominator = .25 * expected + predicted
    np.divide(1.25 * tp, denominator, out=values, where=denominator > 0)
    values[(expected == 0) & (predicted == 0)] = 1.
    result = {"threshold": float(threshold), "macro_f05": float(values.mean()), "entities": len(expected),
        "true_links": int(expected.sum()), "predicted_links": int(predicted.sum()), "true_positive_links": int(tp.sum()),
        "micro_precision": float(tp.sum() / max(1, predicted.sum())),
        "micro_recall": float(tp.sum() / max(1, expected.sum())),
        "singleton_entities": int((expected == 0).sum()),
        "singleton_false_positives": int(((expected == 0) & (predicted > 0)).sum()),
        "non_singleton_empty_predictions": int(((expected > 0) & (predicted == 0)).sum())}
    if countries is not None:
        result["country_macro_f05"] = {str(c): {"entities": int((countries == c).sum()),
            "macro_f05": float(values[countries == c].mean())} for c in sorted(set(countries))}
    return result, values


def predict(model, matrix, threads):
    result = np.empty(len(matrix), dtype=np.float64)
    for offset in range(0, len(matrix), 100000):
        result[offset:offset+100000] = model.predict(matrix[offset:offset+100000], num_threads=threads)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--tune-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--trees", type=int, default=1200)
    p.add_argument("--baseline-model", type=Path, default=Path("models/features_v3_audit01/model.txt"))
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError("Use a fresh model directory")
    if args.train_limit < 0 or min(args.threads, args.trees) < 1: p.error("Invalid limits")
    args.output.mkdir(parents=True); os.nice(10); pa.set_cpu_count(1)
    started = time.perf_counter()
    params = dict(objective="binary", metric="binary_logloss", learning_rate=.04, num_leaves=31,
        min_data_in_leaf=30, lambda_l2=2., feature_fraction=.9, bagging_fraction=.9, bagging_freq=1,
        max_bin=127, seed=42, num_threads=args.threads, deterministic=True, force_col_wise=True, verbosity=-1)
    protocol = {"feature_version": FEATURE_VERSION_V3, "feature_names": list(FEATURE_NAMES_V3), "parameters": params,
        "maximum_trees": args.trees, "early_stopping_rounds": 75,
        "selection": "Maximize tuning macro F0.5; precision >= same-cache baseline minus .002; singleton false positives <= same-cache baseline.",
        "baseline_threshold": .75, "fresh_audit_evaluated": False,
        "train_manifest_sha256": digest(args.train_cache / "manifest.json"),
        "tune_manifest_sha256": digest(args.tune_cache / "manifest.json"),
        "baseline_model_sha256": digest(args.baseline_model),
        "training_supervision": "Five inner-fold alias models exclude each training reference's entire fold in retrieval and pair features; only outer-training labels fit aliases."}
    write_json(args.output / "protocol.json", protocol)
    tx, ty, tw, tg, te, tc, tids, tm = materialize(args.train_cache, "train", args.output, args.train_limit)
    vx, vy, vw, vg, ve, vc, vids, vm = materialize(args.tune_cache, "tune", args.output)
    if set(tids) & set(vids): raise ValueError("Train/tuning entity overlap")
    for key in ("feature_version", "search", "postings", "native_sha256", "counts_fingerprint"):
        if tm[key] != vm[key]: raise ValueError(f"Training/tuning contract mismatch: {key}")
    baseline = lgb.Booster(model_file=str(args.baseline_model))
    if baseline.feature_name() != list(FEATURE_NAMES_V3): raise ValueError("Baseline feature order differs")
    baseline_p = predict(baseline, vx, args.threads)
    baseline_score, baseline_values = score(baseline_p, .75, vy, vg, ve, vc)
    np.save(args.output / "baseline_probabilities_tune.npy", baseline_p)
    print(json.dumps({"stage": "baseline_same_tuning", **baseline_score}), flush=True)
    training = lgb.Dataset(tx, label=ty, weight=tw, feature_name=list(FEATURE_NAMES_V3), free_raw_data=True)
    validation = lgb.Dataset(vx, label=vy, weight=vw, reference=training, feature_name=list(FEATURE_NAMES_V3), free_raw_data=True)
    model = lgb.train(params, training, num_boost_round=args.trees, valid_sets=[validation], valid_names=["tune"],
        callbacks=[lgb.early_stopping(75), lgb.log_evaluation(100)])
    model.save_model(str(args.output / "model.txt"))
    probabilities = predict(model, vx, args.threads)
    np.save(args.output / "probabilities_tune.npy", probabilities)
    trials = [score(probabilities, round(float(t), 6), vy, vg, ve)[0]
              for t in np.r_[np.arange(.30, .951, .01), .96, .97, .98, .99, .995, .999]]
    eligible = [r for r in trials if r["micro_precision"] >= baseline_score["micro_precision"] - .002
                and r["singleton_false_positives"] <= baseline_score["singleton_false_positives"]]
    chosen = max(eligible, key=lambda r: (r["macro_f05"], r["micro_precision"])) if eligible else None
    values = None
    if chosen:
        chosen, values = score(probabilities, chosen["threshold"], vy, vg, ve, vc)
        np.save(args.output / "entity_scores_tune.npy", values)
    np.save(args.output / "baseline_entity_scores_tune.npy", baseline_values)
    write_json(args.output / "threshold_sweep.json", trials)
    report = {"status": "tuning_only_requires_fresh_audit", "chosen": chosen, "baseline_same_tuning": baseline_score,
        "retained_candidate_oracle": score(vy, .5, vy, vg, ve, vc)[0],
        "macro_gain": chosen["macro_f05"] - baseline_score["macro_f05"] if chosen else None,
        "training_entities": len(tids), "training_pairs": len(tx), "tuning_entities": len(vids), "tuning_pairs": len(vx),
        "best_iteration": model.best_iteration, "feature_version": FEATURE_VERSION_V3,
        "feature_names": list(FEATURE_NAMES_V3), "search_config": vm["search"],
        "retrieval": {"engine": "bounded-postings-v1", "options": vm["postings"]},
        "model_sha256": digest(args.output / "model.txt"), "fresh_audit_evaluated": False,
        "leaderboard_score": None, "seconds": time.perf_counter()-started}
    write_json(args.output / "report.json", report)
    write_json(args.output / "feature_registry.json", {"version": FEATURE_VERSION_V3, "names": list(FEATURE_NAMES_V3)})
    shutil.copyfile(vm["aliases"]["None"]["path"], args.output / "name_aliases.json")
    shutil.copyfile(Path("models/features_v3_audit01/MODEL_LICENSE.txt"), args.output / "MODEL_LICENSE.txt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
