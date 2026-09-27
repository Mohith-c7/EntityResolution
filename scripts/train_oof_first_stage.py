"""Out-of-fold first-stage scores for the training references.

Each inner fold is scored by a model fitted on the other four folds with the
extended model's parameters and tree count, so second-stage features built from
these scores are out-of-sample for every training reference.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.features.registry import FEATURE_NAMES_V3
from src.model.crossfit_aliases import inner_fold
from train_scale_model import digest, predict, write_json


def row_folds(model_dir):
    ids = json.loads((model_dir / "train_references.json").read_text())
    groups = np.load(model_dir / "train_groups.npy", mmap_mode="r")
    entity_folds = np.array([inner_fold(e) for e in ids], dtype=np.int8)
    return entity_folds[groups]


def fit_fold(model_dir, output, fold, threads, trees):
    protocol = json.loads((model_dir / "protocol.json").read_text())
    params = {**protocol["parameters"], "num_threads": threads}
    report_path = model_dir / "report.json"
    fitted = json.loads(report_path.read_text()).get("best_iteration") if report_path.exists() else None
    trees = trees or fitted or protocol["maximum_trees"]
    folds = row_folds(model_dir)
    features = np.load(model_dir / "train_features.npy", mmap_mode="r")
    labels = np.load(model_dir / "train_labels.npy", mmap_mode="r")
    weights = np.load(model_dir / "train_weights.npy", mmap_mode="r")
    fit_rows = np.flatnonzero(folds != fold)
    held_rows = np.flatnonzero(folds == fold)
    started = time.perf_counter()
    dataset = lgb.Dataset(np.ascontiguousarray(features[fit_rows]), label=np.asarray(labels[fit_rows]),
                          weight=np.asarray(weights[fit_rows]), feature_name=list(FEATURE_NAMES_V3), free_raw_data=True)
    model = lgb.train(params, dataset, num_boost_round=trees, callbacks=[lgb.log_evaluation(0)])
    model.save_model(str(output / f"model_fold_{fold}.txt"))
    scores = np.full(len(folds), np.nan)
    scores[held_rows] = predict(model, np.ascontiguousarray(features[held_rows]), threads)
    np.save(output / f"oof_fold_{fold}.npy", scores)
    write_json(output / f"fold_{fold}.json", {"fold": fold, "fit_rows": len(fit_rows), "held_rows": len(held_rows),
        "trees": trees, "threads": threads, "seconds": time.perf_counter() - started,
        "model_sha256": digest(output / f"model_fold_{fold}.txt")})


def merge(model_dir, output):
    folds = row_folds(model_dir)
    merged = np.full(len(folds), np.nan)
    for fold in range(5):
        part = np.load(output / f"oof_fold_{fold}.npy")
        rows = folds == fold
        if np.isnan(part[rows]).any() or not np.isnan(part[~rows]).all(): raise ValueError(f"Fold {fold} rows misaligned")
        merged[rows] = part[rows]
    if not np.isfinite(merged).all(): raise ValueError("Missing out-of-fold scores")
    np.save(output / "oof_probabilities_train.npy", merged)
    write_json(output / "manifest.json", {"status": "complete", "rows": len(merged),
        "model_dir": str(model_dir), "train_features_sha256": digest(model_dir / "train_features.npy"),
        "folds": [json.loads((output / f"fold_{f}.json").read_text()) for f in range(5)],
        "policy": "Each row is scored by a model that never saw its reference's inner fold."})
    print(json.dumps({"stage": "merged", "rows": len(merged)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path("models/next_round/extended_model"))
    p.add_argument("--output", type=Path, default=Path("models/next_round/oof_first_stage"))
    p.add_argument("--fold", type=int, choices=range(5))
    p.add_argument("--merge", action="store_true")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--trees", type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); os.nice(5)
    if args.merge: merge(args.model_dir, args.output)
    elif args.fold is not None: fit_fold(args.model_dir, args.output, args.fold, args.threads, args.trees)
    else: p.error("Pass --fold or --merge")


if __name__ == "__main__":
    main()
