"""Fit a small ranking classifier on existing training-fold hard negatives."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.cheap_ranker import RANK_FEATURES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("models/redesign/cheap_ranker_v1"))
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.nice(10)
    pa.set_cpu_count(1)
    started = time.perf_counter()
    artifact = Path("models/features_v3_audit01")
    train = pd.read_parquet(artifact / "pairs_train.parquet")
    cache = Path("models/redesign/wide_tune_cache")
    manifest = json.loads((cache / "manifest.json").read_text())
    if not manifest["complete"] or manifest["test"]: raise ValueError("Invalid tuning cache")
    tune = pd.concat([pd.read_parquet(cache / name) for name in manifest["files"]], ignore_index=True)
    truth = {e: set(ids) for e, ids in json.loads((artifact / "sampled_truth.json").read_text()).items()}
    tune["label"] = [c in truth[e] for e, c in tune[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None)]
    if set(train.source1_entity_id) & set(tune.source1_entity_id): raise ValueError("Entity split overlap")
    weights = 1 / train.groupby("source1_entity_id").label.transform("count")
    tune_weights = 1 / tune.groupby("source1_entity_id").label.transform("count")
    model = lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=.05, num_leaves=31,
        min_child_samples=50, reg_lambda=4., max_bin=127, random_state=42, n_jobs=args.threads,
        deterministic=True, force_col_wise=True, verbosity=-1)
    model.fit(train[list(RANK_FEATURES)].astype(np.float32), train.label, sample_weight=weights,
        eval_set=[(tune[list(RANK_FEATURES)].astype(np.float32), tune.label)], eval_sample_weight=[tune_weights],
        callbacks=[lgb.early_stopping(50, verbose=False)])
    model.booster_.save_model(str(args.output / "model.txt"))
    report = {"status": "experimental_requires_retrieval_evaluation", "features": list(RANK_FEATURES),
        "train_entities": int(train.source1_entity_id.nunique()), "train_pairs": len(train),
        "tune_entities": int(tune.source1_entity_id.nunique()), "tune_pairs": len(tune),
        "best_iteration": model.best_iteration_, "seconds": time.perf_counter() - started,
        "training_pairs_sha256": hashlib.sha256((artifact / "pairs_train.parquet").read_bytes()).hexdigest(),
        "policy": "Training-fold labels only fit trees; tuning labels choose early stopping. No audit or test labels.",
        "limitation": "Initial ranker uses negatives retained by the earlier blocker, not all raw posting negatives.",
        "license": "MIT"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__": main()
