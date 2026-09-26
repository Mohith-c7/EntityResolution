"""Train a complementary matcher without alias-derived or blocking-score inputs."""
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
from src.features.registry import FEATURE_NAMES_V3, FEATURE_NAMES_V2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("models/redesign/alias_free_control"))
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.nice(10); pa.set_cpu_count(1)
    started = time.perf_counter()
    artifact = Path("models/features_v3_audit01")
    cache = Path("models/redesign/wide_tune_cache")
    manifest = json.loads((cache / "manifest.json").read_text())
    if not manifest["complete"] or manifest["test"]: raise ValueError("Not a complete tuning cache")
    removed = set(FEATURE_NAMES_V2[36:44]) | {"blocking_score"}
    names = [name for name in FEATURE_NAMES_V3 if name not in removed]
    protocol = {"features": names, "removed": sorted(removed), "blend_weights_for_new_model": [0., .1, .25, .5, .75, 1.],
        "selection": "Tuning macro F0.5 with precision >= .9847119 and singleton errors <= 33; fresh audit required.",
        "purpose": "Test dependence on supervised aliases and retrieval score; complement the existing matcher.",
        "limitations": "Reuses 20k training entities and earlier blocked hard negatives; does not substitute for larger training or cross-fitted aliases."}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    train = pd.read_parquet(artifact / "pairs_train.parquet", columns=["source1_entity_id", "label", *names])
    tune = pd.concat([pd.read_parquet(cache / file, columns=["source1_entity_id", "candidate_entity_id", "probability", *names])
        for file in manifest["files"]], ignore_index=True)
    labels = json.loads((artifact / "sampled_truth.json").read_text())
    entities = manifest["reference_ids"]
    truth = {e: set(labels[e]) for e in entities}
    if set(train.source1_entity_id) & set(entities): raise ValueError("Entity fold overlap")
    y = np.asarray([c in truth[e] for e, c in tune[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None)])
    weights = 1 / train.groupby("source1_entity_id").label.transform("count")
    tune_weights = 1 / tune.groupby("source1_entity_id").probability.transform("count")
    model = lgb.LGBMClassifier(objective="binary", n_estimators=1200, learning_rate=.04, num_leaves=31,
        min_child_samples=30, reg_lambda=2., colsample_bytree=.9, subsample=.9, subsample_freq=1,
        max_bin=127, random_state=42, n_jobs=args.threads, deterministic=True, force_col_wise=True, verbosity=-1)
    model.fit(train[names].astype(np.float32), train.label, sample_weight=weights,
        eval_set=[(tune[names].astype(np.float32), y)], eval_sample_weight=[tune_weights],
        callbacks=[lgb.early_stopping(75, verbose=False)])
    model.booster_.save_model(str(args.output / "model.txt"))
    alternate = model.predict_proba(tune[names].astype(np.float32))[:, 1]
    position = {e: i for i, e in enumerate(entities)}
    groups = np.fromiter((position[e] for e in tune.source1_entity_id), dtype=np.int32)
    expected = np.array([len(truth[e]) for e in entities])
    old = tune.probability.to_numpy()
    trials = []
    for weight in protocol["blend_weights_for_new_model"]:
        probabilities = (1 - weight) * old + weight * alternate
        for threshold in np.r_[np.arange(.30, .951, .01), .96, .97, .98, .99]:
            chosen = probabilities >= threshold
            predicted = np.bincount(groups[chosen], minlength=len(entities))
            tp = np.bincount(groups[chosen & y], minlength=len(entities))
            scores = np.zeros(len(entities))
            np.divide(1.25 * tp, .25 * expected + predicted, out=scores, where=(.25 * expected + predicted) > 0)
            scores[(expected == 0) & (predicted == 0)] = 1.
            trials.append({"new_model_weight": weight, "threshold": round(float(threshold), 6),
                "macro_f05": float(scores.mean()), "micro_precision": float(tp.sum() / max(predicted.sum(), 1)),
                "micro_recall": float(tp.sum() / expected.sum()),
                "singleton_false_positives": int(((expected == 0) & (predicted > 0)).sum())})
    eligible = [r for r in trials if r["micro_precision"] >= .9847119 and r["singleton_false_positives"] <= 33]
    best = max(eligible, key=lambda r: (r["macro_f05"], r["micro_precision"]))
    result = {"status": "exploratory_tuning_result_not_promoted", "chosen": best, "feature_count": len(names),
        "best_iteration": model.best_iteration_, "seconds": time.perf_counter()-started,
        "training_entities": int(train.source1_entity_id.nunique()), "training_pairs": len(train),
        "fresh_holdout_evaluated": False, "training_pairs_sha256": hashlib.sha256((artifact / "pairs_train.parquet").read_bytes()).hexdigest()}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "threshold_sweep.json").write_text(json.dumps(trials, indent=2))
    tune[["source1_entity_id", "candidate_entity_id"]].assign(probability=alternate, baseline_probability=old).to_parquet(args.output / "predictions_tune.parquet", index=False)
    Path("reports/experiments/redesign/alias_free_control.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__": main()
