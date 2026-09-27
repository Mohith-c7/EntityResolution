"""Capture exact frozen-runner inputs and pair-keyed scores without changing it.

The baseline runner and its models are verified before use.  Transparent booster
wrappers retain the matrices passed to predict; predictions are delegated without
alteration.  Audit labels are never read by this script.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "code/business_entity_resolution"))
import numpy as np
import pandas as pd
import pyarrow as pa
import run_frozen_pipeline as frozen
from src.model.crossfit_aliases import inner_fold
from src.evaluation.validation import entity_fold

CAPTURE = {}
FEATURE_OUTPUT = None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def ids_digest(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def pair_digest(frame):
    h = hashlib.sha256()
    for s, c in zip(frame.source1_entity_id, frame.candidate_entity_id):
        h.update(f"{s}\t{c}\n".encode())
    return h.hexdigest()


class CapturingBooster:
    def __init__(self, model, kind):
        self.model, self.kind = model, kind

    def predict(self, matrix, *args, **kwargs):
        CAPTURE[self.kind] = np.asarray(matrix).copy()
        return self.model.predict(matrix, *args, **kwargs)


def initialize(config, aliases, counts, index_prefix, first_model, features):
    global FEATURE_OUTPUT
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    config = deepcopy(config)
    if first_model:
        config["first_stage"]["path"] = first_model
    frozen.init(config, aliases, index_prefix, counts)
    indexes, first, second, conn, cfg, alias_obj = frozen.WORKER
    frozen.WORKER = (indexes, CapturingBooster(first, "pair"),
                     CapturingBooster(second, "full"), conn, cfg, alias_obj)
    FEATURE_OUTPUT = Path(features)
    pa.set_cpu_count(1)


def capture_chunk(task):
    number, rows, output = task
    marker = Path(output) / f"{number:07d}.json"
    feature_file = FEATURE_OUTPUT / f"{number:07d}.parquet"
    feature_marker = feature_file.with_suffix(".json")
    if marker.exists():
        original = json.loads(marker.read_text())
        if original["input_ids_sha256"] != ids_digest([r["entity_id"] for r in rows]):
            raise ValueError("Existing score chunk belongs to other references")
        score_path = marker.with_suffix(".parquet")
        if frozen.digest(score_path) != original["parquet_sha256"]:
            raise ValueError("Score chunk checksum mismatch")
        if feature_marker.exists():
            saved = json.loads(feature_marker.read_text())
            if saved["input_ids_sha256"] != original["input_ids_sha256"]:
                raise ValueError("Captured features belong to different references")
            if frozen.digest(feature_file) != saved["features_sha256"]:
                raise ValueError("Feature chunk checksum mismatch")
            if saved["score_sha256"] != original["parquet_sha256"]:
                raise ValueError("Feature/score association mismatch")
            scored_keys = pd.read_parquet(score_path, columns=["source1_entity_id", "candidate_entity_id"])
            feature_keys = pd.read_parquet(feature_file, columns=["source1_entity_id", "candidate_entity_id"])
            if not scored_keys.equals(feature_keys) or pair_digest(scored_keys) != saved["pair_order_sha256"]:
                raise ValueError("Captured feature keys are misaligned")
            return saved
        # A score-only partial capture must be regenerated, but never overwrite
        # a source artifact: these files live in this script's new output tree.
        marker.rename(marker.with_suffix(".score_only.json"))
    CAPTURE.clear()
    result = frozen.score_chunk(task)
    score_path = Path(output) / f"{number:07d}.parquet"
    frame = pd.read_parquet(score_path)
    names = list(frozen.FEATURE_NAMES_V3) + frozen.CONTEXT_NAMES + list(frozen.COMPETITION_NAMES)
    matrix = CAPTURE.get("full", np.empty((0, len(names)), dtype=np.float32))
    if matrix.shape != (len(frame), len(names)):
        raise ValueError("Captured matrix is not aligned to emitted pair IDs")
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite captured input")
    features = pd.DataFrame(matrix, columns=names)
    features.insert(0, "candidate_entity_id", frame.candidate_entity_id.to_numpy())
    features.insert(0, "source1_entity_id", frame.source1_entity_id.to_numpy())
    temp = feature_file.with_suffix(".partial.parquet")
    features.to_parquet(temp, index=False)
    temp.replace(feature_file)
    saved = {**result, "score_sha256": result["parquet_sha256"],
             "features_sha256": frozen.digest(feature_file),
             "pair_order_sha256": pair_digest(frame)}
    write_json(feature_marker, saved)
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True,
                   help="JSON list of source records; no truth labels")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--counts", type=Path, required=True)
    p.add_argument("--universe", type=Path, default=Path("dataset/train/train_source1.tsv"))
    p.add_argument("--index-prefix", default="models/index_train")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--training-oof", action="store_true")
    p.add_argument("--split", choices=("train", "development", "confirmation", "audit", "cohort"), required=True)
    args = p.parse_args()
    if (args.split == "train") != args.training_oof:
        raise ValueError("Training requires owner-excluded first-stage OOF models")
    config = frozen.load_frozen(args.frozen)
    rows = json.loads(args.references.read_text())
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError("References must be a JSON list of raw records")
    ids = [r["entity_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate reference")
    if args.training_oof and any(entity_fold(e) != "train" for e in ids):
        raise ValueError("OOF training input crossed outer split")
    if args.split in ("confirmation", "audit", "cohort") and any(entity_fold(e) != "holdout" for e in ids):
        raise ValueError("New holdout input crossed outer split")
    fitted_owner_path = Path("models/bridge_v1/first_stage/train_references.json")
    fitted_owners = set(json.loads(fitted_owner_path.read_text()))
    stage2_report = json.loads(Path("models/bridge_v1/stacked_l255s42/report.json").read_text())
    if stage2_report.get("development_training_added") or stage2_report["training_references"] != len(fitted_owners):
        raise ValueError("Re-establish current second-stage supervised-owner ledger")
    owner_excluded = not bool(fitted_owners.intersection(ids))
    if args.split != "train" and not owner_excluded:
        raise ValueError("Evaluation references entered incumbent parameter training")
    args.output.mkdir(parents=True, exist_ok=True)
    chunks, features = args.output / "chunks", args.output / "feature_chunks"
    chunks.mkdir(exist_ok=True); features.mkdir(exist_ok=True)
    before = {"status": "running", "split": args.split,
              "frozen_sha256": frozen.digest(args.frozen),
              "references_sha256": frozen.digest(args.references),
              "reference_order_sha256": ids_digest(ids),
              "capture_script_sha256": frozen.digest(Path(__file__)),
              "first_stage_owner_excluded_oof": args.training_oof,
              "second_stage_owner_excluded": owner_excluded,
              "incumbent_fitted_owner_ids_sha256": frozen.digest(fitted_owner_path),
              "audit_labels_read": False}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ("split", "frozen_sha256", "references_sha256", "capture_script_sha256"):
            if old.get(key) != before[key]:
                raise ValueError(f"Resume provenance mismatch: {key}")
    write_json(manifest_path, before)
    started = time.perf_counter()
    frozen.build_universe_counts(args.universe, args.counts, args.workers)
    grouped = {}
    if args.training_oof:
        for fold in range(5):
            grouped[fold] = [r for r in rows if inner_fold(r["entity_id"]) == fold]
    else:
        grouped[None] = rows
    ordered_rows = [row for group in grouped.values() for row in group]
    write_json(args.output / "ordered_scored_references.json", ordered_rows)
    before["ordered_scored_reference_ids_sha256"] = ids_digest([r["entity_id"] for r in ordered_rows])
    tasks, number, results, done = [], 0, [], 0
    for fold, group in grouped.items():
        aliases = str(frozen.ALIASES / ("full.json" if fold is None else f"exclude_{fold}.json"))
        first = None if fold is None else str(Path("models/bridge_v1/oof") / f"model_fold_{fold}.txt")
        fold_tasks = []
        for start in range(0, len(group), args.batch_size):
            item = (number, group[start:start + args.batch_size], str(chunks))
            fold_tasks.append(item); tasks.append(item); number += 1
        with ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"), initializer=initialize,
                initargs=(config, aliases, str(args.counts), args.index_prefix, first, str(features))) as pool:
            for result in pool.map(capture_chunk, fold_tasks):
                results.append(result); done += result["entities"]
                if done % 2500 < args.batch_size:
                    print(json.dumps({"stage": "captured", "done": done, "total": len(rows),
                                      "seconds": time.perf_counter() - started}), flush=True)
    frames = [pd.read_parquet(chunks / f"{r['chunk']:07d}.parquet")
              for r in sorted(results, key=lambda r: r["chunk"])]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "first_stage", "second_stage", "probability"])
    if frame.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise ValueError("Duplicate scored pair")
    countries = {r["entity_id"]: r["country"].strip().casefold() for r in rows}
    frame["country"] = frame.source1_entity_id.map(countries)
    frame["candidate_order"] = frame.groupby("source1_entity_id", sort=False).cumcount()
    frame["candidate_source"] = frame.candidate_entity_id.str[:2]
    frame["accepted_threshold"] = frame.probability >= config["threshold"]
    frame["accepted"], _ = frozen.decide(frame, frame, config)
    if args.training_oof:
        frame["oof_excluded_fold"] = frame.source1_entity_id.map(inner_fold)
    pair_path = args.output / "pairs.parquet"
    frame.to_parquet(pair_path, index=False)
    write_json(args.output / "references.json", rows)
    write_json(args.output / "feature_schema.json", {
        "names": list(frozen.FEATURE_NAMES_V3) + frozen.CONTEXT_NAMES + list(frozen.COMPETITION_NAMES)})
    manifest = {**before, "status": "complete", "pairs": len(frame), "entities": len(rows),
                "reference_ids": ids, "pairs_sha256": frozen.digest(pair_path),
                "pair_order_sha256": pair_digest(frame), "verified_current_pipeline": False,
                "verification_scope": "frozen source/model hashes; exact delegated runner scoring; independent replay still required",
                "decision_scope": "declared cohort only; complete outside-rival evaluation pending",
                "first_stage_training_input": "inner-fold-excluded booster and aliases" if args.training_oof else "frozen full model",
                "second_stage_training_input": "in-sample baseline scores; never advertise as OOF" if args.training_oof else "outer-training baseline",
                "chunks": results, "elapsed_seconds": time.perf_counter() - started}
    write_json(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "entities", "pairs", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
