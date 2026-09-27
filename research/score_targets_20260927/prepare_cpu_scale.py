"""Stream a larger pre-bridge CPU feature pilot from the existing caches.

This preparation reads only the immutable ``train`` and ``tune`` caches plus
the reserved training records and raw source2/source3 TSVs.  It never reads an
audit, holdout, or test file.  Its output contract is consumed by
``cpu_feature_experiment.py train``.

The experiment remains exploratory: both validation folds are old tuning-pool
rows from the pre-bridge retrieval cache, so results are not evidence of an
improvement over the final-pipeline 0.97982 development score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

# Cap native libraries before importing NumPy/PyArrow.  The command can lower
# their explicit worker counts further through --threads.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cpu_feature_experiment import (
    FEATURE_NAMES_65,
    NEW_FEATURE_NAMES,
    RawRecord,
    digest,
    pair_features,
    write_json,
)


PAIR_KEYS = ["source1_entity_id", "candidate_entity_id"]
OUTPUT_COLUMNS = [*PAIR_KEYS, "label", "sample_weight", "country",
                  *FEATURE_NAMES_65, *NEW_FEATURE_NAMES]
FORBIDDEN_PATH_PARTS = {"audit", "holdout", "test", "testing"}


def reject_forbidden_input(path: Path) -> None:
    tokens = {token for part in Path(path).parts
              for token in re.split(r"[^a-z]+", part.casefold()) if token}
    forbidden = tokens & FORBIDDEN_PATH_PARTS
    if forbidden:
        raise ValueError(f"Forbidden audit/holdout/test input path: {path} ({sorted(forbidden)})")


def _hash_order(records, namespace: str):
    return sorted(records, key=lambda row: hashlib.sha256(
        (namespace + row["entity_id"]).encode()).digest())


def choose_reference_folds(reservation: dict, train_references: int = 100_000):
    """Return fixed, mutually exclusive stage-two reference records."""
    train = _hash_order(reservation["train"], "cpu-scale-stage2-v1:")[:train_references]
    tune = _hash_order(reservation["tune"], "context-split-v1:")
    if len(train) != train_references:
        raise ValueError("Requested more training references than reserved")
    if len(tune) < 50_000:
        raise ValueError("The fixed context split requires at least 50,000 tune references")
    folds = {"train": train, "early_stop": tune[20_000:30_000],
             "selection": tune[30_000:50_000]}
    ids = {fold: {row["entity_id"] for row in records} for fold, records in folds.items()}
    if any(ids[left] & ids[right] for left, right in
           (("train", "early_stop"), ("train", "selection"), ("early_stop", "selection"))):
        raise ValueError("Reserved reference folds overlap")
    return folds


def load_cache_index(cache: Path, split: str, wanted: set[str]):
    reject_forbidden_input(cache)
    manifest_path, progress_path = cache / "manifest.json", cache / "progress.json"
    manifest = json.loads(manifest_path.read_text())
    progress = json.loads(progress_path.read_text())
    if manifest.get("split") != split or manifest.get("feature_version") != "pairwise-v3-65":
        raise ValueError(f"Wrong cache contract for {split}: {cache}")
    if progress.get("status") != "complete":
        raise ValueError(f"Incomplete {split} cache")
    selected_chunks, entity_rows = [], {}
    for number in range(progress["chunks"]):
        marker_path = cache / f"{number:06d}.json"
        marker = json.loads(marker_path.read_text())
        selected = [row for row in marker["entity_rows"] if row["entity_id"] in wanted]
        if not selected:
            continue
        parquet = marker_path.with_suffix(".parquet")
        if digest(parquet) != marker["parquet_sha256"]:
            raise ValueError(f"Corrupt cache chunk: {parquet}")
        selected_chunks.append((parquet, selected))
        for row in selected:
            entity = row["entity_id"]
            if entity in entity_rows:
                raise ValueError(f"Duplicate cached reference: {entity}")
            entity_rows[entity] = row
    if set(entity_rows) != wanted:
        raise ValueError(f"Missing {len(wanted - set(entity_rows))} selected references in {split} cache")
    return selected_chunks, entity_rows, {
        "manifest_sha256": digest(manifest_path), "progress_sha256": digest(progress_path),
        "selected_chunks": len(selected_chunks), "cache_split": split,
    }


def collect_candidate_ids(indexes) -> set[str]:
    """Read only pair IDs on pass one; do not retain any feature rows."""
    candidates = set()
    for fold, (chunks, entity_rows, _) in indexes.items():
        wanted = set(entity_rows)
        observed = {entity: 0 for entity in wanted}
        for parquet, selected in chunks:
            selected_ids = {row["entity_id"] for row in selected}
            frame = pd.read_parquet(parquet, columns=PAIR_KEYS)
            frame = frame[frame.source1_entity_id.isin(selected_ids)]
            if frame.duplicated(PAIR_KEYS).any():
                raise ValueError(f"Duplicate pair IDs in {fold}: {parquet}")
            candidates.update(frame.candidate_entity_id)
            counts = frame.source1_entity_id.value_counts().to_dict()
            for entity in selected_ids:
                observed[entity] += int(counts.get(entity, 0))
        for entity, count in observed.items():
            expected = int(entity_rows[entity]["retained_pairs"])
            if count != expected:
                raise ValueError(f"Pair-count mismatch for {entity}: {count} != {expected}")
    return candidates


def load_raw_records(folds, candidate_ids: set[str], source2: Path, source3: Path):
    references = {}
    for records in folds.values():
        for row in records:
            entity = row["entity_id"]
            if entity in references:
                raise ValueError(f"Duplicate raw reference: {entity}")
            references[entity] = RawRecord(row["business_name"], row["business_address"], row["country"])
    targets = {}
    for source_number, path in ((2, source2), (3, source3)):
        reject_forbidden_input(path)
        wanted = {item for item in candidate_ids if item.startswith(f"S{source_number}-")}
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=100_000):
            for row in chunk[chunk.entity_id.isin(wanted)].itertuples(index=False):
                if row.entity_id in targets:
                    raise ValueError(f"Duplicate raw target: {row.entity_id}")
                targets[row.entity_id] = RawRecord(row.business_name, row.business_address, row.country)
    if set(targets) != candidate_ids:
        raise ValueError(f"Missing {len(candidate_ids - set(targets))} raw candidate records")
    return references, targets


def feature_rows(left_ids, right_ids, references, targets):
    """Compute only one cache chunk's feature dictionaries at a time."""
    return [pair_features(references[left], targets[right])
            for left, right in zip(left_ids, right_ids)]


def prepare_chunk(frame: pd.DataFrame, selected_rows, references, targets, fold: str):
    selected = {row["entity_id"]: row for row in selected_rows}
    frame = frame[frame.source1_entity_id.isin(selected)].copy()
    if frame.duplicated(PAIR_KEYS).any():
        raise ValueError(f"Duplicate pair IDs in {fold} chunk")
    counts = frame.source1_entity_id.value_counts().to_dict()
    for entity, metadata in selected.items():
        if int(counts.get(entity, 0)) != int(metadata["retained_pairs"]):
            raise ValueError(f"Pair-count mismatch for {entity} in {fold}")
    expected_labels = np.fromiter(
        (candidate in set(selected[reference]["true_ids"])
         for reference, candidate in frame[PAIR_KEYS].itertuples(index=False, name=None)),
        dtype=np.uint8, count=len(frame))
    if not np.array_equal(expected_labels, frame.label.to_numpy(dtype=np.uint8)):
        raise ValueError(f"Cached labels disagree with full truth in {fold}")
    frame["sample_weight"] = frame.source1_entity_id.map(
        {entity: 1.0 / metadata["retained_pairs"] for entity, metadata in selected.items()}).astype(np.float32)
    frame["country"] = frame.source1_entity_id.map({entity: references[entity].country for entity in selected})
    extras = pd.DataFrame(feature_rows(frame.source1_entity_id, frame.candidate_entity_id,
                                       references, targets), index=frame.index,
                          columns=NEW_FEATURE_NAMES)
    for name in NEW_FEATURE_NAMES:
        frame[name] = extras[name]
    frame = frame[OUTPUT_COLUMNS]
    numeric = frame[["label", "sample_weight", *FEATURE_NAMES_65, *NEW_FEATURE_NAMES]].to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError(f"Nonfinite feature in {fold} chunk")
    return frame


def combine_parquet_chunks(paths: list[Path], output: Path) -> int:
    if not paths:
        raise ValueError(f"No chunks to combine for {output}")
    writer = None
    rows = 0
    temporary = output.with_suffix(output.suffix + ".partial")
    try:
        for path in paths:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            elif table.schema != writer.schema:
                raise ValueError(f"Chunk schema mismatch: {path}")
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output)
    return rows


def prepare(args) -> None:
    started = time.perf_counter()
    for path in (args.reservation, args.train_cache, args.tune_cache, args.source2, args.source3):
        reject_forbidden_input(path)
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    if not 1 <= args.threads <= 8:
        raise ValueError("Preparation requires 1..8 threads")
    pa.set_cpu_count(args.threads)
    pa.set_io_thread_count(args.threads)
    args.output.mkdir(parents=True)
    chunk_root = args.output / "chunks"
    chunk_root.mkdir()

    reservation = json.loads(args.reservation.read_text())
    folds = choose_reference_folds(reservation, args.train_references)
    del reservation
    fold_ids = {fold: {row["entity_id"] for row in records} for fold, records in folds.items()}
    indexes = {
        "train": load_cache_index(args.train_cache, "train", fold_ids["train"]),
        "early_stop": load_cache_index(args.tune_cache, "tune", fold_ids["early_stop"]),
        "selection": load_cache_index(args.tune_cache, "tune", fold_ids["selection"]),
    }
    truths = {fold: {entity: metadata["true_ids"] for entity, metadata in value[1].items()}
              for fold, value in indexes.items()}
    owned = {fold: {target for targets in truth.values() for target in targets}
             for fold, truth in truths.items()}
    if any(owned[left] & owned[right] for left, right in
           (("train", "early_stop"), ("train", "selection"), ("early_stop", "selection"))):
        raise ValueError("Owned truth targets overlap across folds")

    candidates = collect_candidate_ids(indexes)
    print(json.dumps({"stage": "candidate_inventory", "unique_candidates": len(candidates)}), flush=True)
    references, targets = load_raw_records(folds, candidates, args.source2, args.source3)
    del candidates, folds
    print(json.dumps({"stage": "raw_records_ready", "references": len(references),
                      "targets": len(targets)}), flush=True)

    summaries = {}
    cache_columns = [*FEATURE_NAMES_65, *PAIR_KEYS, "label"]
    # Emit the two smaller validation folds first so they can be transferred
    # while the larger training fold is still being prepared.
    for fold in ("early_stop", "selection", "train"):
        fold_dir = chunk_root / fold
        fold_dir.mkdir()
        chunk_paths = []
        positive_pairs = 0
        for number, (parquet, selected_rows) in enumerate(indexes[fold][0]):
            raw = pd.read_parquet(parquet, columns=cache_columns)
            frame = prepare_chunk(raw, selected_rows, references, targets, fold)
            chunk_path = fold_dir / f"{number:06d}.parquet"
            frame.to_parquet(chunk_path, index=False, compression="zstd")
            chunk_paths.append(chunk_path)
            positive_pairs += int(frame.label.sum())
            if (number + 1) % 50 == 0:
                print(json.dumps({"stage": "feature_chunks", "fold": fold,
                                  "chunks": number + 1}), flush=True)
        final_path = args.output / f"{fold}.parquet"
        pair_count = combine_parquet_chunks(chunk_paths, final_path)
        truth_path = args.output / f"{fold}_truth.json"
        write_json(truth_path, truths[fold])
        summaries[fold] = {
            "references": len(fold_ids[fold]), "pairs": pair_count,
            "positive_pairs": positive_pairs, "parquet_sha256": digest(final_path),
            "truth_sha256": digest(truth_path), **indexes[fold][2],
        }
        if not args.keep_chunks:
            for path in chunk_paths:
                path.unlink()
            fold_dir.rmdir()
        print(json.dumps({"stage": "fold_complete", "fold": fold, **summaries[fold]}), flush=True)
    if not args.keep_chunks:
        chunk_root.rmdir()

    manifest = {
        "status": "prepared_not_trained", "experiment": "cpu-scale-stage2-prebridge",
        "feature_schema": "pairwise-v3-65+cpu-pilot-12", "baseline_features": list(FEATURE_NAMES_65),
        "new_features": list(NEW_FEATURE_NAMES), "folds": summaries,
        "selection_contract": {
            "train": f"{args.train_references} constant-hash-selected reserved train references",
            "early_stop": "context-split-v1 sorted tune references [20000:30000]",
            "selection": "context-split-v1 sorted tune references [30000:50000]",
            "unused_tune_development": "context-split-v1 [0:20000]",
            "candidates": "all retained cache candidates; no negative downsampling",
            "weights": "equal entity weight: every pair receives 1/retained_pairs for its reference",
        },
        "input_hashes": {"reservation": digest(args.reservation),
                         "source2": digest(args.source2), "source3": digest(args.source3)},
        "source_sha256": digest(Path(__file__)), "threads": args.threads,
        "data_scope": "existing train/tune pre-bridge caches only; no audit, holdout, or test data",
        "comparison_scope": "exploratory old-pool pilot; not comparable to final-pipeline 0.97982",
        "seconds": time.perf_counter() - started,
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"stage": "complete", "output": str(args.output),
                      "seconds": manifest["seconds"]}), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservation", type=Path,
                        default=Path("models/scale_v1_plan/sampled_references.json"))
    parser.add_argument("--train-cache", type=Path, default=Path("models/scale_v1_run/cache_train"))
    parser.add_argument("--tune-cache", type=Path, default=Path("models/scale_v1_run/cache_tune"))
    parser.add_argument("--source2", type=Path, default=Path("dataset/train/train_source2.tsv"))
    parser.add_argument("--source3", type=Path, default=Path("dataset/train/train_source3.tsv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-references", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--keep-chunks", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.train_references < 1:
        raise ValueError("--train-references must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
