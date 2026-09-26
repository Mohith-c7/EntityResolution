"""Resumable cross-fitted training/tuning features over full target indexes.

Audit caching requires a previously frozen model selection. Each chunk records
its supervised alias state, input IDs, schema and training counterpart exclusions.
"""
import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.disk_index import DiskSearchConfig, normalize_record
from src.blocking.postings_index import PostingsConfig, PostingsSourceIndex
from src.evaluation.metrics import entity_f05
from src.evaluation.validation import entity_fold
from src.features.pairwise_features import build_versioned_pair_features
from src.features.registry import FEATURE_VERSION_V3, FEATURE_NAMES_V3
from src.model.crossfit_aliases import inner_fold
from src.model.name_aliases import NameAliases

STATE = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n"); temporary.replace(path)


def init(index_dir, aliases_path, database, search, postings, native, split, output):
    global STATE
    os.nice(10); pa.set_cpu_count(1)
    aliases = NameAliases.load(aliases_path)
    indexes = [PostingsSourceIndex(Path(index_dir) / f"index_train_{s}.sqlite", DiskSearchConfig(**search),
        aliases=aliases, postings_config=PostingsConfig(**postings), native_extension=native) for s in ("S2", "S3")]
    conn = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size=-16384")
    conn.execute("PRAGMA mmap_size=2147418112")
    STATE = indexes, conn, aliases, split, Path(output)


def make_chunk(task):
    number, raw_rows = task
    indexes, conn, aliases, split, output = STATE
    ids = [r["entity_id"] for r in raw_rows]
    if any(entity_fold(e) != split for e in ids): raise ValueError("Reference in wrong outer fold")
    excluded = aliases.metadata["excluded_inner_fold"]
    if split == "train" and any(inner_fold(e) != excluded for e in ids): raise ValueError("Training aliases include the reference's inner fold")
    if split != "train" and excluded is not None: raise ValueError("Evaluation needs outer-training aliases")
    placeholders = ",".join("?" for _ in ids)
    truth = {e: set() for e in ids}
    for target, owner in conn.execute(f"SELECT id,owner FROM owners WHERE owner IN ({placeholders})", ids):
        truth[owner].add(target)
    groups = []
    for raw in raw_rows:
        reference = normalize_record(raw)
        pairs = [(candidate, target, index) for index in indexes for candidate, target in index.query(reference)]
        groups.append((reference, pairs))
    roles = {}
    if split == "train":
        candidates = sorted({c.candidate_entity_id for _, pairs in groups for c, _, _ in pairs})
        for offset in range(0, len(candidates), 500):
            chunk = candidates[offset:offset+500]; marks = ",".join("?" for _ in chunk)
            roles.update(conn.execute(f"SELECT id,outer_fold FROM owners WHERE id IN ({marks})", chunk))
    metadata, features, entity_rows = [], [], []
    retrieved = excluded_targets = true_count = retrieved_true = 0
    oracle_sum = 0.
    for reference, pairs in groups:
        actual = truth[reference.entity_id]
        available = {c.candidate_entity_id for c, _, _ in pairs} & actual
        retrieved += len(pairs); true_count += len(actual); retrieved_true += len(available)
        oracle_sum += entity_f05(actual, available)
        accepted = [(c,t,i) for c,t,i in pairs if split != "train" or roles.get(c.candidate_entity_id, entity_fold(c.candidate_entity_id)) == "train"]
        excluded_targets += len(pairs) - len(accepted)
        if {c.candidate_entity_id for c, _, _ in accepted} & actual != available:
            raise ValueError("A positive training target crossed folds")
        weight = 1 / len(accepted) if accepted else 0.
        for candidate, target, index in accepted:
            values = build_versioned_pair_features(reference, target, candidate, index, FEATURE_VERSION_V3)
            features.append([values[name] for name in FEATURE_NAMES_V3])
            metadata.append((reference.entity_id, candidate.candidate_entity_id, candidate.candidate_source,
                candidate.rank_within_source, ",".join(candidate.blocking_paths), int(target.entity_id in actual), weight))
        entity_rows.append({"entity_id": reference.entity_id, "country": reference.country,
            "true_ids": sorted(actual), "retained_pairs": len(accepted), "retrieved_pairs": len(pairs)})
    frame = pd.DataFrame(np.asarray(features, dtype=np.float32).reshape(-1, len(FEATURE_NAMES_V3)), columns=FEATURE_NAMES_V3)
    for n, name in enumerate(("source1_entity_id", "candidate_entity_id", "candidate_source", "rank_within_source", "blocking_paths", "label", "sample_weight")):
        frame[name] = [row[n] for row in metadata]
    frame["sample_weight"] = frame.sample_weight.astype(np.float32)
    path = output / f"{number:06d}.parquet"
    temporary = path.with_suffix(".partial.parquet"); frame.to_parquet(temporary, index=False); temporary.replace(path)
    result = {"chunk": number, "entities": len(raw_rows), "pairs": len(frame), "retrieved_pairs": retrieved,
        "training_counterparts_excluded": excluded_targets, "true_links": true_count,
        "retrieved_true_links": retrieved_true, "oracle_sum": oracle_sum,
        "input_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "parquet_sha256": digest(path), "entity_rows": entity_rows}
    write_json(output / f"{number:06d}.json", result)
    return {k:v for k,v in result.items() if k != "entity_rows"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=Path("models/scale_v1_plan"))
    p.add_argument("--split", choices=("train", "tune", "holdout"), required=True)
    p.add_argument("--audit-selection", type=Path)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--idle-workers", type=int, default=10)
    p.add_argument("--submission-progress", type=Path, default=Path("output/submission_03/progress.json"))
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.limit < 0 or min(args.workers, args.idle_workers, args.batch_size) < 1: p.error("Invalid size or worker count")
    selection = None
    if args.split == "holdout":
        if not args.audit_selection or args.limit: p.error("Audit requires a frozen selection and the entire reserved holdout")
        selection = json.loads(args.audit_selection.read_text())
        if selection.get("status") != "frozen_before_audit" or digest(selection["model_path"]) != selection["model_sha256"]:
            raise ValueError("Invalid frozen audit selection")
    references = json.loads((args.plan / "sampled_references.json").read_text())[args.split]
    references = sorted(references, key=lambda r: hashlib.sha256(r["entity_id"].encode()).digest())
    if args.limit: references = references[:args.limit]
    if args.split == "train":
        groups = {fold: [r for r in references if inner_fold(r["entity_id"]) == fold] for fold in range(5)}
    else: groups = {None: references}
    search = json.loads(Path("models/features_v3_audit01/report.json").read_text())["search_config"]
    postings = asdict(PostingsConfig(max_postings=100000))
    aliases_dir = args.plan / "aliases"
    native = Path("models/runtime/erpostings.dylib").resolve()
    database = (aliases_dir / "counts.sqlite").resolve()
    alias_paths = {fold: (aliases_dir / ("full.json" if fold is None else f"exclude_{fold}.json")).resolve() for fold in groups}
    source_root = Path(__file__).resolve().parents[1]
    code_files = [Path(__file__).resolve(), *sorted((source_root / "code/business_entity_resolution/src").rglob("*.py")),
                  source_root / "code/business_entity_resolution/src/blocking/pack_postings.c"]
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as conn:
        counts_fingerprint = json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
    manifest = {"split": args.split, "references": len(references), "batch_size": args.batch_size,
        "feature_version": FEATURE_VERSION_V3, "search": search, "postings": postings,
        "reference_ids_sha256": hashlib.sha256("\n".join(r["entity_id"] for r in references).encode()).hexdigest(),
        "aliases": {str(f): {"path": str(path), "sha256": digest(path)} for f,path in alias_paths.items()},
        "native_sha256": digest(native), "code_sha256": {str(path): digest(path) for path in code_files},
        "counts_fingerprint": counts_fingerprint, "audit_selection": selection}
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "build.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest: raise ValueError("Cache input/configuration mismatch")
    write_json(manifest_path, manifest)
    started = time.perf_counter(); completed = 0; total_pairs = 0; global_chunk = 0
    for fold, rows in groups.items():
        tasks = []
        for offset in range(0, len(rows), args.batch_size):
            raw = rows[offset:offset+args.batch_size]; number = global_chunk; global_chunk += 1
            marker = args.output / f"{number:06d}.json"
            if marker.exists():
                meta = json.loads(marker.read_text())
                expected = hashlib.sha256("\n".join(r["entity_id"] for r in raw).encode()).hexdigest()
                if meta["input_ids_sha256"] != expected or digest(args.output / f"{number:06d}.parquet") != meta["parquet_sha256"]:
                    raise ValueError("Corrupt training cache checkpoint")
                completed += meta["entities"]; total_pairs += meta["pairs"]
            else: tasks.append((number, raw))
        cursor = 0
        while cursor < len(tasks):
            if any(digest(path) != expected for path, expected in manifest["code_sha256"].items()):
                raise ValueError("Source changed during cache generation; use a frozen source snapshot")
            # Reassess CPU allocation every bounded wave, never competing with a running submission at full width.
            progress = json.loads(args.submission_progress.read_text()) if args.submission_progress.exists() else {}
            workers = args.idle_workers if progress.get("status") == "complete" else args.workers
            wave = tasks[cursor:cursor+workers*20]; cursor += len(wave)
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=init,
                initargs=(str(Path("models").resolve()), str(alias_paths[fold]), str(database), search, postings,
                          str(native), args.split, str(args.output.resolve()))) as pool:
                for result in pool.map(make_chunk, wave):
                    completed += result["entities"]; total_pairs += result["pairs"]
                    state = {"status": "building", "split": args.split, "entities": completed,
                        "expected_entities": len(references), "pairs": total_pairs, "inner_fold": fold,
                        "workers": workers, "seconds": time.perf_counter()-started}
                    write_json(args.output / "progress.json", state)
                    if completed % 5000 == 0: print(json.dumps(state), flush=True)
    write_json(args.output / "progress.json", {"status": "complete", "split": args.split,
        "entities": completed, "pairs": total_pairs, "chunks": global_chunk, "seconds": time.perf_counter()-started})
    print(json.dumps(json.loads((args.output / "progress.json").read_text())), flush=True)


if __name__ == "__main__": main()
