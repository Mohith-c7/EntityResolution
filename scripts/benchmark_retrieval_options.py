"""Controlled, paired end-to-end timing of read-only retrieval options.

Uses previously inspected tuning entities only. Does not modify indexes or models.
Run with other experiments stopped; measured throughput includes process startup,
retrieval, pair features and matching predictions but excludes final TSV validation.
"""
import argparse
import hashlib
import json
import multiprocessing
import platform
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.disk_index import DiskSearchConfig
from src.evaluation.metrics import score_matches
from src.pipeline import inference

_MMAP_EFFECTIVE = None


def initialize(model, paths, config, threshold, version, alias_path, mmap_bytes):
    global _MMAP_EFFECTIVE
    _MMAP_EFFECTIVE = {}
    inference.initialize_worker(model, paths, config, threshold, version, alias_path)
    for index in inference._WORKER[1]:
        value = index.connection.execute(f"PRAGMA main.mmap_size={int(mmap_bytes)}").fetchone()[0]
        _MMAP_EFFECTIVE[index.source] = value
        if mmap_bytes and value == 0: raise RuntimeError("SQLite mmap unavailable")
        if index.frequency_table.startswith("stats."):
            index.connection.execute(f"PRAGMA stats.mmap_size={int(mmap_bytes)}")


def score_chunk(rows):
    return inference.score_batch(rows), _MMAP_EFFECTIVE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=Path, default=Path("models/larger_alias_v4"))
    p.add_argument("--index-dir", type=Path, default=Path("models"))
    p.add_argument("--entities", type=int, default=600)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", type=Path, default=Path("reports/experiments/retrieval_runtime_options.json"))
    args = p.parse_args()
    report = json.loads((args.artifact_dir / "report.json").read_text())
    references = json.loads((args.artifact_dir / "sampled_references.json").read_text())["tune"]
    references = sorted(references, key=lambda r: hashlib.sha256(r["entity_id"].encode()).digest())[:args.entities]
    all_truth = json.loads((args.artifact_dir / "sampled_truth.json").read_text())
    truth = {r["entity_id"]: set(all_truth[r["entity_id"]]) for r in references}
    config = DiskSearchConfig(**report["search_config"])
    variants = {"baseline": (config, 0), "mmap": (config, 4 * 1024**3),
        "mmap_without_numeric_location": (replace(config, numeric_location_queries=False), 4 * 1024**3)}
    order = ["baseline", "mmap", "mmap_without_numeric_location", "mmap_without_numeric_location", "mmap", "baseline"]
    chunks = [references[i:i + 25] for i in range(0, len(references), 25)]
    outcomes = []; baseline_candidates = None; baseline_predictions = None
    for name in order:
        cfg, mmap_bytes = variants[name]
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize, initargs=(args.artifact_dir / "model.txt",
                [args.index_dir / f"index_train_{s}.sqlite" for s in ("S2", "S3")], cfg,
                report["threshold"], report["feature_version"], args.artifact_dir / "name_aliases.json", mmap_bytes)) as ex:
            batches = list(ex.map(score_chunk, chunks))
            rows = [row for batch, _ in batches for row in batch]
        seconds = time.perf_counter() - start
        candidates = {e: c for e, c, _ in rows}
        predictions = {e: c for e, _, c in rows}
        oracle = score_matches(truth, {e: truth[e] & set(ids) for e, ids in candidates.items()})
        if baseline_candidates is None:
            baseline_candidates, baseline_predictions = candidates, predictions
        result = {"variant": name, "seconds": seconds, "references_per_second": len(rows) / seconds,
            "requested_mmap_bytes": mmap_bytes, "effective_mmap_limits": batches[0][1],
            "entities": len(rows), "mean_candidates": sum(map(len, candidates.values())) / len(rows),
            "oracle_macro_f05": oracle["macro_f05"], "blocking_recall": oracle["micro_recall"],
            "matching": score_matches(truth, predictions),
            "exact_candidates_equal_baseline": candidates == baseline_candidates,
            "exact_predictions_equal_baseline": predictions == baseline_predictions}
        outcomes.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"workers": args.workers, "order": order,
            "sqlite_version": sqlite3.sqlite_version, "platform": platform.platform(),
            "reference_ids_sha256": hashlib.sha256("\n".join(r["entity_id"] for r in references).encode()).hexdigest(),
            "model": str(args.artifact_dir), "complete": len(outcomes) == len(order), "runs": outcomes}, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
