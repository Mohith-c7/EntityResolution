"""Resume a frozen submission, validate it, then publish complete output files.

The frozen package supplies candidate generation, preprocessing and scoring.
This runner adds durable chunks and finalization without changing model behavior.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def id_digest(rows):
    return hashlib.sha256("\n".join(r["entity_id"] for r in rows).encode()).hexdigest()


def initialize(package, model, indexes, config, threshold, version, aliases):
    sys.path.insert(0, package)
    from src.pipeline.inference import initialize_worker
    from src.blocking.disk_index import DiskSearchConfig
    initialize_worker(model, indexes, DiskSearchConfig(**config), threshold, version, aliases)


def score(rows):
    from src.pipeline.inference import score_batch
    return score_batch(rows)


def save_chunk(directory, number, rows, results):
    if [r["entity_id"] for r in rows] != [r[0] for r in results]:
        raise ValueError("Worker result order differs from its Source 1 input")
    paths = [directory / f"{number:07d}.{kind}.tsv" for kind in ("matching", "candidate")]
    stats = {"entities": len(rows), "predicted_links": 0, "scored_candidates": 0,
             "empty_predictions": 0, "empty_candidates": 0, "candidate_histogram": {}, "country": {}}
    histogram = Counter()
    with paths[0].with_suffix(".partial").open("w") as m, paths[1].with_suffix(".partial").open("w") as c:
        for raw, (eid, candidates, predictions) in zip(rows, results):
            if len(candidates) != len(set(candidates)) or len(predictions) != len(set(predictions)):
                raise ValueError("Duplicate candidate or prediction")
            if not set(predictions).issubset(candidates):
                raise ValueError("A prediction was not a scored candidate")
            m.write(eid + "\t" + ",".join(predictions) + "\n")
            c.write(eid + "\t" + ",".join(candidates) + "\n")
            stats["predicted_links"] += len(predictions)
            stats["scored_candidates"] += len(candidates)
            stats["empty_predictions"] += not predictions
            stats["empty_candidates"] += not candidates
            histogram[len(candidates)] += 1
            country = raw["country"].strip().casefold()
            counter = stats["country"].setdefault(country, {"entities": 0, "predicted_links": 0,
                "empty_predictions": 0, "scored_candidates": 0})
            counter["entities"] += 1
            counter["predicted_links"] += len(predictions)
            counter["empty_predictions"] += not predictions
            counter["scored_candidates"] += len(candidates)
        for stream in (m, c):
            stream.flush()
            os.fsync(stream.fileno())
    for path in paths:
        path.with_suffix(".partial").replace(path)
    stats["candidate_histogram"] = dict(histogram)
    stats["input_id_sha256"] = id_digest(rows)
    stats["files"] = {path.name: digest(path) for path in paths}
    # The JSON marker is committed last; both files must exist to resume a chunk.
    write_json(directory / f"{number:07d}.json", stats)
    return stats


def add_stats(total, stats):
    for field in ("entities", "predicted_links", "scored_candidates", "empty_predictions", "empty_candidates"):
        total[field] += stats[field]
    for key, count in stats["candidate_histogram"].items():
        total["candidate_histogram"][int(key)] += count
    for country, counters in stats["country"].items():
        for key, count in counters.items():
            total["country"].setdefault(country, Counter())[key] += count


def percentile(histogram, q, count):
    cumulative = 0
    for value, frequency in sorted(histogram.items()):
        cumulative += frequency
        if cumulative >= q * count:
            return value
    return 0


def run(args):
    start = time.perf_counter()
    submission, test, indexes = args.submission_dir.resolve(), args.test_dir.resolve(), args.index_dir.resolve()
    package, model = submission / "code/business_entity_resolution", submission / "model"
    sys.path.insert(0, str(package))
    from src.pipeline.inference import batches
    from src.pipeline.export import validate_submission
    report = json.loads((model / "report.json").read_text())
    compatibility = json.loads((submission / "compatibility_report.json").read_text())
    if not compatibility["compatible"]:
        raise ValueError("Compatibility preflight failed")
    frozen = json.loads((submission / "frozen_assets_sha256.json").read_text())
    for name, expected in frozen.items():
        if digest(submission / name) != expected:
            raise ValueError(f"Frozen asset changed: {name}")
    checkpoint = submission / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    test_info = {f"test_source{n}.tsv": {"path": str(test / f"test_source{n}.tsv"),
        "sha256": digest(test / f"test_source{n}.tsv"), "bytes": (test / f"test_source{n}.tsv").stat().st_size} for n in (1, 2, 3)}
    with (test / "test_source1.tsv").open() as stream:
        expected_entities = sum(1 for line in stream) - 1
    manifest = {"version": "frozen-submission-chunks-v1", "test_files": test_info,
        "frozen_assets": frozen, "batch_size": args.batch_size, "expected_entities": expected_entities,
        "config": report["search_config"], "feature_version": report["feature_version"],
        "threshold": report["threshold"], "local_validation_macro_f05": report["holdout"]["macro_f05"],
        "leaderboard_score": None}
    manifest_path = submission / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Checkpoint configuration/input mismatch")
    write_json(manifest_path, manifest)
    total = {"entities": 0, "predicted_links": 0, "scored_candidates": 0, "empty_predictions": 0,
        "empty_candidates": 0, "candidate_histogram": Counter(), "country": {}}
    resumed = generated = 0
    def publish_progress():
        elapsed = time.perf_counter() - start
        rate = generated / elapsed if elapsed else 0
        state = {**total, "status": "inference_running", "expected_entities": expected_entities,
            "resumed_entities": resumed, "generated_this_run": generated,
            "elapsed_seconds": elapsed, "entities_per_second_this_run": rate,
            "estimated_remaining_seconds": (expected_entities-total["entities"])/rate if rate else None,
            "ready_for_upload": False}
        write_json(submission / "progress.json", state)
        print(json.dumps({k: v for k, v in state.items() if k not in ("country", "candidate_histogram")}), flush=True)
    tasks = iter(enumerate(batches(test / "test_source1.tsv", args.batch_size)))
    initargs = (str(package), str(model / "model.txt"),
        [str(indexes / f"index_test_{s}.sqlite") for s in ("S2", "S3")],
        report["search_config"], report["threshold"], report["feature_version"], str(model / "name_aliases.json"))
    pending = deque()
    all_chunks = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize, initargs=initargs) as executor:
        def fill():
            nonlocal resumed
            while len(pending) < args.workers * 2:
                task = next(tasks, None)
                if task is None:
                    return
                number, rows = task
                all_chunks.append(number)
                marker = checkpoint / f"{number:07d}.json"
                if marker.exists():
                    saved = json.loads(marker.read_text())
                    if saved["input_id_sha256"] != id_digest(rows):
                        raise ValueError("Checkpoint belongs to different input rows")
                    for name, expected in saved["files"].items():
                        if digest(checkpoint / name) != expected:
                            raise ValueError("Corrupt checkpoint file")
                    add_stats(total, saved)
                    resumed += len(rows)
                else:
                    pending.append((number, rows, executor.submit(score, rows)))
        fill()
        publish_progress()
        while pending:
            number, rows, future = pending.popleft()
            add_stats(total, save_chunk(checkpoint, number, rows, future.result()))
            generated += len(rows)
            fill()
            publish_progress()
    if total["entities"] != expected_entities:
        raise ValueError("Inference did not cover the complete test Source 1 file")
    write_json(submission / "progress.json", {**total, "status": "assembling", "ready_for_upload": False})
    for kind, filename, header in (("matching", "matching_results.tsv", "source1_entity_id\tmatched_entity_ids\n"),
        ("candidate", "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids\n")):
        temporary = submission / (filename + ".assembling")
        with temporary.open("wb") as out:
            out.write(header.encode())
            for number in all_chunks:
                with (checkpoint / f"{number:07d}.{kind}.tsv").open("rb") as source:
                    shutil.copyfileobj(source, out)
        temporary.replace(submission / filename)
    matching, candidate = submission / "matching_results.tsv", submission / "candidate_pairs.tsv"
    write_json(submission / "progress.json", {**total, "status": "validating", "ready_for_upload": False})
    strict_errors = validate_submission(matching, candidate, test)
    write_json(submission / "strict_validation.json", {"passed": not strict_errors, "id_checking": True, "issues": strict_errors})
    if strict_errors:
        raise ValueError(f"Strict validation failed: {strict_errors[:5]}")
    official = package / "src/utils/organizer_validate_submission.py"
    with (submission / "official_validation.log").open("w") as log:
        result = subprocess.run([sys.executable, "-S", str(official), "--matching", str(matching),
            "--candidate", str(candidate), "--test-dir", str(test), "--check-ids"], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise ValueError("Official validation failed; see official_validation.log")
    root_output = submission.parent
    output_hashes = {}
    for path in (matching, candidate):
        output_hashes[path.name] = digest(path)
        destination = root_output / path.name
        temporary = destination.with_name(destination.name + ".publishing")
        shutil.copy2(path, temporary)
        if digest(temporary) != output_hashes[path.name]:
            raise ValueError("Published output copy differs")
        temporary.replace(destination)
    summary = {**total, "status": "complete", "ready_for_upload": True,
        "predicted_no_match_rate": total["empty_predictions"] / total["entities"],
        "mean_candidates_per_s1": total["scored_candidates"] / total["entities"],
        "p95_candidates_per_s1": percentile(total["candidate_histogram"], .95, total["entities"]),
        "elapsed_seconds_this_run": time.perf_counter() - start, "workers": args.workers,
        "validation": {"strict": "PASS", "official": "PASS", "id_checking": True},
        "output_sha256": output_hashes, "local_validation_macro_f05": report["holdout"]["macro_f05"],
        "leaderboard_score": None, "threshold": report["threshold"], "model": "larger_alias_v4",
        "candidate_contract": "Exactly the retained candidates scored by the frozen matching model; matches are subsets.",
        "singleton_note": "Empty predictions are predicted no-match entities; true test singletons are unknown.",
        "environment": {"python": sys.version, "platform": platform.platform()}}
    write_json(submission / "submission_report.json", summary)
    write_json(submission / "progress.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, default=Path("output/submission_01"))
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    parser.add_argument("--index-dir", type=Path, default=Path("models"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
