"""Batched offline inference; candidate export comes from the scored pairs."""

import csv
import itertools
import json
import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ..blocking.disk_index import DiskSearchConfig, DiskSourceIndex, build_disk_index, normalize_record
from ..blocking.anchor_index import build_anchor_index
from ..features.pairwise_features import feature_vector, build_extended_pair_features
from ..features.registry import FEATURE_VERSION, FEATURE_VERSION_V2, names_for_version
from ..model.name_aliases import NameAliases
from .export import validate_submission
from .training import json_write

_WORKER = None


def initialize_worker(model_path, index_paths, config, threshold, version=FEATURE_VERSION, alias_path=None):
    global _WORKER
    model = lgb.Booster(model_file=str(model_path))
    names = names_for_version(version)
    if model.num_feature() != len(names) or tuple(model.feature_name()) != tuple(names):
        raise ValueError("Model feature count differs from registry")
    aliases = NameAliases.load(alias_path) if config.use_name_aliases else None
    _WORKER = model, [DiskSourceIndex(path, config, aliases=aliases) for path in index_paths], threshold, version


def score_batch(raw_rows):
    model, indexes, threshold, version = _WORKER
    index_by_source = {index.source:index for index in indexes}
    features, groups = [], []
    for raw in raw_rows:
        reference = normalize_record(raw)
        pairs = [pair for index in indexes for pair in index.query(reference)]
        start = len(features)
        if version == FEATURE_VERSION_V2:
            features.extend(np.fromiter(build_extended_pair_features(reference,target,candidate,index_by_source[candidate.candidate_source]).values(),
                dtype=np.float32,count=50) for candidate,target in pairs)
        else:
            features.extend(feature_vector(reference, target, candidate) for candidate, target in pairs)
        groups.append((reference.entity_id, [c.candidate_entity_id for c, _ in pairs], start, len(features)))
    probabilities = model.predict(np.vstack(features), num_threads=1) if features else np.empty(0)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned nonfinite scores")
    result = []
    for eid, ids, start, end in groups:
        predicted = sorted(cid for cid, probability in zip(ids, probabilities[start:end]) if probability >= threshold)
        result.append((eid, ids, predicted))
    return result


def batches(source1: Path, batch_size: int, limit: int = 0):
    with source1.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ("entity_id", "business_name", "business_address", "country"):
            raise ValueError("Unexpected Source 1 columns")
        rows = itertools.islice(reader, limit) if limit else iter(reader)
        while chunk := list(itertools.islice(rows, batch_size)):
            yield chunk


def bounded_results(executor, chunks, workers):
    chunks = iter(chunks)
    pending = deque()
    for _ in range(workers * 2):
        if (chunk := next(chunks, None)) is not None:
            pending.append(executor.submit(score_batch, chunk))
    while pending:
        yield pending.popleft().result()
        if (chunk := next(chunks, None)) is not None:
            pending.append(executor.submit(score_batch, chunk))


def run_inference(test_dir: Path, index_dir: Path, artifact_dir: Path, output_dir: Path, *, workers=4, batch_size=250, reference_limit=0):
    if workers < 1 or batch_size < 1 or reference_limit < 0:
        raise ValueError("Invalid inference batch/worker/limit settings")
    test_dir, index_dir, artifact_dir, output_dir = map(Path, (test_dir, index_dir, artifact_dir, output_dir))
    report = json.loads((artifact_dir / "report.json").read_text())
    version = report["feature_version"]
    names_for_version(version)
    config = DiskSearchConfig(**report["search_config"])
    model_path = artifact_dir / "model.txt"
    alias_path = artifact_dir / "name_aliases.json" if config.use_name_aliases else None
    index_paths = [index_dir / f"index_test_{source}.sqlite" for source in ("S2", "S3")]
    start = time.perf_counter()
    for number, path, source in zip((2, 3), index_paths, ("S2", "S3")):
        build_disk_index(test_dir / f"test_source{number}.tsv", path, source)
        if config.retrieval_mode == "anchored":
            build_anchor_index(path, NameAliases.load(alias_path) if alias_path else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    matching, candidate = output_dir / "matching_results.tsv", output_dir / "candidate_pairs.tsv"
    temporary_m = matching.with_name(matching.name + f".tmp-{os.getpid()}")
    temporary_c = candidate.with_name(candidate.name + f".tmp-{os.getpid()}")
    completed = scored = matched = 0
    chunks = batches(test_dir / "test_source1.tsv", batch_size, reference_limit)
    def write(results, m, c):
        nonlocal completed, scored, matched
        for eid, candidates, predictions in results:
            c.write(eid + "\t" + ",".join(candidates) + "\n")
            m.write(eid + "\t" + ",".join(predictions) + "\n")
            completed += 1
            scored += len(candidates)
            matched += len(predictions)
        print(json.dumps({"stage": "inference", "entities": completed, "scored_pairs": scored,
            "matched_pairs": matched, "seconds": round(time.perf_counter() - start, 1)}), flush=True)
    with temporary_m.open("w", encoding="utf-8", newline="") as m, temporary_c.open("w", encoding="utf-8", newline="") as c:
        m.write("source1_entity_id\tmatched_entity_ids\n")
        c.write("source1_entity_id\tcandidate_entity_ids\n")
        if workers == 1:
            initialize_worker(model_path, index_paths, config, report["threshold"], version, alias_path)
            try:
                for chunk in chunks:
                    write(score_batch(chunk), m, c)
            finally:
                for index in _WORKER[1]:
                    index.close()
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_worker, initargs=(model_path, index_paths, config, report["threshold"],version,alias_path)) as executor:
                for results in bounded_results(executor, chunks, workers):
                    write(results, m, c)
    temporary_m.replace(matching)
    temporary_c.replace(candidate)
    errors = validate_submission(matching, candidate, test_dir) if not reference_limit else ["Limited inference: not a complete submission"]
    metadata = {"entities": completed, "scored_pairs": scored, "matched_pairs": matched, "workers": workers,
        "seconds": time.perf_counter() - start, "training_artifact": str(artifact_dir), "search_config": report["search_config"],
        "threshold": report["threshold"], "ready_for_submission": not errors,
        "validation_errors": errors, "reference_limit": reference_limit,
        "candidate_contract": "Every exported candidate received exactly one matching-model prediction; final matches are subsets."}
    json_write(output_dir / "inference_report.json", metadata)
    if errors and not reference_limit:
        raise ValueError(f"Submission validation failed: {errors[:10]}")
    return metadata
