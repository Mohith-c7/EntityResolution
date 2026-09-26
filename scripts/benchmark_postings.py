"""Benchmark experimental postings retrieval on tuning data, never audit data."""
import argparse
import csv
import hashlib
import json
import multiprocessing
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.disk_index import DiskSearchConfig, normalize_record
from src.blocking.postings_index import PostingsConfig, PostingsSourceIndex
from src.evaluation.metrics import score_matches
from src.features.pairwise_features import build_versioned_pair_features
from src.features.registry import names_for_version
from src.model.name_aliases import NameAliases

STATE = None
CACHE = None
BRIDGE_OPTIONS = None


def initialize(artifact, index_dir, options, top_k, native_extension, split="train", cache_dir=None, ranking_path=None, bridge_options=None):
    global STATE, CACHE, BRIDGE_OPTIONS
    artifact = Path(artifact)
    report = json.loads((artifact / "report.json").read_text())
    aliases = NameAliases.load(artifact / "name_aliases.json")
    config = replace(DiskSearchConfig(**report["search_config"]), top_k=top_k)
    ranker = lgb.Booster(model_file=str(ranking_path)) if ranking_path else None
    if ranker:
        from src.blocking.cheap_ranker import RANK_FEATURES
        if ranker.feature_name() != list(RANK_FEATURES): raise ValueError("Ranker feature schema mismatch")
    indexes = [PostingsSourceIndex(Path(index_dir) / f"index_{split}_{s}.sqlite", config,
        aliases=aliases, postings_config=PostingsConfig(**options), native_extension=native_extension,
        ranking_model=ranker) for s in ("S2", "S3")]
    model = lgb.Booster(model_file=str(artifact / "model.txt"))
    assert model.feature_name() == list(names_for_version(report["feature_version"]))
    STATE = report, indexes, model
    CACHE = cache_dir
    BRIDGE_OPTIONS = bridge_options


def score(rows):
    report, indexes, model = STATE
    matrix, groups, provenance, profile = [], [], [], {"features_seconds": 0., "prediction_seconds": 0., "cache_write_seconds": 0.}
    before = [{k: v for k, v in index.profile.items()} for index in indexes]
    for row in rows:
        reference = normalize_record(row)
        bridged = None
        if BRIDGE_OPTIONS:
            from src.blocking.bridge import retrieve_with_bridges
            bridged = retrieve_with_bridges(reference, indexes, indexes[0].ranking_model, **BRIDGE_OPTIONS)
        ids = []
        start = len(matrix)
        for index in indexes:
            pairs = bridged[index.source] if bridged is not None else index.query(reference)
            stamp = time.perf_counter()
            for candidate, target in pairs:
                features = build_versioned_pair_features(reference, target, candidate, index, report["feature_version"])
                matrix.append(list(features.values()))
                ids.append(candidate.candidate_entity_id)
                if CACHE:
                    provenance.append((candidate.source1_entity_id, candidate.candidate_entity_id,
                        candidate.candidate_source, candidate.rank_within_source, ",".join(candidate.blocking_paths)))
            profile["features_seconds"] += time.perf_counter() - stamp
        groups.append((reference.entity_id, ids, start, len(matrix)))
    stamp = time.perf_counter()
    probabilities = model.predict(np.asarray(matrix, dtype=np.float32), num_threads=1) if matrix else []
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite model prediction")
    profile["prediction_seconds"] = time.perf_counter() - stamp
    if CACHE:
        import pandas as pd
        stamp = time.perf_counter()
        frame = pd.DataFrame(np.asarray(matrix, dtype=np.float32), columns=names_for_version(report["feature_version"]))
        for i, key in enumerate(("source1_entity_id", "candidate_entity_id", "candidate_source", "rank_within_source", "blocking_paths")):
            frame[key] = [p[i] for p in provenance]
        frame["probability"] = probabilities
        chunk_id = hashlib.sha256("\n".join(r["entity_id"] for r in rows).encode()).hexdigest()
        path = Path(CACHE) / (chunk_id + ".parquet")
        temporary = path.with_suffix(".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        profile["cache_write_seconds"] = time.perf_counter() - stamp
    for index, old in zip(indexes, before):
        for key, value in index.profile.items():
            profile[key] = profile.get(key, 0) + value - old[key]
    return [(eid, ids, [cid for cid, probability in zip(ids, probabilities[a:b])
                        if probability >= report["threshold"]]) for eid, ids, a, b in groups], profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-postings", type=int, default=20000)
    parser.add_argument("--fused-budget", type=int, default=64)
    parser.add_argument("--field-budget", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--save-pairs", type=Path)
    parser.add_argument("--native-extension", type=Path)
    parser.add_argument("--test", action="store_true", help="Unlabelled test throughput/country diagnostics only")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--ranking-model", type=Path)
    parser.add_argument("--baseline-predictions", type=Path, help="Optional saved scores for a same-reference baseline")
    parser.add_argument("--bridge", action="store_true")
    parser.add_argument("--bridge-min-probability", type=float, default=.99)
    parser.add_argument("--informative-bridges-only", action="store_true")
    parser.add_argument("--timing-note", default="Includes worker startup, retrieval, features, scoring and optional feature-cache writes; excludes full output assembly and validators.")
    parser.add_argument("--artifact", type=Path, default=Path("models/features_v3_audit01"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.entities < 1 or args.workers < 1:
        parser.error("Entity and worker counts must be positive")
    if args.baseline_predictions and not args.baseline_predictions.is_file():
        parser.error("Explicit baseline predictions file does not exist")
    if args.bridge and not args.ranking_model: parser.error("Bridge retrieval requires a blocking ranker")
    if not 0 <= args.bridge_min_probability <= 1: parser.error("Invalid bridge seed probability")
    if args.informative_bridges_only and not args.bridge: parser.error("Informative seed gating requires --bridge")
    bridge_options = {"min_probability": args.bridge_min_probability, "max_seeds": 2,
                      "informative_only": args.informative_bridges_only} if args.bridge else None
    if args.test:
        refs = []
        rng = random.Random(42)
        with Path("dataset/test/test_source1.tsv").open(encoding="utf-8-sig", newline="") as stream:
            for number, row in enumerate(csv.DictReader(stream, delimiter="\t")):
                if number < args.entities:
                    refs.append(row)
                else:
                    position = rng.randrange(number + 1)
                    if position < args.entities:
                        refs[position] = row
        truth = None
    else:
        refs = json.loads((args.artifact / "sampled_references.json").read_text())["tune"]
        refs = sorted(refs, key=lambda r: hashlib.sha256(r["entity_id"].encode()).digest())[:args.entities]
        all_truth = json.loads((args.artifact / "sampled_truth.json").read_text())
        truth = {r["entity_id"]: set(all_truth[r["entity_id"]]) for r in refs}
    options = asdict(PostingsConfig(max_postings=args.max_postings,
        fused_budget=args.fused_budget, field_budget=args.field_budget))
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=False)
        (args.cache_dir / "manifest.json").write_text(json.dumps({"complete": False,
            "model_sha256": hashlib.sha256((args.artifact / "model.txt").read_bytes()).hexdigest(),
            "alias_sha256": hashlib.sha256((args.artifact / "name_aliases.json").read_bytes()).hexdigest(),
            "ranking_model_sha256": hashlib.sha256(args.ranking_model.read_bytes()).hexdigest() if args.ranking_model else None,
            "bridge_options": bridge_options,
            "config": options, "top_k": args.top_k, "test": args.test,
            "reference_ids": [r["entity_id"] for r in refs]}, indent=2))
    started = time.perf_counter()
    chunks = [refs[i:i+25] for i in range(0, len(refs), 25)]
    if args.workers == 1:
        initialize(args.artifact, Path("models"), options, args.top_k, args.native_extension, "test" if args.test else "train", args.cache_dir, args.ranking_model, bridge_options)
        batches = []
        for chunk in chunks:
            batches.append(score(chunk))
            print(json.dumps({"completed": sum(len(b[0]) for b in batches), "seconds": time.perf_counter()-started}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize, initargs=(args.artifact, Path("models"), options, args.top_k, args.native_extension,
                                            "test" if args.test else "train", args.cache_dir, args.ranking_model, bridge_options)) as pool:
            batches = []
            for batch in pool.map(score, chunks):
                batches.append(batch)
                if len(batches) % 40 == 0:
                    print(json.dumps({"completed": sum(len(b[0]) for b in batches),
                        "seconds": time.perf_counter() - started}), flush=True)
    seconds = time.perf_counter() - started
    rows = [row for batch, profile in batches for row in batch]
    candidates = {e: set(c) for e, c, _ in rows}
    predictions = {e: p for e, _, p in rows}
    profile = {k: sum(p[k] for _, p in batches) for k in batches[0][1]}
    result = {"status": "experimental_not_submission_ready", "split": "unlabelled_test_sample" if args.test else "previously_inspected_tuning",
        "entities": len(rows), "workers": args.workers, "seconds": seconds,
        "references_per_second": len(rows)/seconds, "options": options, "top_k_per_source": args.top_k,
        "native_extension": str(args.native_extension) if args.native_extension else None,
        "ranking_model": str(args.ranking_model) if args.ranking_model else None,
        "bridge_options": bridge_options,
        "timing_note": args.timing_note,
        "mean_candidates": sum(map(len, candidates.values()))/len(rows),
        "profile_worker_seconds_summed": profile,
        "reference_ids_sha256": hashlib.sha256("\n".join(r["entity_id"] for r in refs).encode()).hexdigest(),
        "caveat": "New retrieval changes blocking score distribution; old classifier/threshold are exploratory controls."}
    by_country = {}
    for ref in refs:
        country = ref["country"]
        entry = by_country.setdefault(country, {"references": 0, "candidates": 0, "matches": 0, "empty": 0, "eight_or_more": 0})
        entry["references"] += 1
        entry["candidates"] += len(candidates[ref["entity_id"]])
        count = len(predictions[ref["entity_id"]])
        entry["matches"] += count
        entry["empty"] += count == 0
        entry["eight_or_more"] += count >= 8
    result["country_counts_unlabelled_when_test"] = by_country
    import pandas as pd
    if truth is not None:
        result["matching"] = score_matches(truth, predictions)
        result["oracle"] = score_matches(truth, {e: truth[e] & ids for e, ids in candidates.items()})
    baseline_path = args.baseline_predictions or args.artifact / "predictions_tune.parquet"
    if truth is not None and baseline_path.exists():
        previous = pd.read_parquet(baseline_path)
        selected = previous[previous.source1_entity_id.isin(truth)]
        baseline = {e: [] for e in truth}
        baseline_candidates = {e: set() for e in truth}
        threshold = json.loads((args.artifact / "report.json").read_text())["threshold"]
        for row in selected.itertuples():
            baseline_candidates[row.source1_entity_id].add(row.candidate_entity_id)
            if row.probability >= threshold:
                baseline[row.source1_entity_id].append(row.candidate_entity_id)
        result["same_entities_saved_baseline_matching"] = score_matches(truth, baseline)
        result["same_entities_saved_baseline_oracle"] = score_matches(truth, {
            e: truth[e] & ids for e, ids in baseline_candidates.items()})
    if args.save_pairs:
        args.save_pairs.parent.mkdir(parents=True, exist_ok=True)
        args.save_pairs.write_text(json.dumps(rows))
    if args.cache_dir:
        path = args.cache_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["complete"] = True
        manifest["rows_scored"] = sum(map(len, candidates.values()))
        manifest["files"] = [p.name for p in sorted(args.cache_dir.glob("*.parquet"))]
        path.write_text(json.dumps(manifest, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
