"""Two-pass, sharded blocking with disk-backed top-K merge and exact exports."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.preprocessing.preprocess import preprocess_dataframe

from .scalable_generator import SourceIndex
from .contracts import CANDIDATE_COLUMNS, PATHS, BlockingConfig, BlockingRecord, Candidate, rank_candidates, records_from_dataframe
from .token_index import CorpusStatistics

try:
    import resource
except ImportError:  # Windows does not expose getrusage.
    resource = None


logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = ("entity_id", "business_name", "business_address", "country")


def iter_records(path: Path, source: str, chunk_size: int, limit: int | None = None):
    columns = pd.read_csv(path, sep="\t", nrows=0).columns
    missing = set(REQUIRED_COLUMNS) - set(columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}; TSV separator must be a tab")
    with pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[""],
        chunksize=chunk_size, nrows=limit, encoding="utf-8", on_bad_lines="error",
    ) as chunks:
        for chunk in chunks:
            yield records_from_dataframe(preprocess_dataframe(chunk), source)


def load_truth(connection: sqlite3.Connection, path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("Ground truth must have source1_entity_id and matched_entity_ids TSV columns")
        for row in reader:
            sid = row["source1_entity_id"]
            if not sid or not sid.startswith("S1-") or sid != sid.strip():
                raise ValueError(f"Invalid ground-truth reference ID {sid!r}")
            # A reference limit is a development convenience. Extra truth rows
            # are ignored, but truth for every evaluated reference is required.
            if not connection.execute("SELECT 1 FROM refs WHERE id=?", (sid,)).fetchone():
                continue
            raw = row["matched_entity_ids"] or ""
            values = [value.strip() for value in raw.split(",") if value.strip()]
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate ground-truth targets for {sid}")
            if any(not value.startswith(("S2-", "S3-")) for value in values):
                raise ValueError(f"Invalid ground-truth target for {sid}")
            connection.execute("INSERT INTO truth_entities VALUES (?, ?)", (sid, len(values)))
            connection.executemany("INSERT INTO truth VALUES (?, ?)", ((sid, value) for value in values))
    if connection.execute("SELECT COUNT(*) FROM refs WHERE id NOT IN (SELECT id FROM truth_entities)").fetchone()[0]:
        raise ValueError("Ground truth does not cover all evaluated Source 1 entities")
    connection.commit()


def create_store(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE refs (position INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, country TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE target_ids (id TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE truth_entities (id TEXT PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE truth (s1 TEXT, target TEXT, PRIMARY KEY(s1, target)) WITHOUT ROWID;
        CREATE TABLE hits (s1 TEXT, target TEXT, paths TEXT NOT NULL, PRIMARY KEY(s1, target)) WITHOUT ROWID;
        CREATE TABLE pairs (s1 TEXT, target TEXT, source TEXT, score REAL, paths TEXT, PRIMARY KEY(s1, target)) WITHOUT ROWID;
        CREATE INDEX pairs_by_reference_source ON pairs(s1, source);
    """)
    return connection


def merge_query(connection: sqlite3.Connection, result, config: BlockingConfig, evaluate: bool) -> None:
    reference = result.reference
    source = result.candidates[0].candidate_source if result.candidates else None
    if source:
        old = [
            Candidate(reference.entity_id, row[0], row[1], row[2], tuple(json.loads(row[3])))
            for row in connection.execute(
                "SELECT target, source, score, paths FROM pairs WHERE s1=? AND source=?",
                (reference.entity_id, source),
            )
        ]
        # Each target ID occurs in one shard only. Keep top-K across all shards.
        selected = rank_candidates(old + list(result.candidates), config.top_k)
        connection.execute("DELETE FROM pairs WHERE s1=? AND source=?", (reference.entity_id, source))
        connection.executemany(
            "INSERT INTO pairs VALUES (?, ?, ?, ?, ?)",
            ((c.source1_entity_id, c.candidate_entity_id, c.candidate_source, c.blocking_score, json.dumps(c.blocking_paths)) for c in selected),
        )
    if evaluate:
        truth = {row[0] for row in connection.execute("SELECT target FROM truth WHERE s1=?", (reference.entity_id,))}
        connection.executemany(
            "INSERT INTO hits VALUES (?, ?, ?)",
            ((reference.entity_id, c.candidate_entity_id, json.dumps(c.blocking_paths)) for c in result.before_top_k if c.candidate_entity_id in truth),
        )


def selected_pairs(connection: sqlite3.Connection, sid: str) -> list[Candidate]:
    rows = connection.execute(
        "SELECT target, source, score, paths FROM pairs WHERE s1=? ORDER BY source, score DESC, target",
        (sid,),
    )
    ranks = Counter()
    candidates = []
    for target, source, score, paths in rows:
        ranks[source] += 1
        candidates.append(Candidate(sid, target, source, score, tuple(json.loads(paths)), ranks[source]))
    return candidates


def export_candidates(connection: sqlite3.Connection, output_dir: Path) -> None:
    list_path = output_dir / "candidate_pairs.tsv"
    score_path = output_dir / "candidate_scores.tsv"
    list_tmp = list_path.with_suffix(".tsv.tmp")
    score_tmp = score_path.with_suffix(".tsv.tmp")
    try:
        with list_tmp.open("w", encoding="utf-8", newline="") as lists, score_tmp.open("w", encoding="utf-8", newline="") as scores:
            list_writer = csv.writer(lists, delimiter="\t", lineterminator="\n")
            score_writer = csv.writer(scores, delimiter="\t", lineterminator="\n")
            list_writer.writerow(("source1_entity_id", "candidate_entity_ids"))
            score_writer.writerow(CANDIDATE_COLUMNS)
            for sid, in connection.execute("SELECT id FROM refs ORDER BY position"):
                candidates = selected_pairs(connection, sid)
                list_writer.writerow((sid, ",".join(sorted(c.candidate_entity_id for c in candidates))))
                for c in candidates:
                    score_writer.writerow((sid, c.candidate_entity_id, c.candidate_source, repr(c.blocking_score), c.rank_within_source, ",".join(c.blocking_paths)))
        list_tmp.replace(list_path)
        score_tmp.replace(score_path)
    finally:
        list_tmp.unlink(missing_ok=True)
        score_tmp.unlink(missing_ok=True)


def summarize(connection: sqlite3.Connection, target_count: int, evaluate: bool) -> dict:
    histogram = Counter()
    country_totals = defaultdict(Counter)
    paths = Counter()
    unique_path_hits = Counter()
    source_truth = Counter()
    source_hits = Counter()
    total_truth = total_raw_hits = total_final_hits = nonsingletons = any_hits = complete_hits = 0
    oracle_sum = 0.0
    reference_count = connection.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
    for sid, country in connection.execute("SELECT id, country FROM refs ORDER BY position"):
        candidates = selected_pairs(connection, sid)
        ids = {c.candidate_entity_id for c in candidates}
        histogram[len(ids)] += 1
        country_totals[country]["reference_count"] += 1
        country_totals[country]["candidate_count"] += len(ids)
        if not evaluate:
            continue
        truth = {row[0] for row in connection.execute("SELECT target FROM truth WHERE s1=?", (sid,))}
        hit_rows = list(connection.execute("SELECT target, paths FROM hits WHERE s1=?", (sid,)))
        hits = ids & truth
        total_truth += len(truth)
        total_raw_hits += len(hit_rows)
        total_final_hits += len(hits)
        country_totals[country]["true_links"] += len(truth)
        country_totals[country]["retrieved_links"] += len(hits)
        for target in truth:
            source_truth[target[:2]] += 1
        for target in hits:
            source_hits[target[:2]] += 1
        for _, encoded in hit_rows:
            hit_paths = json.loads(encoded)
            paths.update(hit_paths)
            if len(hit_paths) == 1:
                unique_path_hits[hit_paths[0]] += 1
        if truth:
            nonsingletons += 1
            any_hits += bool(hits)
            complete_hits += hits == truth
            oracle_sum += 1.25 * len(hits) / (1.25 * len(hits) + 0.25 * (len(truth) - len(hits)))
        else:
            oracle_sum += 1.0
    total_candidates = sum(count * frequency for count, frequency in histogram.items())
    rank = max(1, math.ceil(0.95 * reference_count))
    cumulative = 0
    p95 = 0
    for count, frequency in sorted(histogram.items()):
        cumulative += frequency
        if cumulative >= rank:
            p95 = count
            break
    report = {
        "reference_count": reference_count,
        "target_count": target_count,
        "candidate_count": total_candidates,
        "mean_candidates_per_reference": total_candidates / reference_count if reference_count else 0.0,
        "p95_candidates_per_reference": p95,
        "max_candidates_per_reference": max(histogram, default=0),
        "references_without_candidates": histogram[0],
        "reduction_ratio": 1 - total_candidates / (reference_count * target_count) if reference_count and target_count else None,
        "country_breakdown": {country: dict(values) for country, values in sorted(country_totals.items())},
    }
    if evaluate:
        report.update({
            "true_link_count": total_truth,
            "retrieved_true_links_before_top_k": total_raw_hits,
            "retrieved_true_links_after_top_k": total_final_hits,
            "link_recall_before_top_k": total_raw_hits / total_truth if total_truth else None,
            "link_recall_after_top_k": total_final_hits / total_truth if total_truth else None,
            "nonsingleton_any_match_coverage": any_hits / nonsingletons if nonsingletons else None,
            "nonsingleton_complete_match_coverage": complete_hits / nonsingletons if nonsingletons else None,
            "oracle_macro_f05": oracle_sum / reference_count if reference_count else None,
            "path_true_link_hits": {path: paths[path] for path in PATHS},
            "path_unique_true_link_hits": {path: unique_path_hits[path] for path in PATHS},
            "source_breakdown": {
                source: {"true_links": source_truth[source], "retrieved_links": source_hits[source],
                         "link_recall": source_hits[source] / source_truth[source] if source_truth[source] else None}
                for source in ("S2", "S3")
            },
        })
    return report


def run_blocking(source1: Path, source2: Path, source3: Path, output_dir: Path,
                 config: BlockingConfig | None = None, shard_size: int = 100_000,
                 query_batch_size: int = 1000, ground_truth: Path | None = None,
                 reference_limit: int | None = None, target_limit: int | None = None) -> dict:
    config = config or BlockingConfig()
    if shard_size < 1 or query_batch_size < 1:
        raise ValueError("Shard and query batch sizes must be positive")
    for value in (reference_limit, target_limit):
        if value is not None and value < 1:
            raise ValueError("Development record limits must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source_reports = {}
    with tempfile.TemporaryDirectory(prefix="blocking-", dir=output_dir) as work_dir:
        connection = create_store(Path(work_dir) / "candidates.sqlite")
        try:
            position = 0
            for batch in iter_records(source1, "S1", query_batch_size, reference_limit):
                rows = []
                for record in batch:
                    rows.append((position, record.entity_id, record.country, json.dumps(record.to_dict(), ensure_ascii=False)))
                    position += 1
                connection.executemany("INSERT INTO refs VALUES (?, ?, ?, ?)", rows)
            connection.commit()
            if ground_truth is not None:
                load_truth(connection, ground_truth)
            total_targets = 0
            for source, path in (("S2", source2), ("S3", source3)):
                source_start = time.perf_counter()
                statistics = CorpusStatistics(config)
                for batch in iter_records(path, source, shard_size, target_limit):
                    connection.executemany("INSERT INTO target_ids VALUES (?)", ((record.entity_id,) for record in batch))
                    statistics.update(batch)
                    connection.commit()
                total_targets += statistics.record_count
                logger.info("%s corpus: %s records; building shard indexes", source, statistics.record_count)
                shard_count = 0
                raw_candidate_count = 0
                for batch in iter_records(path, source, shard_size, target_limit):
                    shard_count += 1
                    index = SourceIndex(batch, source, config, statistics)
                    cursor = connection.execute("SELECT payload FROM refs ORDER BY position")
                    while rows := cursor.fetchmany(query_batch_size):
                        for (payload,) in rows:
                            result = index.query(BlockingRecord.from_dict(json.loads(payload)))
                            raw_candidate_count += len(result.before_top_k)
                            merge_query(connection, result, config, ground_truth is not None)
                        connection.commit()
                    del index
                    logger.info("%s shard %s complete", source, shard_count)
                source_reports[source] = {
                    "record_count": statistics.record_count, "shard_count": shard_count,
                    "raw_union_candidate_count": raw_candidate_count,
                    "seconds": time.perf_counter() - source_start,
                    "vocabulary_sizes": {kind: len(values) for kind, values in statistics.document_frequency.items()},
                }
                del statistics
            report = summarize(connection, total_targets, ground_truth is not None)
            export_candidates(connection, output_dir)
        finally:
            connection.close()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None
    report.update({
        "config": config.to_dict(), "shard_size": shard_size, "query_batch_size": query_batch_size,
        "reference_limit": reference_limit, "target_limit": target_limit,
        "seconds": time.perf_counter() - start,
        "peak_process_rss_mib": rss / (1024 ** 2 if sys.platform == "darwin" else 1024) if rss is not None else None,
        "source_processing": source_reports,
        "matching_model_scored": False,
        "candidate_export_status": "ready_for_model_inference; not a completed challenge submission",
        "retrieval_statistics_policy": "label-free frequencies from each supplied target corpus, shared across its shards",
    })
    (output_dir / "blocking_diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", required=True, type=Path)
    parser.add_argument("--source2", required=True, type=Path)
    parser.add_argument("--source3", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--config", type=Path, help="JSON object with BlockingConfig fields")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--query-batch-size", type=int, default=1000)
    parser.add_argument("--reference-limit", type=int, help="Development only; restrict S1 rows")
    parser.add_argument("--target-limit", type=int, help="Development only; recall uses a reduced distractor pool")
    args = parser.parse_args(argv)
    config = BlockingConfig(**json.loads(args.config.read_text())) if args.config else BlockingConfig()
    if args.top_k is not None:
        config = replace(config, top_k=args.top_k)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = run_blocking(args.source1, args.source2, args.source3, args.output_dir, config,
                              args.shard_size, args.query_batch_size, args.ground_truth,
                              args.reference_limit, args.target_limit)
    except (ValueError, OSError, sqlite3.IntegrityError) as error:
        parser.error(str(error))
    print(json.dumps({key: report[key] for key in (
        "reference_count", "target_count", "candidate_count", "mean_candidates_per_reference", "seconds",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
