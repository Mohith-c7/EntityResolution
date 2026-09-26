"""
Memory-efficient EDA and Data Profiling script for Amazon Business Entity Resolution 2026.
Runs chunked streaming analysis over ~26 million entity records.
"""

from __future__ import annotations

import gc
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
REPORTS_DIR = PROJECT_ROOT / "reports" / "eda"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 250_000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def profile_source_file(file_path: Path, dataset_name: str) -> dict[str, Any]:
    """Profile a single source TSV file using chunked streaming."""
    logger.info("Profiling %s (%s)...", dataset_name, file_path.name)
    total_rows = 0
    missing_counts = Counter()
    country_counts = Counter()
    id_seen = set()
    dup_id_count = 0

    # Sample rows for length statistics (deterministic reservoir / systematic sampling)
    sample_names = []
    sample_addrs = []
    max_sample_size = 50_000

    columns = []
    dtypes = {}

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=["", "null", "NULL", "None"],
        chunksize=CHUNK_SIZE,
        on_bad_lines="error",
    ):
        if not columns:
            columns = list(chunk.columns)
            dtypes = {col: str(chunk[col].dtype) for col in columns}

        chunk_len = len(chunk)
        total_rows += chunk_len

        # Check missing values
        for col in columns:
            series = chunk[col]
            # null or whitespace only
            null_count = int(series.isna().sum()) + int((series.fillna("").str.strip() == "").sum() - series.isna().sum())
            missing_counts[col] += null_count

        # Check entity_id uniqueness
        if "entity_id" in chunk.columns:
            for eid in chunk["entity_id"]:
                if eid in id_seen:
                    dup_id_count += 1
                else:
                    id_seen.add(eid)

        # Country distribution
        if "country" in chunk.columns:
            clean_country = chunk["country"].fillna("").str.strip()
            for c, cnt in clean_country.value_counts().items():
                if c:
                    country_counts[c] += int(cnt)
                else:
                    country_counts["<MISSING>"] += int(cnt)

        # Reservoir sampling for lengths
        if len(sample_names) < max_sample_size:
            n_to_take = min(chunk_len, max_sample_size - len(sample_names))
            sampled_chunk = chunk.sample(n=n_to_take, random_state=RANDOM_SEED)
            sample_names.extend(sampled_chunk["business_name"].fillna("").tolist())
            sample_addrs.extend(sampled_chunk["business_address"].fillna("").tolist())

    unique_ids = len(id_seen)
    del id_seen
    gc.collect()

    # Compute length distributions from sample
    name_char_lens = [len(s) for s in sample_names]
    name_word_lens = [len(s.split()) for s in sample_names]
    addr_char_lens = [len(s) for s in sample_addrs]
    addr_word_lens = [len(s.split()) for s in sample_addrs]

    missing_pcts = {col: (missing_counts[col] / total_rows * 100.0) if total_rows > 0 else 0.0 for col in columns}

    return {
        "dataset_name": dataset_name,
        "filename": file_path.name,
        "file_size_bytes": file_path.stat().st_size,
        "total_rows": total_rows,
        "columns": columns,
        "dtypes": dtypes,
        "missing_counts": dict(missing_counts),
        "missing_pcts": missing_pcts,
        "unique_ids": unique_ids,
        "duplicate_ids": dup_id_count,
        "country_counts": dict(country_counts.most_common(20)),
        "total_distinct_countries": len([c for c in country_counts if c != "<MISSING>"]),
        "name_char_len_stats": {
            "mean": float(np.mean(name_char_lens)),
            "median": float(np.median(name_char_lens)),
            "min": int(np.min(name_char_lens)),
            "max": int(np.max(name_char_lens)),
            "p95": float(np.percentile(name_char_lens, 95)),
        },
        "name_word_len_stats": {
            "mean": float(np.mean(name_word_lens)),
            "median": float(np.median(name_word_lens)),
            "min": int(np.min(name_word_lens)),
            "max": int(np.max(name_word_lens)),
            "p95": float(np.percentile(name_word_lens, 95)),
        },
        "addr_char_len_stats": {
            "mean": float(np.mean(addr_char_lens)),
            "median": float(np.median(addr_char_lens)),
            "min": int(np.min(addr_char_lens)),
            "max": int(np.max(addr_char_lens)),
            "p95": float(np.percentile(addr_char_lens, 95)),
        },
        "addr_word_len_stats": {
            "mean": float(np.mean(addr_word_lens)),
            "median": float(np.median(addr_word_lens)),
            "min": int(np.min(addr_word_lens)),
            "max": int(np.max(addr_word_lens)),
            "p95": float(np.percentile(addr_word_lens, 95)),
        },
    }


def profile_ground_truth(file_path: Path) -> dict[str, Any]:
    """Analyze train_ground_truth.tsv in streaming chunks."""
    logger.info("Profiling Ground Truth (%s)...", file_path.name)
    total_rows = 0
    id_seen = set()
    dup_id_count = 0
    match_count_dist = Counter()
    s2_positives = 0
    s3_positives = 0
    intra_row_dups = 0

    all_match_counts = []
    max_counts_sample = 200_000

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
        on_bad_lines="error",
    ):
        total_rows += len(chunk)

        s1_series = chunk["source1_entity_id"]
        for s1 in s1_series:
            if s1 in id_seen:
                dup_id_count += 1
            else:
                id_seen.add(s1)

        matched_series = chunk["matched_entity_ids"].fillna("")
        for raw_matches in matched_series:
            raw_s = raw_matches.strip()
            if not raw_s:
                m_list = []
            else:
                m_list = [m.strip() for m in raw_s.split(",") if m.strip()]

            # Intra-row duplicate check
            if len(m_list) != len(set(m_list)):
                intra_row_dups += 1

            cnt = len(m_list)
            match_count_dist[cnt] += 1
            if len(all_match_counts) < max_counts_sample:
                all_match_counts.append(cnt)

            for mid in m_list:
                if mid.startswith("S2-"):
                    s2_positives += 1
                elif mid.startswith("S3-"):
                    s3_positives += 1

    unique_s1 = len(id_seen)
    del id_seen
    gc.collect()

    zero_matches = match_count_dist[0]
    one_match = match_count_dist[1]
    multi_matches = sum(cnt for m_len, cnt in match_count_dist.items() if m_len > 1)
    total_positives = s2_positives + s3_positives

    # Weighted mean matches across full distribution
    total_matches_exact = sum(m_len * cnt for m_len, cnt in match_count_dist.items())
    mean_matches = total_matches_exact / total_rows if total_rows > 0 else 0.0

    return {
        "file_size_bytes": file_path.stat().st_size,
        "total_rows": total_rows,
        "unique_s1_ids": unique_s1,
        "duplicate_s1_ids": dup_id_count,
        "intra_row_duplicates": intra_row_dups,
        "singleton_count": zero_matches,
        "singleton_rate": (zero_matches / total_rows * 100.0) if total_rows > 0 else 0.0,
        "one_match_count": one_match,
        "one_match_rate": (one_match / total_rows * 100.0) if total_rows > 0 else 0.0,
        "multi_match_count": multi_matches,
        "multi_match_rate": (multi_matches / total_rows * 100.0) if total_rows > 0 else 0.0,
        "total_positive_pairs": total_positives,
        "s2_positives": s2_positives,
        "s3_positives": s3_positives,
        "mean_matches_per_s1": mean_matches,
        "median_matches_per_s1": float(np.median(all_match_counts)),
        "max_matches_per_s1": max(match_count_dist.keys()) if match_count_dist else 0,
        "match_count_distribution": {int(k): int(v) for k, v in sorted(match_count_dist.items())},
    }


def find_ground_truth_examples(train_dir: Path, n_examples: int = 15) -> list[dict[str, Any]]:
    """Deterministically find real ground truth match pairs across sources and compare raw text."""
    logger.info("Extracting real ground-truth matching examples...")
    gt_path = train_dir / "train_ground_truth.tsv"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"

    # Read top 50,000 ground truth rows and sample n_examples with multiple matches
    gt_sample = pd.read_csv(gt_path, sep="\t", dtype=str, nrows=50_000)
    # Filter for non-empty matches
    gt_with_matches = gt_sample[gt_sample["matched_entity_ids"].str.len() > 0].copy()
    sampled_gt = gt_with_matches.sample(n=n_examples * 2, random_state=RANDOM_SEED)

    target_s1_ids = set(sampled_gt["source1_entity_id"])
    target_cand_ids = set()
    s1_to_cands: dict[str, list[str]] = {}

    for _, row in sampled_gt.iterrows():
        s1_id = row["source1_entity_id"]
        cands = [c.strip() for c in row["matched_entity_ids"].split(",") if c.strip()]
        s1_to_cands[s1_id] = cands
        target_cand_ids.update(cands)

    # Load matching records from source files
    s1_records = {}
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
        sub = chunk[chunk["entity_id"].isin(target_s1_ids)]
        for _, row in sub.iterrows():
            s1_records[row["entity_id"]] = dict(row)
        if len(s1_records) >= len(target_s1_ids):
            break

    target_s2_ids = {c for c in target_cand_ids if c.startswith("S2-")}
    target_s3_ids = {c for c in target_cand_ids if c.startswith("S3-")}

    cand_records = {}
    if target_s2_ids:
        for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
            sub = chunk[chunk["entity_id"].isin(target_s2_ids)]
            for _, row in sub.iterrows():
                cand_records[row["entity_id"]] = dict(row)
            if len(cand_records.keys() & target_s2_ids) >= len(target_s2_ids):
                break

    if target_s3_ids:
        for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
            sub = chunk[chunk["entity_id"].isin(target_s3_ids)]
            for _, row in sub.iterrows():
                cand_records[row["entity_id"]] = dict(row)
            if len(cand_records.keys() & target_s3_ids) >= len(target_s3_ids):
                break

    examples = []
    for s1_id, cands in s1_to_cands.items():
        if s1_id not in s1_records:
            continue
        s1_data = s1_records[s1_id]
        for cid in cands:
            if cid in cand_records:
                c_data = cand_records[cid]
                examples.append({
                    "s1_id": s1_id,
                    "s1_name": s1_data.get("business_name", ""),
                    "s1_address": s1_data.get("business_address", ""),
                    "s1_country": s1_data.get("country", ""),
                    "cand_id": cid,
                    "cand_source": "S2" if cid.startswith("S2-") else "S3",
                    "cand_name": c_data.get("business_name", ""),
                    "cand_address": c_data.get("business_address", ""),
                    "cand_country": c_data.get("country", ""),
                })
                if len(examples) >= n_examples:
                    return examples

    return examples


def main() -> None:
    logger.info("Starting Comprehensive Raw-Data EDA...")

    results: dict[str, Any] = {
        "datasets": {},
        "ground_truth": {},
        "examples": [],
    }

    # 1. Profile Ground Truth
    gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    results["ground_truth"] = profile_ground_truth(gt_path)

    # 2. Profile Source Datasets
    sources = [
        (TRAIN_DIR / "train_source1.tsv", "Train S1"),
        (TRAIN_DIR / "train_source2.tsv", "Train S2"),
        (TRAIN_DIR / "train_source3.tsv", "Train S3"),
        (TEST_DIR / "test_source1.tsv", "Test S1"),
        (TEST_DIR / "test_source2.tsv", "Test S2"),
        (TEST_DIR / "test_source3.tsv", "Test S3"),
    ]

    for f_path, d_name in sources:
        results["datasets"][d_name] = profile_source_file(f_path, d_name)

    # 3. Ground Truth Matching Examples
    results["examples"] = find_ground_truth_examples(TRAIN_DIR, n_examples=15)

    # Save raw JSON results
    out_json = REPORTS_DIR / "eda_profile_results.json"
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    logger.info("Profiling complete. JSON saved to %s", out_json)


if __name__ == "__main__":
    main()
