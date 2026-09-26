"""
Profiling and EDA statistics module for Amazon Business Entity Resolution 2026.
Provides reusable streaming profiling across multi-million record entity sources
and ground-truth match relationships.
"""

from __future__ import annotations

import gc
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 250_000
DEFAULT_RANDOM_SEED = 42


def compute_exact_median_from_histogram(histogram: dict[int | str, int]) -> float:
    """
    Compute the exact median from a frequency distribution histogram.

    Derives the exact median using cumulative observation counts without
    approximations or row-level subsampling. Handles both odd and even
    numbers of total observations.

    Args:
        histogram: Mapping from integer match count to occurrence frequency.

    Returns:
        Exact median as a float.
    """
    clean_hist = {int(k): int(v) for k, v in histogram.items() if int(v) > 0}
    if not clean_hist:
        return 0.0

    sorted_items = sorted(clean_hist.items())
    total_n = sum(cnt for _, cnt in sorted_items)
    if total_n == 0:
        return 0.0

    if total_n % 2 == 1:
        # Odd number of observations: single middle element at 1-indexed (total_n + 1) // 2
        target_pos = (total_n + 1) // 2
        running = 0
        for val, cnt in sorted_items:
            running += cnt
            if running >= target_pos:
                return float(val)
    else:
        # Even number of observations: average of elements at total_n // 2 and (total_n // 2) + 1
        pos1 = total_n // 2
        pos2 = pos1 + 1
        val1 = None
        val2 = None
        running = 0
        for val, cnt in sorted_items:
            running += cnt
            if val1 is None and running >= pos1:
                val1 = val
            if val2 is None and running >= pos2:
                val2 = val
                break
        if val1 is not None and val2 is not None:
            return float(val1 + val2) / 2.0
        return float(val1 if val1 is not None else 0.0)

    return 0.0


def profile_source_dataset(
    file_path: Path | str,
    dataset_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_sample_size: int = 50_000,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """
    Profile a single entity source TSV file using chunked streaming.

    Calculates row counts, missing value rates, ID uniqueness, country frequencies,
    and text length statistics across the entire file without loading the full file into memory.
    Uses streaming reservoir sampling (Algorithm R) with a fixed seed to ensure representative
    sampling across all chunks.
    """
    path = Path(file_path).resolve()
    logger.info("Profiling %s (%s)...", dataset_name, path.name)

    total_rows = 0
    missing_counts = Counter()
    country_counts = Counter()
    id_seen: set[str] = set()
    dup_id_count = 0

    rng = np.random.default_rng(random_seed)
    sample_names: list[str] = []
    sample_addrs: list[str] = []

    columns: list[str] = []
    dtypes: dict[str, str] = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=["", "null", "NULL", "None"],
        chunksize=chunk_size,
        on_bad_lines="error",
    ):
        if not columns:
            columns = list(chunk.columns)
            dtypes = {col: str(chunk[col].dtype) for col in columns}

        chunk_len = len(chunk)
        total_rows += chunk_len

        # Missing values check using boolean union-mask (avoids arithmetic double counting)
        for col in columns:
            series = chunk[col]
            is_missing = series.isna() | (series.astype(str).str.strip() == "")
            missing_counts[col] += int(is_missing.sum())

        # Entity ID uniqueness
        if "entity_id" in chunk.columns:
            for eid in chunk["entity_id"]:
                if eid in id_seen:
                    dup_id_count += 1
                else:
                    id_seen.add(eid)

        # Country distribution (strictly open-set)
        if "country" in chunk.columns:
            clean_country = chunk["country"].fillna("").str.strip()
            for c, cnt in clean_country.value_counts().items():
                if c:
                    country_counts[c] += int(cnt)
                else:
                    country_counts["<MISSING>"] += int(cnt)

        # Streaming reservoir sampling (Algorithm R) across the entire file
        chunk_names = chunk["business_name"].fillna("").tolist() if "business_name" in chunk.columns else [""] * chunk_len
        chunk_addrs = chunk["business_address"].fillna("").tolist() if "business_address" in chunk.columns else [""] * chunk_len

        prev_total = total_rows - chunk_len
        if len(sample_names) < max_sample_size:
            n_take = min(chunk_len, max_sample_size - len(sample_names))
            sample_names.extend(chunk_names[:n_take])
            sample_addrs.extend(chunk_addrs[:n_take])
            rem_start = n_take
        else:
            rem_start = 0

        if rem_start < chunk_len:
            global_indices = np.arange(prev_total + rem_start, total_rows) + 1
            j_arr = rng.integers(0, global_indices)
            selected = np.where(j_arr < max_sample_size)[0]
            for idx in selected:
                target_slot = j_arr[idx]
                item_idx = rem_start + idx
                sample_names[target_slot] = chunk_names[item_idx]
                sample_addrs[target_slot] = chunk_addrs[item_idx]

    unique_ids = len(id_seen)
    del id_seen
    gc.collect()

    name_char_lens = [len(s) for s in sample_names]
    name_word_lens = [len(s.split()) for s in sample_names]
    addr_char_lens = [len(s) for s in sample_addrs]
    addr_word_lens = [len(s.split()) for s in sample_addrs]

    missing_pcts = {
        col: (missing_counts[col] / total_rows * 100.0) if total_rows > 0 else 0.0
        for col in columns
    }

    return {
        "dataset_name": dataset_name,
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "total_rows": total_rows,
        "columns": columns,
        "dtypes": dtypes,
        "missing_counts": dict(missing_counts),
        "missing_pcts": missing_pcts,
        "unique_ids": unique_ids,
        "duplicate_ids": dup_id_count,
        "country_counts": dict(country_counts.most_common(20)),
        "total_distinct_countries": len([c for c in country_counts if c != "<MISSING>"]),
        "sampling_metadata": {
            "method": "streaming_reservoir_algorithm_r",
            "sample_size": len(sample_names),
            "random_seed": random_seed,
            "purpose": "text_length_and_word_distribution_statistics",
        },
        "name_char_len_stats": {
            "mean": float(np.mean(name_char_lens)) if name_char_lens else 0.0,
            "median": float(np.median(name_char_lens)) if name_char_lens else 0.0,
            "min": int(np.min(name_char_lens)) if name_char_lens else 0,
            "max": int(np.max(name_char_lens)) if name_char_lens else 0,
            "p95": float(np.percentile(name_char_lens, 95)) if name_char_lens else 0.0,
        },
        "name_word_len_stats": {
            "mean": float(np.mean(name_word_lens)) if name_word_lens else 0.0,
            "median": float(np.median(name_word_lens)) if name_word_lens else 0.0,
            "min": int(np.min(name_word_lens)) if name_word_lens else 0,
            "max": int(np.max(name_word_lens)) if name_word_lens else 0,
            "p95": float(np.percentile(name_word_lens, 95)) if name_word_lens else 0.0,
        },
        "addr_char_len_stats": {
            "mean": float(np.mean(addr_char_lens)) if addr_char_lens else 0.0,
            "median": float(np.median(addr_char_lens)) if addr_char_lens else 0.0,
            "min": int(np.min(addr_char_lens)) if addr_char_lens else 0,
            "max": int(np.max(addr_char_lens)) if addr_char_lens else 0,
            "p95": float(np.percentile(addr_char_lens, 95)) if addr_char_lens else 0.0,
        },
        "addr_word_len_stats": {
            "mean": float(np.mean(addr_word_lens)) if addr_word_lens else 0.0,
            "median": float(np.median(addr_word_lens)) if addr_word_lens else 0.0,
            "min": int(np.min(addr_word_lens)) if addr_word_lens else 0,
            "max": int(np.max(addr_word_lens)) if addr_word_lens else 0,
            "p95": float(np.percentile(addr_word_lens, 95)) if addr_word_lens else 0.0,
        },
    }


def profile_ground_truth(
    file_path: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """
    Analyze ground truth relationships in streaming chunks.
    Computes singleton rate, multi-match distribution, positive pair counts, and checks intra-row duplicates.
    Derives the exact median match count from the complete histogram.
    """
    path = Path(file_path).resolve()
    logger.info("Profiling Ground Truth (%s)...", path.name)

    total_rows = 0
    id_seen: set[str] = set()
    dup_id_count = 0
    match_count_dist = Counter()
    s2_positives = 0
    s3_positives = 0
    intra_row_dups = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
        on_bad_lines="error",
    ):
        total_rows += len(chunk)

        for s1 in chunk["source1_entity_id"]:
            if s1 in id_seen:
                dup_id_count += 1
            else:
                id_seen.add(s1)

        for raw_matches in chunk["matched_entity_ids"].fillna(""):
            raw_s = raw_matches.strip()
            if not raw_s:
                m_list = []
            else:
                m_list = [m.strip() for m in raw_s.split(",") if m.strip()]

            if len(m_list) != len(set(m_list)):
                intra_row_dups += 1

            cnt = len(m_list)
            match_count_dist[cnt] += 1

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

    total_matches_exact = sum(m_len * cnt for m_len, cnt in match_count_dist.items())
    mean_matches = total_matches_exact / total_rows if total_rows > 0 else 0.0
    exact_median = compute_exact_median_from_histogram(match_count_dist)

    return {
        "file_size_bytes": path.stat().st_size,
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
        "median_matches_per_s1": exact_median,
        "median_calculation_method": "exact_from_full_histogram",
        "max_matches_per_s1": max(match_count_dist.keys()) if match_count_dist else 0,
        "match_count_distribution": {int(k): int(v) for k, v in sorted(match_count_dist.items())},
    }


def find_ground_truth_examples(
    train_dir: Path | str,
    n_examples: int = 15,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """
    Deterministically find real ground truth match pairs across sources and compare raw text.
    """
    t_dir = Path(train_dir).resolve()
    logger.info("Extracting real ground-truth matching examples from %s...", t_dir)
    gt_path = t_dir / "train_ground_truth.tsv"
    s1_path = t_dir / "train_source1.tsv"
    s2_path = t_dir / "train_source2.tsv"
    s3_path = t_dir / "train_source3.tsv"

    gt_sample = pd.read_csv(gt_path, sep="\t", dtype=str, nrows=50_000)
    gt_with_matches = gt_sample[gt_sample["matched_entity_ids"].str.len() > 0].copy()
    sampled_gt = gt_with_matches.sample(n=min(len(gt_with_matches), n_examples * 2), random_state=random_seed)

    target_s1_ids = set(sampled_gt["source1_entity_id"])
    target_cand_ids: set[str] = set()
    s1_to_cands: dict[str, list[str]] = {}

    for _, row in sampled_gt.iterrows():
        s1_id = row["source1_entity_id"]
        cands = [c.strip() for c in row["matched_entity_ids"].split(",") if c.strip()]
        s1_to_cands[s1_id] = cands
        target_cand_ids.update(cands)

    s1_records: dict[str, dict[str, Any]] = {}
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=chunk_size):
        sub = chunk[chunk["entity_id"].isin(target_s1_ids)]
        for _, row in sub.iterrows():
            s1_records[row["entity_id"]] = dict(row)
        if len(s1_records) >= len(target_s1_ids):
            break

    target_s2_ids = {c for c in target_cand_ids if c.startswith("S2-")}
    target_s3_ids = {c for c in target_cand_ids if c.startswith("S3-")}

    cand_records: dict[str, dict[str, Any]] = {}
    if target_s2_ids:
        for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=chunk_size):
            sub = chunk[chunk["entity_id"].isin(target_s2_ids)]
            for _, row in sub.iterrows():
                cand_records[row["entity_id"]] = dict(row)
            if len(cand_records.keys() & target_s2_ids) >= len(target_s2_ids):
                break

    if target_s3_ids:
        for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=chunk_size):
            sub = chunk[chunk["entity_id"].isin(target_s3_ids)]
            for _, row in sub.iterrows():
                cand_records[row["entity_id"]] = dict(row)
            if len(cand_records.keys() & target_s3_ids) >= len(target_s3_ids):
                break

    examples: list[dict[str, Any]] = []
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


def print_profiling_summary(results: dict[str, Any]) -> str:
    """Format and return a printable human-readable summary of profiling results."""
    lines = [
        "=" * 60,
        "STAGE 0 EDA & DATA PROFILING SUMMARY",
        "=" * 60,
    ]

    gt = results.get("ground_truth", {})
    if gt:
        lines.extend([
            f"Ground Truth Total S1 Entities: {gt.get('total_rows', 0):,}",
            f"Singletons (0 matches):         {gt.get('singleton_count', 0):,} ({gt.get('singleton_rate', 0.0):.2f}%)",
            f"1-Match Entities:               {gt.get('one_match_count', 0):,} ({gt.get('one_match_rate', 0.0):.2f}%)",
            f"Multi-Match (>1) Entities:      {gt.get('multi_match_count', 0):,} ({gt.get('multi_match_rate', 0.0):.2f}%)",
            f"Total Positive Pairs:           {gt.get('total_positive_pairs', 0):,} (S2: {gt.get('s2_positives', 0):,}, S3: {gt.get('s3_positives', 0):,})",
            f"Mean Matches / S1:              {gt.get('mean_matches_per_s1', 0.0):.3f}",
            f"Max Matches / S1:               {gt.get('max_matches_per_s1', 0)}",
            "-" * 60,
        ])

    datasets = results.get("datasets", {})
    for name, d in datasets.items():
        lines.extend([
            f"{name} ({d.get('filename', '')}): {d.get('total_rows', 0):,} rows",
            f"  Missing: {d.get('missing_pcts', {})}",
            f"  Countries: {d.get('country_counts', {})}",
        ])

    lines.append("=" * 60)
    summary_text = "\n".join(lines)
    print(summary_text)
    return summary_text
