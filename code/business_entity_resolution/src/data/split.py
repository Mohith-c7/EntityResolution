"""
Shared Grouped Validation Split Protocol.

Implements entity-disjoint connected-components grouping, stratification by country,
singleton status, and match cardinality, distractor target partitioning, and
zero-leakage verification per ARCHITECTURE.md Section 12.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UnionFind:
    """Disjoint Set Union (DSU) with path compression for connected components."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self.parent.setdefault(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def compute_connected_components(gt_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute connected components from bipartite edges between S1 and matched targets.

    Entities that share targets (e.g. S1-A -> S2-X and S1-B -> S2-X) are grouped into
    the same connected component so they are never split across train/validation folds.

    Parameters
    ----------
    gt_df : pd.DataFrame
        Ground truth dataframe containing 'source1_entity_id' and 'matched_entity_ids'.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of root_id -> {
            "s1_ids": list of S1 entity IDs in cluster,
            "target_ids": set of matched S2/S3 entity IDs in cluster
        }
    """
    uf = UnionFind()
    s1_to_targets: Dict[str, List[str]] = {}

    s1_vals = gt_df["source1_entity_id"].astype(str).values
    matches_vals = gt_df["matched_entity_ids"].values if "matched_entity_ids" in gt_df else ["" for _ in range(len(gt_df))]

    for s1_id, raw_matches in zip(s1_vals, matches_vals):
        s1_clean = str(s1_id).strip()
        targets: List[str] = []
        if pd.notna(raw_matches):
            raw_s = str(raw_matches).strip()
            if raw_s:
                targets = [t.strip() for t in raw_s.split(",") if t.strip()]

        s1_to_targets[s1_clean] = targets
        uf.find(s1_clean)  # ensure s1_id is in union-find even if singleton

        for target in targets:
            uf.union(s1_clean, target)

    # Group S1 and targets by component root
    components: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"s1_ids": [], "target_ids": set()})

    for s1_id, targets in s1_to_targets.items():
        root = uf.find(s1_id)
        components[root]["s1_ids"].append(s1_id)
        components[root]["target_ids"].update(targets)

    return dict(components)


def verify_split_leakage(
    train_s1: Set[str],
    val_s1: Set[str],
    train_targets: Set[str],
    val_targets: Set[str],
) -> List[str]:
    """
    Verify strict zero-leakage invariants between train and validation partitions.

    Returns
    -------
    List[str]
        List of violation descriptions; empty if split is strictly leak-free.
    """
    violations: List[str] = []

    s1_overlap = train_s1 & val_s1
    if s1_overlap:
        violations.append(
            f"CRITICAL: {len(s1_overlap)} Source 1 entities leak across train and validation!"
        )

    target_overlap = train_targets & val_targets
    if target_overlap:
        violations.append(
            f"CRITICAL: {len(target_overlap)} matched targets leak across train and validation!"
        )

    return violations


def create_grouped_validation_split(
    gt_df: pd.DataFrame,
    s1_df: pd.DataFrame | Dict[str, str],
    s2_ids: Optional[Set[str]] = None,
    s3_ids: Optional[Set[str]] = None,
    val_fraction: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Create a shared, entity-disjoint, grouped train/validation split.

    Guarantees:
    1. Entire connected components remain strictly in one fold (zero target leakage).
    2. Stratified across country, singleton status, and match-count bucket.
    3. Deterministic whole-dataset partition with configurable holdout fraction.
    4. Distractors (unmatched targets) partitioned consistently across folds.

    Parameters
    ----------
    gt_df : pd.DataFrame
        Ground truth dataframe with 'source1_entity_id' and 'matched_entity_ids'.
    s1_df : pd.DataFrame or Dict[str, str]
        Source 1 records (with 'entity_id' and 'country') or mapping of s1_id -> country.
    s2_ids : Set[str], optional
        Complete set of Source 2 entity IDs (to identify distractors).
    s3_ids : Set[str], optional
        Complete set of Source 3 entity IDs (to identify distractors).
    val_fraction : float, default 0.15
        Target proportion of Source 1 entities in validation fold.
    random_seed : int, default 42
        Seed for deterministic pseudo-random shuffling.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'train_s1_ids', 'val_s1_ids', 'train_target_ids',
        'val_target_ids', 'distractor_split', and 'metadata'.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    # Build S1 -> Country mapping
    if isinstance(s1_df, pd.DataFrame):
        s1_country_map = dict(zip(s1_df["entity_id"].astype(str).str.strip(), s1_df["country"].astype(str).str.strip()))
    else:
        s1_country_map = dict(s1_df)

    # Step 1: Compute connected components
    components = compute_connected_components(gt_df)
    total_components = len(components)
    total_s1 = sum(len(c["s1_ids"]) for c in components.values())

    # Step 2: Stratify clusters directly without expensive dataframe allocation
    # Group components into strata: (country, is_singleton, match_bucket)
    strata: Dict[Tuple[str, bool, str], List[str]] = defaultdict(list)

    for root_id, comp in components.items():
        s1_list = comp["s1_ids"]
        target_set = comp["target_ids"]

        # Majority country for cluster
        countries = [s1_country_map.get(s1, "Unknown") for s1 in s1_list]
        country = Counter(countries).most_common(1)[0][0]

        is_singleton = (len(target_set) == 0)
        n_targets = len(target_set)

        if is_singleton:
            match_bucket = "0"
        elif n_targets <= 2:
            match_bucket = "1-2"
        elif n_targets <= 5:
            match_bucket = "3-5"
        else:
            match_bucket = "6+"

        strat_key = (country, is_singleton, match_bucket)
        strata[strat_key].append(root_id)

    # Step 3: Stratified allocation to Train and Validation
    rng = np.random.default_rng(random_seed)

    train_roots: List[str] = []
    val_roots: List[str] = []

    # Sort keys for absolute determinism across platforms
    sorted_strata_keys = sorted(strata.keys(), key=lambda k: (k[0], k[1], k[2]))

    for strat_key in sorted_strata_keys:
        root_ids = strata[strat_key]
        shuffled = list(root_ids)
        # Use rng.permutation for deterministic array shuffling
        perm = rng.permutation(len(shuffled))
        shuffled_roots = [shuffled[i] for i in perm]

        # Calculate target validation S1 count for this stratum
        stratum_total_s1 = sum(len(components[r]["s1_ids"]) for r in shuffled_roots)
        target_val_s1 = int(round(stratum_total_s1 * val_fraction))

        curr_val_s1 = 0
        for r_id in shuffled_roots:
            comp_s1_len = len(components[r_id]["s1_ids"])
            if (curr_val_s1 + comp_s1_len <= target_val_s1) or (curr_val_s1 == 0 and target_val_s1 > 0):
                val_roots.append(r_id)
                curr_val_s1 += comp_s1_len
            else:
                train_roots.append(r_id)

    # Step 4: Assemble S1 and Target ID sets
    train_s1_ids: List[str] = []
    val_s1_ids: List[str] = []
    train_targets: Set[str] = set()
    val_targets: Set[str] = set()

    for root_id in train_roots:
        train_s1_ids.extend(components[root_id]["s1_ids"])
        train_targets.update(components[root_id]["target_ids"])

    for root_id in val_roots:
        val_s1_ids.extend(components[root_id]["s1_ids"])
        val_targets.update(components[root_id]["target_ids"])

    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)

    # Step 5: Verify strict zero leakage
    leakage_violations = verify_split_leakage(train_s1_set, val_s1_set, train_targets, val_targets)
    if leakage_violations:
        raise RuntimeError(f"Split validation failed with leakage violations: {leakage_violations}")

    # Step 6: Distractor partitioning
    train_distractors: List[str] = []
    val_distractors: List[str] = []

    if s2_ids is not None or s3_ids is not None:
        all_pool_targets = set()
        if s2_ids:
            all_pool_targets.update(s2_ids)
        if s3_ids:
            all_pool_targets.update(s3_ids)

        all_matched_targets = train_targets | val_targets
        distractors = sorted(list(all_pool_targets - all_matched_targets))

        if distractors:
            shuffled_distractors = list(distractors)
            rng.shuffle(shuffled_distractors)
            n_val_distractors = int(round(len(shuffled_distractors) * val_fraction))
            val_distractors = shuffled_distractors[:n_val_distractors]
            train_distractors = shuffled_distractors[n_val_distractors:]

    # Step 7: Compute audit metadata
    train_singletons = sum(1 for s in train_s1_ids if len(components[s if s in components else train_roots[0]]["target_ids"]) == 0)
    val_singletons = sum(1 for s in val_s1_ids if len(components[s if s in components else val_roots[0]]["target_ids"]) == 0)

    train_countries = Counter(s1_country_map.get(s, "Unknown") for s in train_s1_ids)
    val_countries = Counter(s1_country_map.get(s, "Unknown") for s in val_s1_ids)

    cluster_sizes = [len(c["s1_ids"]) for c in components.values()]
    multi_s1_clusters = sum(1 for sz in cluster_sizes if sz > 1)
    max_cluster_size = max(cluster_sizes) if cluster_sizes else 0

    metadata = {
        "random_seed": random_seed,
        "target_val_fraction": val_fraction,
        "total_source1_entities": total_s1,
        "total_connected_components": total_components,
        "multi_s1_component_count": multi_s1_clusters,
        "max_s1_per_component": max_cluster_size,
        "train": {
            "s1_count": len(train_s1_ids),
            "s1_proportion": round(len(train_s1_ids) / total_s1, 6) if total_s1 else 0.0,
            "target_count": len(train_targets),
            "distractor_count": len(train_distractors),
            "singleton_count": train_singletons,
            "singleton_ratio": round(train_singletons / len(train_s1_ids), 4) if train_s1_ids else 0.0,
            "country_distribution": dict(train_countries),
        },
        "validation": {
            "s1_count": len(val_s1_ids),
            "s1_proportion": round(len(val_s1_ids) / total_s1, 6) if total_s1 else 0.0,
            "target_count": len(val_targets),
            "distractor_count": len(val_distractors),
            "singleton_count": val_singletons,
            "singleton_ratio": round(val_singletons / len(val_s1_ids), 4) if val_s1_ids else 0.0,
            "country_distribution": dict(val_countries),
        },
        "zero_leakage_verified": True,
        "s1_overlap_count": 0,
        "target_overlap_count": 0,
    }

    return {
        "metadata": metadata,
        "train_s1_ids": sorted(train_s1_ids),
        "val_s1_ids": sorted(val_s1_ids),
        "train_target_ids": sorted(list(train_targets)),
        "val_target_ids": sorted(list(val_targets)),
        "train_distractor_ids": sorted(train_distractors),
        "val_distractor_ids": sorted(val_distractors),
    }


def generate_geographic_transfer_split(
    gt_df: pd.DataFrame,
    s1_df: pd.DataFrame | Dict[str, str],
    train_country: str = "US",
    test_country: str = "India",
) -> Dict[str, Any]:
    """
    Generate an entity-disjoint cross-country transfer diagnostic split.

    Used for ARCHITECTURE.md Section 12 Transfer Diagnostics:
    - US -> India: Train on US entities, evaluate on India holdout
    - India -> US: Train on India entities, evaluate on US holdout
    """
    if isinstance(s1_df, pd.DataFrame):
        s1_country_map = dict(zip(s1_df["entity_id"].astype(str).str.strip(), s1_df["country"].astype(str).str.strip()))
    else:
        s1_country_map = dict(s1_df)

    components = compute_connected_components(gt_df)

    train_s1_ids: List[str] = []
    val_s1_ids: List[str] = []
    train_targets: Set[str] = set()
    val_targets: Set[str] = set()

    for root_id, comp in components.items():
        s1_list = comp["s1_ids"]
        target_set = comp["target_ids"]

        countries = [s1_country_map.get(s1, "Unknown") for s1 in s1_list]
        cluster_country = Counter(countries).most_common(1)[0][0]

        if cluster_country == train_country:
            train_s1_ids.extend(s1_list)
            train_targets.update(target_set)
        elif cluster_country == test_country:
            val_s1_ids.extend(s1_list)
            val_targets.update(target_set)

    leakage_violations = verify_split_leakage(set(train_s1_ids), set(val_s1_ids), train_targets, val_targets)
    if leakage_violations:
        logger.warning("Geographic transfer split has overlapping target entities: %s", leakage_violations)

    metadata = {
        "train_country": train_country,
        "test_country": test_country,
        "train_s1_count": len(train_s1_ids),
        "val_s1_count": len(val_s1_ids),
        "train_target_count": len(train_targets),
        "val_target_count": len(val_targets),
        "zero_target_leakage": len(train_targets & val_targets) == 0,
    }

    return {
        "metadata": metadata,
        "train_s1_ids": sorted(train_s1_ids),
        "val_s1_ids": sorted(val_s1_ids),
        "train_target_ids": sorted(list(train_targets)),
        "val_target_ids": sorted(list(val_targets)),
    }


def save_split_manifest(split_result: Dict[str, Any], output_dir: str | Path) -> Dict[str, Path]:
    """
    Save split manifest, metadata, and entity ID text files.

    Outputs:
    - {output_dir}/split_manifest.json (full dictionary with all IDs and metadata)
    - {output_dir}/split_metadata.json (metadata only, lightweight for quick inspection)
    - {output_dir}/train_s1_ids.txt (newline-delimited train S1 IDs)
    - {output_dir}/val_s1_ids.txt (newline-delimited validation S1 IDs)

    Returns
    -------
    Dict[str, Path]
        Dictionary of created file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "split_manifest.json"
    metadata_path = out_dir / "split_metadata.json"
    train_ids_path = out_dir / "train_s1_ids.txt"
    val_ids_path = out_dir / "val_s1_ids.txt"

    # Save full JSON manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_result, f, indent=2)

    # Save lightweight metadata JSON
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(split_result.get("metadata", {}), f, indent=2)

    # Save plaintext ID lists for memory-efficient loading by downstream modules
    with open(train_ids_path, "w", encoding="utf-8") as f:
        for s1_id in split_result["train_s1_ids"]:
            f.write(f"{s1_id}\n")

    with open(val_ids_path, "w", encoding="utf-8") as f:
        for s1_id in split_result["val_s1_ids"]:
            f.write(f"{s1_id}\n")

    logger.info("Saved split manifest bundle to %s", out_dir)
    return {
        "manifest": manifest_path,
        "metadata": metadata_path,
        "train_ids": train_ids_path,
        "val_ids": val_ids_path,
    }


def load_split_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    """Load split manifest from JSON."""
    in_p = Path(manifest_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Split manifest not found: {in_p}")
    with open(in_p, "r", encoding="utf-8") as f:
        return json.load(f)
