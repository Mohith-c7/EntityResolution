"""Stable entity folds and whole-stream sampling, independent of row order."""

import csv
import hashlib
import heapq
from pathlib import Path

SPLIT_VERSION = "entity-hash-v1-seed42-70-15-15"


def stable_hash(value: str, domain: str, seed: int = 42) -> int:
    return int.from_bytes(hashlib.blake2b(f"{domain}:{seed}:{value}".encode(), digest_size=8).digest(), "big")


def entity_fold(entity_id: str, seed: int = 42) -> str:
    value = stable_hash(entity_id, "fold", seed) / 2**64
    return "train" if value < .70 else "tune" if value < .85 else "holdout"


def sample_references(path: Path, sizes: dict[str, int], manifest: Path, seed: int = 42) -> tuple[dict, dict]:
    if any(type(n) is not int or n < 1 for n in sizes.values()):
        raise ValueError("Each requested fold sample must contain at least one entity")
    heaps = {fold: [] for fold in sizes}
    counts = {fold: 0 for fold in sizes}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with path.open(encoding="utf-8-sig", newline="") as stream, manifest.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "fold"])
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            fold = entity_fold(eid, seed)
            counts[fold] += 1
            writer.writerow([eid, fold])
            priority = stable_hash(eid, "sample", seed)
            heap = heaps[fold]
            item = (-priority, eid, row)
            if len(heap) < sizes[fold]:
                heapq.heappush(heap, item)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, item)
    references = {fold: [item[2] for item in sorted(heap, key=lambda item: item[1])] for fold, heap in heaps.items()}
    return references, {"version": SPLIT_VERSION, "seed": seed, "population_counts": counts,
        "sample_counts": {fold: len(rows) for fold, rows in references.items()},
        "policy": "Hash-based S1 entity folds; not exact stratification. Matching targets follow their owner; unmatched targets use their own stable hash."}
