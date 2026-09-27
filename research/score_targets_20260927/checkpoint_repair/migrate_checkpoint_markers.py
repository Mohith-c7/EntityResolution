#!/usr/bin/env python3
"""Validate and migrate resumable chunk marker hashes.

Dry-run is the default. Applying requires both --apply and a new --backup-dir.
No row identifiers are printed or stored in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


class RepairError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def read_source_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        ids = [row["entity_id"] for row in csv.DictReader(stream, delimiter="\t")]
    if len(ids) != len(set(ids)):
        raise RepairError("Source 1 contains duplicate entity_id values")
    return ids


def validate_universe(source1: Path, universe_counts: Path, source_ids: list[str]) -> dict:
    with sqlite3.connect(universe_counts) as connection:
        row = connection.execute("SELECT value FROM meta WHERE key='universe'").fetchone()
    if row is None:
        raise RepairError("Universe-count database has no universe metadata")
    metadata = json.loads(row[0])
    actual_sha = sha256_file(source1)
    if metadata.get("sha256") != actual_sha:
        raise RepairError("Current Source 1 SHA-256 differs from universe-count metadata")
    if metadata.get("records") != len(source_ids):
        raise RepairError("Current Source 1 row count differs from universe-count metadata")
    return {"source1_sha256": actual_sha, "source1_records": len(source_ids)}


def read_parquet_ids(path: Path) -> tuple[list[str], list[str], int]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RepairError("pyarrow is required to inspect production parquet chunks") from exc
    table = parquet.read_table(path, columns=["source1_entity_id", "candidate_entity_id"], memory_map=True)
    source_ids = table.column("source1_entity_id").to_pylist()
    candidate_ids = table.column("candidate_entity_id").to_pylist()
    if any(value is None for value in source_ids + candidate_ids):
        raise RepairError(f"Null ID in {path}")
    return source_ids, candidate_ids, table.num_rows


@dataclass(frozen=True)
class MarkerPlan:
    chunk: int
    marker: str
    parquet: str
    entities: int
    pairs: int
    old_input_ids_sha256: str
    new_input_ids_sha256: str
    parquet_sha256: str
    state: str
    observed_source1_ids: int
    missing_source1_ids_without_candidates: int


def inspect_markers(
    source_ids: list[str],
    chunks: Path,
    *,
    batch_size: int = 250,
    wrong_hash_prefix: str = "S3",
    parquet_reader: Callable[[Path], tuple[list[str], list[str], int]] = read_parquet_ids,
) -> list[MarkerPlan]:
    marker_paths = sorted(chunks.glob("*.json"))
    parquet_paths = {path.stem: path for path in chunks.glob("*.parquet")}
    marker_stems = {path.stem for path in marker_paths}
    orphans = sorted(set(parquet_paths) - marker_stems)
    if orphans:
        raise RepairError(f"Parquet files without markers: {orphans[:10]}")

    plans: list[MarkerPlan] = []
    for marker_path in marker_paths:
        marker = json.loads(marker_path.read_text())
        try:
            chunk = int(marker_path.stem)
        except ValueError as exc:
            raise RepairError(f"Non-numeric marker name: {marker_path}") from exc
        if marker.get("chunk") != chunk:
            raise RepairError(f"Chunk field/name mismatch in {marker_path}")
        start = chunk * batch_size
        expected_ids = source_ids[start : start + batch_size]
        if not expected_ids:
            raise RepairError(f"Marker chunk {chunk} is outside current Source 1")
        if marker.get("entities") != len(expected_ids):
            raise RepairError(f"Entity count mismatch in chunk {chunk}")

        parquet_path = parquet_paths.get(marker_path.stem)
        if parquet_path is None:
            raise RepairError(f"Missing parquet for chunk {chunk}")
        actual_parquet_sha = sha256_file(parquet_path)
        if marker.get("parquet_sha256") != actual_parquet_sha:
            raise RepairError(f"Parquet SHA-256 mismatch in chunk {chunk}")

        parquet_source_ids, candidate_ids, pair_count = parquet_reader(parquet_path)
        if marker.get("pairs") != pair_count:
            raise RepairError(f"Pair count mismatch in chunk {chunk}")
        expected_set = set(expected_ids)
        observed_set = set(parquet_source_ids)
        unexpected = observed_set - expected_set
        if unexpected:
            raise RepairError(
                f"Chunk {chunk} parquet contains {len(unexpected)} Source 1 IDs outside its current slice"
            )

        correct_hash = sha256_ids(expected_ids)
        legacy_hash = sha256_ids(sorted({value for value in candidate_ids if value.startswith(wrong_hash_prefix)}))
        stored_hash = marker.get("input_ids_sha256")
        if stored_hash == correct_hash:
            state = "current_reference_hash"
        elif stored_hash == legacy_hash:
            state = "legacy_s3_candidate_hash"
        else:
            raise RepairError(f"Chunk {chunk} has neither the correct nor observed legacy input hash")
        plans.append(
            MarkerPlan(
                chunk=chunk,
                marker=str(marker_path),
                parquet=str(parquet_path),
                entities=len(expected_ids),
                pairs=pair_count,
                old_input_ids_sha256=stored_hash,
                new_input_ids_sha256=correct_hash,
                parquet_sha256=actual_parquet_sha,
                state=state,
                observed_source1_ids=len(observed_set),
                missing_source1_ids_without_candidates=len(expected_set - observed_set),
            )
        )
    return plans


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=1, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def apply_repairs(plans: list[MarkerPlan], backup_dir: Path, manifest: dict) -> int:
    if backup_dir.exists():
        raise RepairError(f"Backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True)
    marker_backup = backup_dir / "markers"
    marker_backup.mkdir()
    for plan in plans:
        shutil.copy2(plan.marker, marker_backup / Path(plan.marker).name)
    atomic_json(backup_dir / "manifest.json", manifest)

    changed = 0
    for plan in plans:
        if plan.state != "legacy_s3_candidate_hash":
            continue
        marker_path = Path(plan.marker)
        marker = json.loads(marker_path.read_text())
        marker["input_ids_sha256"] = plan.new_input_ids_sha256
        atomic_json(marker_path, marker)
        changed += 1
    return changed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--universe-counts", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--wrong-hash-prefix", default="S3")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.apply and args.backup_dir is None:
        raise RepairError("--apply requires an explicit --backup-dir")
    source_ids = read_source_ids(args.source1)
    source_evidence = validate_universe(args.source1, args.universe_counts, source_ids)
    plans = inspect_markers(
        source_ids,
        args.chunks,
        batch_size=args.batch_size,
        wrong_hash_prefix=args.wrong_hash_prefix,
    )
    manifest = {
        "format": 1,
        "source": source_evidence,
        "batch_size": args.batch_size,
        "wrong_hash_prefix": args.wrong_hash_prefix,
        "markers": [asdict(plan) for plan in plans],
    }
    legacy = sum(plan.state == "legacy_s3_candidate_hash" for plan in plans)
    current = len(plans) - legacy
    missing = sum(plan.missing_source1_ids_without_candidates for plan in plans)
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "markers_checked": len(plans),
                "legacy_markers": legacy,
                "already_current": current,
                "source1_ids_without_candidates": missing,
                **source_evidence,
            },
            sort_keys=True,
        )
    )
    if args.apply:
        changed = apply_repairs(plans, args.backup_dir, manifest)
        print(json.dumps({"markers_changed": changed, "backup_dir": str(args.backup_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
