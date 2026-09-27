#!/usr/bin/env python3
"""Prepare a new frozen descriptor pinned to the metadata-only runner patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


class FreezeRepairError(RuntimeError):
    pass


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def expected_patched_runner(original: str) -> str:
    replacements = [
        ('    ids = [r["entity_id"] for r in rows]\n', '    reference_ids = [r["entity_id"] for r in rows]\n'),
        ('hashlib.sha256("\\n".join(ids).encode()).hexdigest():\n', 'hashlib.sha256("\\n".join(reference_ids).encode()).hexdigest():\n'),
        ('            ids = sorted({c for c in frame.candidate_entity_id if c.startswith(index.source)})\n', '            candidate_ids = sorted({c for c in frame.candidate_entity_id if c.startswith(index.source)})\n'),
        ('            for i in range(0, len(ids), 500):\n', '            for i in range(0, len(candidate_ids), 500):\n'),
        ('                part = ids[i:i + 500]\n', '                part = candidate_ids[i:i + 500]\n'),
        ('"input_ids_sha256": hashlib.sha256("\\n".join(ids).encode()).hexdigest(),', '"input_ids_sha256": hashlib.sha256("\\n".join(reference_ids).encode()).hexdigest(),'),
    ]
    patched = original
    for before, after in replacements:
        count = patched.count(before)
        if count != 1:
            raise FreezeRepairError(f"Expected exactly one runner occurrence, found {count}: {before.strip()}")
        patched = patched.replace(before, after)
    return patched


def build_descriptor(
    original_descriptor: Path,
    original_runner: Path,
    patched_runner: Path,
    audit_report: Path,
) -> tuple[dict, dict]:
    descriptor_bytes = original_descriptor.read_bytes()
    descriptor = json.loads(descriptor_bytes)
    audit = json.loads(audit_report.read_text())
    original_runner_sha = digest_file(original_runner)
    patched_runner_sha = digest_file(patched_runner)
    recorded_runner_sha = descriptor["code_sha256"]["scripts/run_frozen_pipeline.py"]
    if original_runner_sha != recorded_runner_sha:
        raise FreezeRepairError("Original runner does not match the original frozen descriptor")
    if patched_runner.read_text() != expected_patched_runner(original_runner.read_text()):
        raise FreezeRepairError("Patched runner differs from the exact metadata-only repair")
    original_descriptor_sha = digest_bytes(descriptor_bytes)
    if audit.get("frozen_sha256") != original_descriptor_sha:
        raise FreezeRepairError("Audit report does not belong to the original frozen descriptor")

    provenance = {
        "repair": "checkpoint_reference_id_hash_shadowing",
        "scope": "variable rename plus checkpoint metadata hash correction; score calculations unchanged",
        "score_logic_parity": "exact_expected_metadata_only_patch",
        "original_frozen_sha256": original_descriptor_sha,
        "original_runner_sha256": original_runner_sha,
        "patched_runner_sha256": patched_runner_sha,
        "original_audit_report_sha256": digest_file(audit_report),
        "original_audit_status": audit.get("status"),
        "original_audit_ship_accepted": audit.get("ship_decision", {}).get("accepted"),
    }
    repaired = json.loads(descriptor_bytes)
    repaired["code_sha256"]["scripts/run_frozen_pipeline.py"] = patched_runner_sha
    repaired["checkpoint_repair"] = provenance
    return repaired, provenance


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=1)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-descriptor", type=Path, required=True)
    parser.add_argument("--original-runner", type=Path, required=True)
    parser.add_argument("--patched-runner", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() in {args.original_descriptor.resolve(), args.audit_report.resolve()}:
        raise FreezeRepairError("Output must not overwrite the original descriptor or audit report")
    repaired, provenance = build_descriptor(
        args.original_descriptor, args.original_runner, args.patched_runner, args.audit_report
    )
    proposed = (json.dumps(repaired, indent=1) + "\n").encode()
    print(json.dumps({**provenance, "mode": "apply" if args.apply else "dry-run", "proposed_frozen_sha256": digest_bytes(proposed)}, sort_keys=True))
    if args.apply:
        if args.output.exists():
            raise FreezeRepairError(f"Refusing to overwrite existing output: {args.output}")
        atomic_write(args.output, repaired)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
