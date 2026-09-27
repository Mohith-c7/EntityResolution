"""Stream official supplied target text into an atomic, hash-bound SQLite lookup.

No labels, normalization or identity augmentation. Consumers require a complete
marker and retrieve only their candidate IDs through ``lookup_raw_records``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

LOOKUP_VERSION = "supplied-raw-target-lookup-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_manifest(sources):
    manifest = {}
    for source, path in sources.items():
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        manifest[source] = {"path": str(path), "bytes": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}
    return manifest


def lookup_manifest(path):
    """Fail on an incomplete/incorrect schema rather than serving partial data."""
    path = Path(path)
    marker = path.with_name(path.name + ".complete.json")
    if not marker.is_file():
        raise ValueError("Raw target lookup has no complete marker")
    external = json.loads(marker.read_text())
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        metadata = dict(conn.execute("SELECT key,value FROM metadata"))
        if metadata.get("status") != "complete" or metadata.get("version") != LOOKUP_VERSION:
            raise ValueError("Raw target lookup is not complete or has unsupported version")
        manifest = json.loads(metadata["manifest"])
        if any(external.get(key) != manifest[key] for key in manifest):
            raise ValueError("Raw target lookup complete marker disagrees with metadata")
        return manifest


def open_raw_lookup(path):
    lookup_manifest(path)
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def lookup_raw_records(conn, entity_ids):
    """Return raw record dictionaries keyed by ID; raise for missing candidates."""
    wanted = sorted(set(entity_ids))
    records = {}
    for start in range(0, len(wanted), 500):
        part = wanted[start:start + 500]
        query = f"SELECT id,business_name,business_address,country FROM records WHERE id IN ({','.join('?' for _ in part)})"
        for entity_id, name, address, country in conn.execute(query, part):
            records[entity_id] = {"entity_id": entity_id, "business_name": name,
                                  "business_address": address, "country": country}
    missing = set(wanted) - records.keys()
    if missing:
        raise ValueError(f"Raw target lookup missing {len(missing)} requested IDs")
    return records


def build_raw_lookup(sources, output, batch_size=5000, expected_hashes=None):
    """Write a new complete lookup, or verify and reuse an identical one."""
    if not sources or set(sources) - {"S1", "S2", "S3"} or batch_size < 1:
        raise ValueError("Use nonempty S1/S2/S3 source mapping and positive batch size")
    output = Path(output)
    inputs = _input_manifest(sources)
    if expected_hashes:
        if set(expected_hashes) != set(inputs) or any(inputs[source]["sha256"] != expected_hashes[source]
            for source in inputs):
            raise ValueError("Expected source hashes do not match raw lookup inputs")
    if output.exists():
        manifest = lookup_manifest(output)
        if manifest["inputs"] != inputs:
            raise ValueError("Existing raw lookup belongs to different source inputs")
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise ValueError("Partial raw lookup already exists; inspect it before retrying")
    counts = {}
    conn = sqlite3.connect(partial)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA cache_size=-131072")
        # Load sequential heap rows, then sort/build one unique lookup index.
        # Random ID insertions into a growing primary-key B-tree are needlessly
        # expensive for millions of supplied rows on bounded-IOPS disks.
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("CREATE TABLE records(id TEXT NOT NULL, business_name TEXT, business_address TEXT, country TEXT, source TEXT)")
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID")
        conn.executemany("INSERT INTO metadata VALUES (?,?)", [("version", LOOKUP_VERSION), ("status", "building")])
        conn.commit()
        for source, details in inputs.items():
            count, batch = 0, []
            with Path(details["path"]).open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                required = {"entity_id", "business_name", "business_address", "country"}
                if not required <= set(reader.fieldnames or []):
                    raise ValueError("Raw target TSV lacks required columns")
                for row in reader:
                    if any(row.get(field) is None for field in required):
                        raise ValueError("Raw target TSV contains a ragged record")
                    entity_id = row["entity_id"]
                    if not entity_id or not entity_id.startswith(source):
                        raise ValueError("Target row has missing/wrong-source ID")
                    batch.append((entity_id, row["business_name"], row["business_address"], row["country"], source))
                    count += 1
                    if len(batch) >= batch_size:
                        conn.executemany("INSERT INTO records VALUES (?,?,?,?,?)", batch)
                        conn.commit()
                        batch.clear()
                if batch:
                    conn.executemany("INSERT INTO records VALUES (?,?,?,?,?)", batch)
                    conn.commit()
            current = Path(details["path"]).stat()
            if current.st_size != details["bytes"] or current.st_mtime_ns != details["mtime_ns"]:
                raise ValueError("Source TSV changed during raw lookup build")
            counts[source] = count
        conn.execute("CREATE UNIQUE INDEX record_ids ON records(id)")
        conn.commit()
        manifest = {"version": LOOKUP_VERSION, "inputs": inputs, "source_counts": counts,
                    "records": sum(counts.values()), "normalization": "none", "labels_opened": False,
                    "builder_code_sha256": sha256(__file__)}
        conn.execute("INSERT INTO metadata VALUES ('manifest',?)", (json.dumps(manifest, sort_keys=True),))
        conn.execute("UPDATE metadata SET value='complete' WHERE key='status'")
        conn.commit()
        if conn.execute("SELECT count(*) FROM records").fetchone()[0] != manifest["records"]:
            raise ValueError("Raw target lookup row count mismatch")
        conn.close()
        partial.rename(output)
        marker = output.with_name(output.name + ".complete.json")
        marker_partial = marker.with_name(marker.name + ".partial")
        marker_partial.write_text(json.dumps({**manifest, "sqlite_sha256": sha256(output)}, indent=2, sort_keys=True) + "\n")
        marker_partial.rename(marker)
        return manifest
    except Exception:
        conn.close()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", type=Path)
    parser.add_argument("--s2", type=Path)
    parser.add_argument("--s3", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--expected-s2-sha256")
    parser.add_argument("--expected-s3-sha256")
    parser.add_argument("--expected-s1-sha256")
    args = parser.parse_args()
    sources = {source: path for source, path in (("S1", args.s1), ("S2", args.s2), ("S3", args.s3)) if path}
    expected = {source: value for source, value in (("S1", args.expected_s1_sha256), ("S2", args.expected_s2_sha256),
        ("S3", args.expected_s3_sha256)) if value}
    print(json.dumps(build_raw_lookup(sources, args.output, args.batch_size, expected or None), sort_keys=True))


if __name__ == "__main__":
    main()
