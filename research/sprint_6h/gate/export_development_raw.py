"""Recover raw records only for explicitly supplied development owner truth.

No truth expansion, text normalization, pseudo-label training or record fitting.
The caller supplies the already-authorized exact development label universe.
"""
import argparse
import json
from pathlib import Path

from build_raw_lookup import lookup_manifest, lookup_raw_records, open_raw_lookup, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--truth", required=True, type=Path)
    p.add_argument("--raw-lookup", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    truth = json.loads(a.truth.read_text())
    if len(truth) != 50000 or any(not key.startswith("S1-") for key in truth):
        raise ValueError("Expected the current exact development50k owner export")
    targets = [target for values in truth.values() for target in values]
    if len(targets) != len(set(targets)) or any(not t.startswith(("S2-", "S3-")) for t in targets):
        raise ValueError("Targets must be unique supplied Source2/3 records")
    complete = lookup_manifest(a.raw_lookup)
    conn = open_raw_lookup(a.raw_lookup)
    a.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, ids in (("references", sorted(truth)), ("target_records", sorted(targets))):
        records = lookup_raw_records(conn, ids)
        path = a.output / (label + ".json")
        if path.exists():
            raise ValueError("Raw development export already exists")
        path.write_text(json.dumps([records[key] for key in ids], ensure_ascii=False) + "\n")
        results[label] = {"path": str(path.resolve()), "count": len(ids), "sha256": sha256(path)}
    manifest = {"status": "complete", "role": "current50k_development_only",
                "truth_sha256": sha256(a.truth), "raw_source_manifest": complete,
                "normalization": "none", "labels_usage": "restrict exact evaluation record IDs only",
                "builder_sha256": sha256(__file__), "exports": results}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
