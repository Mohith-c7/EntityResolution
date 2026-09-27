"""Collect exposure IDs and reserve disjoint outer-holdout cohorts without labels.

Scan experiment/model/report artifacts, never the raw truth database. Parquet
reads project only reference ID columns. Unknown artifact files are inventoried
for review; remote ledgers must be merged before a reservation is called fresh.
"""
import argparse
import csv
import hashlib
import heapq
import json
import re
import sys
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution"))
from src.evaluation.validation import entity_fold, stable_hash

ID = re.compile(r"^S1-[A-Za-z0-9_.:-]+$")
ID_COLUMNS = ("source1_entity_id", "entity_id", "reference_id", "owner", "owner_id")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def json_ids(value):
    if isinstance(value, str):
        if ID.fullmatch(value): yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if ID.fullmatch(str(key)): yield key
            yield from json_ids(item)
    elif isinstance(value, list):
        for item in value: yield from json_ids(item)


def collect(paths, output, excluded=()):
    import pyarrow.parquet as pq
    all_ids, sources, skipped, definitions = set(), [], [], []
    exclude = {Path(p).resolve() for p in excluded}
    files = sorted({f.resolve() for p in paths for f in ([p] if p.is_file() else p.rglob("*")) if f.is_file()})
    for path in files:
        if any(path == item or item in path.parents for item in exclude) or output.resolve() in path.parents: continue
        if path.name == "split_manifest.tsv":
            definitions.append({"path": str(path), "sha256": sha(path), "reason": "raw universe split definition; membership alone is not label exposure"})
            continue
        ids, method = set(), None
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                ids.update(json_ids(json.loads(path.read_text()))); method = "recursive_json_reference_ids"
            elif suffix == ".parquet":
                table = pq.ParquetFile(path)
                columns = [c for c in ID_COLUMNS if c in table.schema_arrow.names]
                if columns:
                    for batch in table.iter_batches(columns=columns, batch_size=65536, use_threads=False):
                        for col in columns:
                            ids.update(v for v in batch.column(col).to_pylist() if isinstance(v, str) and ID.fullmatch(v))
                    method = "projected_reference_columns"
            elif suffix in (".csv", ".tsv"):
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream, delimiter="\t" if suffix == ".tsv" else ",")
                    columns = [c for c in ID_COLUMNS if c in (reader.fieldnames or [])]
                    if columns:
                        for row in reader:
                            ids.update(row[c] for c in columns if ID.fullmatch(row[c] or ""))
                        method = "projected_reference_columns"
            elif suffix in (".md", ".log", ".jsonl"):
                for line in path.read_text().splitlines():
                    ids.update(re.findall(r"(?<![A-Za-z0-9])S1-[A-Za-z0-9_]+", line))
                method = "text_reference_id_tokens"
            elif suffix == ".npy":
                import numpy as np
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.dtype.kind in "US":
                    for start in range(0, array.size, 65536):
                        ids.update(str(v) for v in array.reshape(-1)[start:start + 65536] if ID.fullmatch(str(v)))
                    method = "string_array_reference_ids"
                else:
                    definitions.append({"path": str(path), "dtype": str(array.dtype), "shape": list(array.shape), "reason": "numeric array header inspected; no reference IDs"})
                    continue
            elif suffix in (".txt", ".ids") and ("id" in path.name or "reference" in path.name):
                for line in path.read_text().splitlines():
                    if ID.fullmatch(line.strip()): ids.add(line.strip())
                method = "reference_id_lines"
        except Exception as error:
            raise ValueError(f"Cannot safely scan {path}: {error}") from error
        if method:
            all_ids.update(ids)
            sources.append({"path": str(path), "sha256": sha(path), "reference_ids": len(ids), "method": method})
        elif suffix in (".npy", ".npz", ".sqlite", ".zip", ".gz", ".tar", ".tgz"):
            skipped.append({"path": str(path), "bytes": path.stat().st_size, "reason": "manual_review_required; numeric scores/indexes are not reference manifests"})
    output.mkdir(parents=True, exist_ok=False)
    (output / "exposed_ids.json").write_text(json.dumps(sorted(all_ids)) + "\n")
    report = {"status": "exposure_scan_complete_pending_scope_review", "references": len(all_ids), "sources": sources,
              "unparsed_artifacts": skipped, "universe_definitions_not_exposure": definitions, "roots": [str(p.resolve()) for p in paths], "excluded_paths": [str(p) for p in exclude],
              "scope_note": "Not a freshness certificate until all local/remote roots and unparsed ID-bearing archives are accounted for."}
    (output / "ledger.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def reserve(source, ledgers, output, count=10000, domain="sprint-6h-20260927", scope_reviewed=False, save_heldout_universe=False):
    if not scope_reviewed: raise ValueError("Review and merge every remote/local exposure source before reservation")
    exposed = set(); provenance = []
    for path in ledgers:
        raw = json.loads(path.read_text())
        if not isinstance(raw, list) or any(not isinstance(e, str) or not ID.fullmatch(e) for e in raw):
            raise ValueError("Ledger must be a JSON list of S1 IDs")
        exposed.update(raw); provenance.append({"path": str(path), "sha256": sha(path), "ids": len(raw)})
    before = source.stat(); heap, eligible, seen, heldout_rows = [], 0, set(), []
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            eid = row["entity_id"]
            if eid in seen: raise ValueError("Duplicate Source 1 entity ID")
            seen.add(eid)
            if entity_fold(eid) != "holdout": continue
            if save_heldout_universe: heldout_rows.append(row)
            if eid in exposed: continue
            eligible += 1; item = (-stable_hash(eid, domain), eid, row)
            if len(heap) < 2 * count: heapq.heappush(heap, item)
            elif item[0] > heap[0][0]: heapq.heapreplace(heap, item)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns): raise ValueError("Source changed")
    if len(heap) != 2 * count: raise ValueError("Insufficient unexposed outer-holdout references")
    rows = [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
    cohorts = {"confirmation": rows[:count], "audit": rows[count:]}
    output.mkdir(parents=True, exist_ok=False)
    for name, values in cohorts.items():
        target = output / name; target.mkdir()
        (target / "sampled_references.json").write_text(json.dumps({"holdout": values}, ensure_ascii=False) + "\n")
        (target / "entity_ids.json").write_text(json.dumps([r["entity_id"] for r in values]) + "\n")
        (target / "references.json").write_text(json.dumps(values, ensure_ascii=False) + "\n")
    if save_heldout_universe:
        (output / "heldout_universe.json").write_text(json.dumps(heldout_rows, ensure_ascii=False) + "\n")
    report = {"status": "confirmation_and_audit_reserved_no_labels_read", "sampling_domain": domain,
              "reserved_at": datetime.now(timezone.utc).isoformat(), "scope_reviewed": True,
              "cohort_fingerprints": {name: {"entities": len(values), "reference_ids_sha256": hashlib.sha256("\n".join(r["entity_id"] for r in values).encode()).hexdigest(), "sampled_references_sha256": sha(output / name / "sampled_references.json"), "references_sha256": sha(output / name / "references.json")} for name, values in cohorts.items()},
              "entities_per_cohort": count, "eligible_holdout": eligible, "excluded_references": len(exposed), "exposure_ledgers": provenance,
              "source": {"path": str(source), "sha256": sha(source)},
              "country_counts": {name: dict(Counter(r["country"] for r in values)) for name, values in cohorts.items()},
              "audit_policy": "Predictions may be generated now; labels remain sealed until the complete pipeline freeze. Evaluate once, with the control on identical entities."}
    if save_heldout_universe:
        report["claimant_universe"] = {"kind": "complete_heldout", "entities": len(heldout_rows), "records_sha256": sha(output / "heldout_universe.json"), "labels_read": False, "scoring_complete": False, "exposed_holdout_claimants_retained": True}
    (output / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan"); scan.add_argument("--root", type=Path, action="append", required=True); scan.add_argument("--exclude", type=Path, action="append", default=[]); scan.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("reserve"); r.add_argument("--source", type=Path, required=True); r.add_argument("--ledger", type=Path, action="append", required=True); r.add_argument("--output", type=Path, required=True); r.add_argument("--count", type=int, default=10000); r.add_argument("--domain", default="sprint-6h-20260927"); r.add_argument("--scope-reviewed", action="store_true"); r.add_argument("--save-heldout-universe", action="store_true")
    args = parser.parse_args()
    report = collect(args.root, args.output, args.exclude) if args.command == "scan" else reserve(args.source, args.ledger, args.output, args.count, args.domain, args.scope_reviewed, args.save_heldout_universe)
    print(json.dumps({k: v for k, v in report.items() if k not in ("sources", "unparsed_artifacts")}))

if __name__ == "__main__": main()
