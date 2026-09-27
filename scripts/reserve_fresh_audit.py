"""Reserve a fresh entity-level audit from the outer holdout fold without reading labels.

Every reference in any earlier sampled_references.json is excluded, so no
inspected or previously audited entity can enter the new audit.
"""
import argparse
import csv
import heapq
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.evaluation.validation import entity_fold, stable_hash
from prepare_scale_experiment import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("models/audit_v2"))
    p.add_argument("--entities", type=int, default=10000)
    p.add_argument("--domain", default="frozen-bridge-audit-20260927")
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    seen, sources = set(), []
    for path in sorted(Path("models").rglob("sampled_references.json")):
        for values in json.loads(path.read_text()).values(): seen.update(r["entity_id"] for r in values)
        sources.append({"file": str(path), "sha256": sha(path)})
    source = Path("dataset/train/train_source1.tsv")
    before = source.stat()
    heap, eligible = [], 0
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            eid = row["entity_id"]
            if eid in seen or entity_fold(eid) != "holdout": continue
            eligible += 1
            item = (-stable_hash(eid, args.domain), eid, row)
            if len(heap) < args.entities: heapq.heappush(heap, item)
            elif item[0] > heap[0][0]: heapq.heapreplace(heap, item)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns): raise ValueError("Input changed")
    if len(heap) != args.entities: raise ValueError("Insufficient eligible references")
    rows = sorted((item[2] for item in heap), key=lambda r: r["entity_id"])
    if {r["entity_id"] for r in rows} & seen: raise ValueError("Audit contains inspected references")
    args.output.mkdir(parents=True)
    (args.output / "sampled_references.json").write_text(json.dumps({"holdout": rows}, ensure_ascii=False))
    report = {"status": "audit_reserved_no_labels_read", "entities": len(rows), "eligible_holdout_entities": eligible,
              "country_counts": dict(Counter(r["country"] for r in rows)), "sampling_domain": args.domain,
              "outer_split": "Existing deterministic entity_fold, seed 42", "excluded_references": len(seen),
              "exclusion_sources": sources, "input": {"path": str(source), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns},
              "audit_policy": "Freeze the complete pipeline before reading these labels; evaluate once."}
    (args.output / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("exclusion_sources", "input")}))


if __name__ == "__main__": main()
