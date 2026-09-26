"""Reserve larger entity splits without reading any new audit labels."""
import argparse
import csv
import hashlib
import heapq
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.evaluation.validation import entity_fold, stable_hash


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("models/scale_v1_plan"))
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    os.nice(10)
    base = json.loads(Path("models/features_v3_audit01/sampled_references.json").read_text())
    kept = {"train": base["train"], "tune": base["tune"], "holdout": []}
    totals = {"train": 300000, "tune": 50000, "holdout": 10000}
    needed = {f: totals[f] - len(kept[f]) for f in kept}
    seen, sources = set(), []
    for path in sorted(Path("models").rglob("sampled_references.json")):
        data = json.loads(path.read_text())
        for values in data.values(): seen.update(r["entity_id"] for r in values)
        sources.append({"file": str(path), "sha256": sha(path)})
    heaps = {f: [] for f in kept}
    source = Path("dataset/train/train_source1.tsv")
    before = source.stat()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            eid = row["entity_id"]
            if eid in seen: continue
            fold = entity_fold(eid)
            if fold not in heaps: continue
            priority = stable_hash(eid, "scale-v1-reservation-20260926")
            heap = heaps[fold]
            item = (-priority, eid, row)
            if len(heap) < needed[fold]: heapq.heappush(heap, item)
            elif priority < -heap[0][0]: heapq.heapreplace(heap, item)
    if any(len(heaps[f]) != needed[f] for f in kept): raise ValueError("Insufficient eligible references")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns): raise ValueError("Input changed")
    refs = {f: sorted(kept[f] + [item[2] for item in heaps[f]], key=lambda r: r["entity_id"]) for f in kept}
    sets = {f: {r["entity_id"] for r in rows} for f, rows in refs.items()}
    if sets["train"] & sets["tune"] or sets["train"] & sets["holdout"] or sets["tune"] & sets["holdout"]:
        raise ValueError("Entity split overlap")
    if sets["holdout"] & seen: raise ValueError("Audit contains inspected references")
    args.output.mkdir(parents=True)
    (args.output / "sampled_references.json").write_text(json.dumps(refs, ensure_ascii=False))
    report = {"status": "splits_reserved_no_model_trained_no_audit_labels_read", "counts": totals,
        "country_counts": {f: dict(Counter(r["country"] for r in rows)) for f, rows in refs.items()},
        "outer_split": "Existing deterministic entity_fold, seed 42",
        "sampling_domain": "scale-v1-reservation-20260926", "exclusion_sources": sources,
        "training_alias_policy": "Five inner entity folds. Each training reference uses aliases fitted without its inner fold; tuning/audit use aliases fitted on outer training only.",
        "cache_policy": "Version candidates and supervised features by alias-model hash and fold. Shared normalized records/postings may be label-free; a single supervised feature cache is not universal.",
        "ownership_policy": "Do not claim full-universe ownership accuracy from these sampled references. Require a separate complete competition evaluation with compatible fold-specific supervision.",
        "audit_policy": "Holdout IDs reserved without labels. Freeze selection on tuning first; read audit labels once for final confirmation.",
        "input": {"path": str(source), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns}}
    (args.output / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    public = {k: v for k, v in report.items() if k not in ("exclusion_sources", "input")}
    Path("reports/experiments/redesign/scale_experiment_plan.json").write_text(json.dumps(public, indent=2) + "\n")
    print(json.dumps(public))


if __name__ == "__main__": main()
