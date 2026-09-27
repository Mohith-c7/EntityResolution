"""Attach label-free Source 1 competition counts to a cached candidate set.

Only record text is read: the split's complete Source 1 file, the references and
the target indexes. Labels are never loaded. Output chunks align row for row
with the input cache chunks.
"""
import argparse
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.disk_index import normalize_record
from src.features.competition import COMPETITION_NAMES, KEY_KINDS, competition_features, count_universe, record_keys
from train_scale_model import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("models/next_round/extended_context_cache"))
    p.add_argument("--references", type=Path, default=Path("models/scale_v1_plan/sampled_references.json"))
    p.add_argument("--reference-split", default="tune")
    p.add_argument("--universe", type=Path, default=Path("dataset/train/train_source1.tsv"))
    p.add_argument("--index-prefix", default="models/index_train")
    p.add_argument("--output", type=Path, default=Path("models/next_round/competition_v1"))
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    start = time.perf_counter()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if "status" not in manifest and (args.cache / "progress.json").exists():
        manifest = {**manifest, **json.loads((args.cache / "progress.json").read_text())}
    if manifest.get("status") != "complete": raise ValueError("Candidate cache is incomplete")
    chunks = [(n, pd.read_parquet(args.cache / f"{n:06d}.parquet", columns=["source1_entity_id", "candidate_entity_id"]))
              for n in range(manifest["chunks"])]
    rows = json.loads(args.references.read_text())[args.reference_split]
    references = {r["entity_id"]: normalize_record(r) for r in rows}
    if any(not set(f.source1_entity_id) <= set(references) for _, f in chunks):
        raise ValueError("Cached references are missing from the reference file")
    reference_keys = {e: record_keys(r.name, r.core, r.address) for e, r in references.items()}

    wanted_targets = sorted({c for _, f in chunks for c in f.candidate_entity_id})
    target_keys = {}
    for source in ("S2", "S3"):
        path = Path(f"{args.index_prefix}_{source}.sqlite").resolve()
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        ids = [t for t in wanted_targets if t.startswith(source + "-")]
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            for cid, name, core, address in connection.execute(
                    f"SELECT id,name,core,address FROM records WHERE id IN ({','.join('?' * len(part))})", part):
                target_keys[cid] = record_keys(name, core, address)
        connection.close()
    if len(target_keys) != len(wanted_targets): raise ValueError("Some candidate targets are missing from the indexes")
    print(json.dumps({"stage": "records_loaded", "references": len(references), "targets": len(target_keys),
                      "seconds": round(time.perf_counter() - start, 1)}), flush=True)

    wanted = {kind: set() for kind in KEY_KINDS}
    for keys in (*reference_keys.values(), *target_keys.values()):
        for kind in KEY_KINDS:
            if keys[kind] is not None: wanted[kind].add(keys[kind])
    seen, universe_size = set(), 0

    def universe():
        nonlocal universe_size
        with args.universe.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                record = normalize_record(row)
                universe_size += 1
                if record.entity_id in references: seen.add(record.entity_id)
                yield record_keys(record.name, record.core, record.address)

    counts = count_universe(universe(), wanted)
    if seen != set(references): raise ValueError("Every reference must belong to the counted Source 1 universe")
    print(json.dumps({"stage": "universe_counted", "records": universe_size,
                      "seconds": round(time.perf_counter() - start, 1)}), flush=True)

    args.output.mkdir(parents=True)
    pairs = 0
    for number, frame in chunks:
        matrix = np.asarray([competition_features(reference_keys[s], target_keys[c], counts)
                             for s, c in zip(frame.source1_entity_id, frame.candidate_entity_id)], dtype=np.float32)
        for i, name in enumerate(COMPETITION_NAMES): frame[name] = matrix[:, i]
        destination = args.output / f"{number:06d}.parquet"
        frame.to_parquet(destination, index=False)
        write_json(args.output / f"{number:06d}.json", {"pairs": len(frame), "sha256": digest(destination)})
        pairs += len(frame)
    write_json(args.output / "manifest.json", {
        "status": "complete", "feature_names": list(COMPETITION_NAMES), "chunks": len(chunks), "pairs": pairs,
        "source_cache": str(args.cache), "source_manifest_sha256": digest(args.cache / "manifest.json"),
        "references": str(args.references), "reference_split": args.reference_split,
        "universe": str(args.universe), "universe_sha256": digest(args.universe), "universe_records": universe_size,
        "index_prefix": args.index_prefix, "labels_read": False,
        "policy": "Counts use Source 1 record text from one split only; the current reference is excluded; -1 marks a missing key.",
        "seconds": time.perf_counter() - start})
    print(json.dumps({"stage": "complete", "pairs": pairs, "seconds": round(time.perf_counter() - start, 1)}), flush=True)


if __name__ == "__main__":
    main()
