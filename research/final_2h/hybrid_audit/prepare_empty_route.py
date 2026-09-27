"""Label-free disjoint Blue05-empty top4 evidence on the complete heldout graph."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "research/sprint_6h"),
                str(ROOT / "research/sprint_6h/sibling")]
import numpy as np
import pandas as pd
import reverse_competition as reverse
import train_reverse_adapter as adapter
from evaluate_sprint import decide_control
from prepare_neural_inputs import serialize, sha

KEYS = reverse.KEYS
INDEX = None


def init_worker(path):
    global INDEX
    INDEX = reverse.ReverseIndex(Path(path), corpus="train")


def features_for(task):
    e, c, owner, target = task
    values, diagnostics = reverse.pair_features(INDEX, target, owner)
    if e in diagnostics["rival_ids"]:
        raise ValueError("Current owner entered reverse rivals")
    return dict(zip(KEYS, [e, c]), **values)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def select_disjoint_top4(original, blue_chosen, main):
    """Select frozen p2 top4 FIRST, then drop main-route keys without refill."""
    accepted = set(original.loc[np.asarray(blue_chosen, dtype=bool), "source1_entity_id"])
    owners = set(original.source1_entity_id) - accepted
    all40 = original.loc[original.source1_entity_id.isin(owners)].copy().reset_index(drop=True)
    if all40.groupby("source1_entity_id").size().ne(40).any():
        raise ValueError("Truncated empty-owner groups")
    ranked = all40.sort_values(["source1_entity_id", "probability", "candidate_entity_id"],
                              ascending=[True, False, True], kind="stable")
    top4 = ranked.groupby("source1_entity_id", sort=False).head(4)
    main_keys = set(main[KEYS].itertuples(index=False, name=None))
    drop = np.array([k in main_keys for k in top4[KEYS].itertuples(index=False, name=None)], dtype=bool)
    extra = top4.loc[~drop].copy()
    return all40, extra, len(owners), int(drop.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, default=Path("models/sprint_6h/heldout_cohort/pairs.parquet"))
    p.add_argument("--original-manifest", type=Path, default=Path("models/sprint_6h/heldout_cohort/manifest.json"))
    p.add_argument("--blue", type=Path, default=Path("research/final_2h/neural_last4_fine_scored_heldout331k/pairs.parquet"))
    p.add_argument("--main-features", type=Path, default=Path("research/sprint_6h/sibling/neural_v2_features_heldout331k/features.parquet"))
    p.add_argument("--index", type=Path, default=Path("research/sprint_6h/reverse_competition/train_s1.sqlite"))
    p.add_argument("--decision-config", type=Path, default=Path("research/sprint_6h/coordination/baseline_freeze/frozen.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-seconds", type=int, default=1100)
    a = p.parse_args(); started = time.monotonic()
    if a.output.exists(): raise FileExistsError(a.output)
    if not 1 <= a.workers <= 8: raise ValueError("At most8 allocated workers")
    def cap():
        if time.monotonic() - started > a.max_seconds: raise TimeoutError("Bounded preparation elapsed")
    source = json.loads(a.original_manifest.read_text())
    blue_marker = json.loads((a.blue.parent / "manifest.json").read_text())
    original_sha, blue_sha = sha(a.original), sha(a.blue)
    if (source["pairs"] != 13240480 or len(source["reference_ids"]) != 331012
            or original_sha != source["pairs_sha256"]
            or blue_marker["original_pairs_sha256"] != original_sha
            or blue_marker["scored_pairs_sha256"] != blue_sha):
        raise ValueError("Complete original/Blue heldout graph provenance differs")
    original, blue = pd.read_parquet(a.original), pd.read_parquet(a.blue)
    if (not original[[*KEYS, "candidate_order"]].equals(blue[[*KEYS, "candidate_order"]])
            or not np.array_equal(original.probability.to_numpy(), blue.probability_original.to_numpy())
            or original.duplicated(KEYS).any()):
        raise ValueError("Original40 keys/order/p2 anchor changed")
    config = json.loads(a.decision_config.read_text())
    config = {k: config[k] for k in ("threshold", "t_first", "t_rest")}
    chosen, _ = decide_control(blue, blue, config)
    main_route = pd.read_parquet(a.main_features, columns=KEYS)
    if len(main_route) != 123600 or main_route.duplicated(KEYS).any():
        raise ValueError("Wrong complete original neural route")
    all40, extra, empty_count, overlap = select_disjoint_top4(original, chosen, main_route)
    del blue, original, chosen
    a.output.mkdir(parents=True)
    main_route.to_parquet(a.output / "main_route.parquet", index=False)
    extra[KEYS].sort_values(KEYS, kind="stable").to_parquet(a.output / "route.parquet", index=False)
    write(a.output / "route_manifest.json", {"status": "complete", "labels_read": False,
        "owners": 331012, "pairs": 13240480, "blue_empty_owners": empty_count,
        "top4_pairs_before_exclusion": 4 * empty_count, "original_route_overlaps_excluded": overlap,
        "extra_rows": len(extra), "extra_owners": int(extra.source1_entity_id.nunique()),
        "route_sha256": sha(a.output / "route.parquet"), "main_route_sha256": sha(a.output / "main_route.parquet"),
        "original_pairs_sha256": original_sha, "blue_pairs_sha256": blue_sha,
        "rule": "Blue4 fullgraph empty; ORIGINALp2 top4,targetIDasc ties; then exclude main keys; no refill",
        "source_sha256": sha(__file__), "seconds": time.monotonic() - started})
    print(json.dumps({"stage": "route_complete", "empty_owners": empty_count, "extra_pairs": len(extra),
                      "overlap_excluded": overlap, "seconds": time.monotonic() - started}), flush=True)
    cap()
    index = reverse.ReverseIndex(a.index, corpus="train")
    if index.meta["records"] != 2206821: raise ValueError("Incomplete train Source1 reverse index")
    owner_records = index.records(extra.source1_entity_id.unique())
    index_meta = index.index_manifest
    index.close()
    targets = {}
    for s in (2, 3):
        path = Path(f"models/index_train_S{s}.sqlite").resolve()
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
            meta = json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
            if not meta["complete"]: raise ValueError("Incomplete train target index")
            ids = extra.loc[extra.candidate_entity_id.str.startswith(f"S{s}-"), "candidate_entity_id"].unique()
            for begin in range(0, len(ids), 500):
                group = list(ids[begin:begin + 500]); placeholders = ",".join("?" for _ in group)
                for e, n, ad, c in conn.execute(f"SELECT id,name,address,country FROM records WHERE id IN ({placeholders})", group):
                    targets[e] = reverse.clean_record(dict(entity_id=e, business_name=n, business_address=ad, country=c))
    if set(owner_records) != set(extra.source1_entity_id) or set(targets) != set(extra.candidate_entity_id):
        raise ValueError("Missing actual Source1 or target indexed records")
    tasks = ((e, c, owner_records[e], targets[c]) for e, c in extra[KEYS].itertuples(index=False, name=None))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=init_worker, initargs=(str(a.index),)) as pool:
        for i, row in enumerate(pool.map(features_for, tasks, chunksize=24)):
            rows.append(row)
            if i % 1000 == 0:
                cap(); print(json.dumps({"stage": "reverse", "rows": i + 1, "total": len(extra),
                                        "seconds": time.monotonic() - started}), flush=True)
    raw = pd.DataFrame(rows, columns=[*KEYS, *reverse.FEATURE_NAMES])
    raw[reverse.FEATURE_NAMES] = raw[reverse.FEATURE_NAMES].astype(np.float32)
    joined = adapter.join_reverse(all40, raw, set(extra.index))
    if not np.isfinite(joined[adapter.FEATURES]).all().all(): raise ValueError("Nonfinite reverse36 inputs")
    joined.to_parquet(a.output / "features.parquet", index=False)
    raw.to_parquet(a.output / "reverse33.parquet", index=False)
    write(a.output / "manifest.json", {"status": "complete", "version": "hybrid-heldout-empty-extra36-v1",
        "features": adapter.FEATURES, "rows": len(joined), "labels_read": False,
        "audit_labels_read": False, "extension_labels_read": False,
        "features_sha256": sha(a.output / "features.parquet"), "pair_order_sha256": reverse.pair_order_hash(joined),
        "original_pairs_sha256": original_sha, "blue_pairs_sha256": blue_sha,
        "route_manifest_sha256": sha(a.output / "route_manifest.json"), "index_metadata": index_meta,
        "query_config": asdict(reverse.QueryConfig()), "feature_code_sha256": sha(reverse.__file__),
        "join_code_sha256": sha(adapter.__file__), "source_sha256": sha(__file__)})
    # Original supplied text only; never substitute normalized index text.
    records, hashes = {}, {}; owners, target_ids = set(joined.source1_entity_id), set(joined.candidate_entity_id)
    for s in (1, 2, 3):
        path = Path(f"dataset/train/train_source{s}.tsv"); hashes[f"S{s}"] = sha(path)
        wanted = owners if s == 1 else target_ids
        for batch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=100000):
            cap()
            for row in batch.loc[batch.entity_id.isin(wanted)].to_dict("records"):
                if row["entity_id"] in records: raise ValueError("Repeated raw supplied record")
                records[row["entity_id"]] = serialize(row)
    if set(records) != owners | target_ids: raise ValueError("Missing original raw supplied text")
    inputs = a.output / "neural_inputs"; inputs.mkdir()
    with (inputs / "pairs.jsonl").open("x") as stream:
        for e, c in joined[KEYS].itertuples(index=False, name=None):
            stream.write(json.dumps(dict(zip(KEYS, [e, c]), text_left=records[e], text_right=records[c]), ensure_ascii=False) + "\n")
    write(inputs / "manifest.json", {"status": "complete", "rows": len(joined), "labels_read": False,
        "source_split": "heldout", "input_sha256": sha(inputs / "pairs.jsonl"),
        "feature_manifest_sha256": sha(a.output / "manifest.json"), "feature_file_sha256": sha(a.output / "features.parquet"),
        "source_tsv_sha256": hashes, "code_sha256": sha(__file__), "audit_labels_read": False,
        "extension_labels_read": False, "seconds": time.monotonic() - started})
    print(json.dumps({"stage": "raw_inputs_complete", "rows": len(joined), "output": str(a.output),
                      "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__": main()
