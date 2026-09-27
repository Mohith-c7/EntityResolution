"""Subset a verified current 50k tune workbench without changing its decisions.

Labels stay in separate files. Incumbent accepted states retain the FULL source
50k claimant universe, including rivals outside each 20k/10k role. This is not a
historical full-ownership replay. No old probability array is consumed.
"""
import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import train_sibling_matcher as sibling


def read_owner_truth(ids, database):
    """One indexed TEMP-table join; do not scan owners once per 500-ID batch."""
    truth = {e: [] for e in ids}
    with sqlite3.connect(Path(database).resolve().as_uri()+"?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("CREATE TEMP TABLE wanted(owner TEXT PRIMARY KEY) WITHOUT ROWID")
        conn.executemany("INSERT INTO wanted VALUES(?)", ((e,) for e in ids))
        for target, owner in conn.execute("SELECT o.id,o.owner FROM owners o JOIN wanted w ON o.owner=w.owner"):
            truth[owner].append(target)
    return {e: sorted(targets) for e, targets in truth.items()}


def role_ids(ids):
    if len(ids) != 50000 or len(set(ids)) != 50000:
        raise ValueError("Expected the exact 50,000 current outer-tune owner IDs")
    ordered = sorted(ids, key=lambda e: hashlib.sha256(("context-split-v1:"+e).encode()).digest())
    return {"residual_train": ordered[:20000], "early_stop": ordered[20000:30000], "selection": ordered[30000:]}


def build_roles(frame, manifest, truth, countries, roles, output, source_manifest_hash):
    source_pair_hash = manifest["pairs_sha256"]
    reports = {}
    for role, ids in roles.items():
        directory = Path(output) / role
        directory.mkdir(parents=True)
        part = frame[frame.source1_entity_id.isin(ids)].reset_index(drop=True)
        role_truth = {e: truth[e] for e in ids}
        role_countries = {e: countries[e] for e in ids}
        values = sibling.entity_values(part, part.accepted.to_numpy(dtype=bool), role_truth)
        pair_path = directory / "pairs.parquet"
        part.to_parquet(pair_path, index=False)
        sibling.write_json(directory / "truth.json", role_truth)
        sibling.write_json(directory / "countries.json", role_countries)
        sibling.write_json(directory / "reference_ids.json", ids)
        role_manifest = {**manifest, "status": "complete", "split": "development", "role": role,
            "reference_ids": ids, "entities": len(ids), "pairs": len(part),
            "pairs_sha256": sibling.digest(pair_path), "pair_order_sha256": sibling.pair_order_hash(part),
            "expected_macro_f05": float(np.mean(list(values.values()))),
            "expected_country_macro_f05": {c: float(np.mean([values[e] for e in ids if countries[e] == c]))
                for c in sorted(set(role_countries.values()))},
            "claims_universe_pairs_sha256": source_pair_hash,
            "routing_universe_pair_order_sha256": manifest.get("routing_universe_pair_order_sha256", manifest.get("pair_order_sha256")),
            "claims_universe_entities": len(manifest["reference_ids"]),
            "claims_universe_reference_ids_sha256": hashlib.sha256("\n".join(manifest["reference_ids"]).encode()).hexdigest(),
            "source_workbench_manifest_sha256": source_manifest_hash,
            "decision_scope": "Incumbent accepted decisions inherited unchanged from complete supplied 50k tune cohort; role-only decisions were not rerun. Outside-cohort rivals are incomplete.",
            "historical_full_ownership_replayed": False, "labels_separate": True,
            "truth_sha256": sibling.digest(directory / "truth.json"),
            "countries_sha256": sibling.digest(directory / "countries.json")}
        role_manifest.pop("chunks", None)
        sibling.write_json(directory / "manifest.json", role_manifest)
        reports[role] = {k: role_manifest[k] for k in ("entities", "pairs", "expected_macro_f05", "expected_country_macro_f05")}
    sibling.write_json(Path(output) / "report.json", {"status": "complete", "roles": reports,
        "role_disjoint": not any(set(a)&set(b) for i,a in enumerate(roles.values()) for b in list(roles.values())[i+1:]),
        "scope": "Current frozen scoring and supplied claimant cohort only; not a reproduction of historical full-ownership 0.979820."})
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tune-reference-ids", type=Path, default=Path("models/bridge_v1/first_stage/tune_references.json"))
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--truth", type=Path)
    group.add_argument("--owner-db", type=Path)
    p.add_argument("--references", type=Path, help="Raw record JSON list for countries, defaults to pairs directory/references.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    pa.set_cpu_count(1)
    started = time.perf_counter()
    frame, manifest = sibling.load_pairs(args.pairs, args.manifest)
    if manifest["split"] != "development": raise ValueError("Role builder only accepts current outer-tune development")
    ids = json.loads(args.tune_reference_ids.read_text())
    if set(manifest["reference_ids"]) != set(ids): raise ValueError("Current tune workbench owner set differs from model tune IDs")
    roles = role_ids(ids)
    truth = json.loads(args.truth.read_text()) if args.truth else read_owner_truth(ids, args.owner_db)
    if not set(ids) <= set(truth): raise ValueError("Truth omits known current tune owners")
    references = args.references or args.pairs.parent / "references.json"
    if references.exists():
        countries = {r["entity_id"]: r["country"].strip().casefold() for r in json.loads(references.read_text())}
    elif "country" in frame:
        countries = frame.groupby("source1_entity_id").country.first().to_dict()
    else: raise ValueError("Country coverage requires raw references or pair country field")
    if not set(ids) <= set(countries): raise ValueError("Missing country for zero-candidate or other owner")
    report = build_roles(frame, manifest, truth, countries, roles, args.output, sibling.digest(args.manifest))
    sibling.write_json(args.output / "lineage.json", {"source_pairs_sha256": sibling.digest(args.pairs),
        "source_manifest_sha256": sibling.digest(args.manifest), "tune_ids_sha256": sibling.digest(args.tune_reference_ids),
        "owner_db_sha256": sibling.digest(args.owner_db) if args.owner_db else None,
        "truth_sha256": sibling.digest(args.truth) if args.truth else None,
        "seconds": time.perf_counter()-started, "code_sha256": sibling.digest(__file__)})
    print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
