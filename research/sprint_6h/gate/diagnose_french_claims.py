"""Unlabelled saved-claim diagnostics, not France accuracy or a release.

Scan the verified complete scored universe. Only exact baseline-floor-eligible
French pairs receive raw pair signals. Seven separate pair-keyed tables compare
the original six v3 settings and the single predeclared fixed v4 ablation.
Every omitted scored row is explicitly skipped, not observed street agreement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import france_gate as signals_v2
import house_agreement_gate_v3 as v3
import distinctive_street_gate_v4 as v4
from build_raw_lookup import lookup_manifest, lookup_raw_records, open_raw_lookup, sha256

KEYS = ["source1_entity_id", "candidate_entity_id"]
SCORE_SCHEMA = pa.schema([(key, pa.string()) for key in KEYS] + [("probability", pa.float64())])
GATE_SCHEMA = pa.schema([(key, pa.string()) for key in KEYS] + [("gate_veto", pa.bool_()), ("gate_reason", pa.string())])
STATE = None


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def update_pairs(digest, left, right):
    for a, b in zip(left, right):
        digest.update((a + "\t" + b + "\n").encode())


def capture(seal, root, stats, floor, output):
    output.mkdir(parents=True, exist_ok=True)
    eligible_path = output / "all_country_eligible.parquet"
    if eligible_path.exists():
        raise ValueError("Diagnostic capture already exists; use a fresh directory")
    counts, french_digest, eligible_digest = Counter(), hashlib.sha256(), hashlib.sha256()
    paths = []
    with pq.ParquetWriter(eligible_path, SCORE_SCHEMA, compression="zstd") as writer:
        for entry in seal["chunks"]:
            path = root / entry["parquet"]["path"]
            if sha256(path) != entry["parquet"]["sha256"]:
                raise ValueError("Scored input differs from the verified portable seal")
            chunk_rows = []
            for batch in pq.ParquetFile(path).iter_batches(batch_size=10000, columns=KEYS + ["probability"]):
                counts["scanned_pairs"] += len(batch)
                table = pa.Table.from_batches([batch]).cast(SCORE_SCHEMA)
                eligible = table.filter(pc.greater_equal(table["probability"], floor))
                counts["eligible_pairs"] += len(eligible)
                if not len(eligible):
                    continue
                writer.write_table(eligible)
                values = eligible.to_pydict()
                update_pairs(eligible_digest, values[KEYS[0]], values[KEYS[1]])
                countries = [stats["owner_keys"][key][0] for key in values[KEYS[0]]]
                counts.update(country + "_eligible_pairs" for country in countries)
                french = eligible.filter(pa.array([country == "france" for country in countries]))
                if len(french):
                    chunk_rows.append(french)
                    values = french.to_pydict()
                    update_pairs(french_digest, values[KEYS[0]], values[KEYS[1]])
            if chunk_rows:
                target = output / "french_eligible_chunks" / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                table = pa.concat_tables(chunk_rows)
                pq.write_table(table, target, compression="zstd")
                dump(target.with_suffix(".json"), {"status": "complete", "rows": len(table),
                     "parquet_sha256": sha256(target), "source_parquet_sha256": entry["parquet"]["sha256"]})
                paths.append(target)
    if counts["scanned_pairs"] != seal["pairs"]:
        raise ValueError("Capture does not cover the complete sealed scored universe")
    counts["skipped_below_floor_pairs"] = counts["scanned_pairs"] - counts["eligible_pairs"]
    result = {"counts": dict(counts), "floor": floor, "paths": [str(p) for p in paths],
              "all_country_eligible": {"path": str(eligible_path.resolve()), "sha256": sha256(eligible_path),
                                       "pair_order_sha256": eligible_digest.hexdigest()},
              "french_eligible_pair_order_sha256": french_digest.hexdigest(),
              "full_scored_pair_order_sha256": seal["pair_order_sha256"],
              "interpretation": "Unlabelled diagnostic capture; skipped rows are not observed street agreement"}
    dump(output / "capture.json", result)
    return result


def initialize(stats_path, references, targets, variants, output):
    global STATE
    pa.set_cpu_count(1)
    stats = json.loads(Path(stats_path).read_text())
    STATE = stats, open_raw_lookup(references), open_raw_lookup(targets), variants, Path(output)


def process(path):
    stats, refs, targets, variants, output = STATE
    path = Path(path)
    if sha256(path) != json.loads(path.with_suffix(".json").read_text())["parquet_sha256"]:
        raise ValueError("French eligible checkpoint differs")
    table = pq.read_table(path)
    frame = table.to_pydict()
    reference_records = lookup_raw_records(refs, frame[KEYS[0]])
    target_records = lookup_raw_records(targets, frame[KEYS[1]])
    columns = [{**{key: frame[key] for key in KEYS}, "gate_veto": [], "gate_reason": []} for _ in variants]
    coverage, reasons = Counter(), [Counter() for _ in variants]
    examples = []
    for left, right, probability in zip(frame[KEYS[0]], frame[KEYS[1]], frame["probability"]):
        s1, target = reference_records[left], target_records[right]
        signal = v3.pair_signals(s1, target, stats)
        signal4 = v4.pair_signals(s1, target, stats)
        coverage["eligible_pairs"] += 1
        coverage["name_equal"] += signal["name_match"]
        coverage["name_ambiguous_rivals1"] += signal["name_rivals"] >= 1
        coverage["name_ambiguous_rivals4"] += signal["name_rivals"] >= 4
        coverage["missing_address"] += signal["reference_address_missing"] or signal["target_address_missing"]
        coverage["reliable_street_both"] += signal["street_reliable"]
        coverage["reliable_two_street_tokens"] += signal["street_reliable"] and min(
            signal["reference_street_token_count"], signal["target_street_token_count"]) >= 2
        coverage["qualified_house_agreement"] += signal["house_number_match"] is True
        coverage["distinctive_tokens_both"] += bool(signal4["reference_distinctive_street_tokens"] and
                                                       signal4["target_distinctive_street_tokens"])
        decisions = []
        for i, variant in enumerate(variants):
            decision = (v4.gate_veto(signal4, variant["config"]) if variant["rule"] == "v4-distinctive"
                        else v3.gate_veto(signal, variant["config"]))
            columns[i]["gate_veto"].append(decision["gate_veto"])
            columns[i]["gate_reason"].append(decision["gate_reason"])
            reasons[i][decision["gate_reason"]] += 1
            decisions.append(decision["gate_veto"])
        if any(decisions) and len(examples) < 20:
            examples.append({"source1_entity_id": left, "candidate_entity_id": right, "probability": probability,
                             "variant_veto": decisions, "v3_signals": signal, "v4_signals": signal4})
    for variant, values in zip(variants, columns):
        target = output / variant["name"] / "chunks" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pydict(values, schema=GATE_SCHEMA), target, compression="zstd")
    result = {"input": str(path), "coverage": dict(coverage), "reasons": [dict(v) for v in reasons],
              "examples": examples}
    dump(output / "coverage_chunks" / (path.stem + ".json"), result)
    return result


def main():
    p = argparse.ArgumentParser()
    for key in ("seal", "copy_proof", "decision_config", "stats", "reference_lookup", "target_lookup", "grid", "output"):
        p.add_argument("--" + key.replace("_", "-"), required=True, type=Path)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        raise ValueError("Use the allocated one to eight CPUs")
    pa.set_cpu_count(1)
    seal = json.loads(a.seal.read_text())
    proof = json.loads(a.copy_proof.read_text())
    if proof.get("copy_byte_parity") is not True or proof["portable_manifest_sha256"] != sha256(a.seal):
        raise ValueError("Portable scored seal lacks verified byte parity")
    config = json.loads(a.decision_config.read_text())
    if sha256(a.decision_config) != seal["baseline_frozen"]["sha256"]:
        raise ValueError("Decision config differs from the sealed baseline")
    stats = json.loads(a.stats.read_text())
    if stats["provenance"]["code_sha256"] != sha256(ROOT / "scripts/france_gate.py"):
        raise ValueError("Fitted v2 statistics differ from current immutable signal code")
    for lookup in (a.reference_lookup, a.target_lookup):
        lookup_manifest(lookup)
        if sha256(lookup) != json.loads(lookup.with_name(lookup.name + ".complete.json").read_text())["sqlite_sha256"]:
            raise ValueError("Raw lookup differs from its completed hash")
    variants = [{"name": f"v3_setting_{i}", "rule": "v3-house", "config": setting}
                for i, setting in enumerate(json.loads(a.grid.read_text()))]
    if len(variants) != 6:
        raise ValueError("Expected exactly the existing six settings")
    variants += [{"name": "v4_fixed", "rule": "v4-distinctive", "config": v4.DEFAULT_CONFIG}]
    provenance = {"seal_sha256": sha256(a.seal), "copy_proof_sha256": sha256(a.copy_proof),
                  "decision_config_sha256": sha256(a.decision_config), "stats_sha256": sha256(a.stats),
                  "signal_code_sha256": sha256(ROOT / "scripts/france_gate.py"),
                  "v3_wrapper_sha256": sha256(Path(v3.__file__)), "v4_wrapper_sha256": sha256(Path(v4.__file__)),
                  "diagnostic_code_sha256": sha256(__file__), "labels": "none",
                  "scope": "Complete sealed scored universe; signals only for baseline-eligible France pairs"}
    captured = capture(seal, ROOT, stats, min(config["t_first"], config["t_rest"]), a.output)
    del stats
    with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize,
            initargs=(str(a.stats), str(a.reference_lookup), str(a.target_lookup), variants, str(a.output))) as pool:
        chunks = list(pool.map(process, captured["paths"]))
    coverage, reasons = Counter(), [Counter() for _ in variants]
    for chunk in chunks:
        coverage.update(chunk["coverage"])
        for counter, values in zip(reasons, chunk["reasons"]):
            counter.update(values)
    tables = []
    for i, variant in enumerate(variants):
        directory = a.output / variant["name"]
        path, digest, rows = directory / "gate_table.parquet", hashlib.sha256(), 0
        with pq.ParquetWriter(path, GATE_SCHEMA, compression="zstd") as writer:
            for source in sorted((directory / "chunks").glob("*.parquet")):
                table = pq.read_table(source)
                frame = table.to_pydict()
                update_pairs(digest, frame[KEYS[0]], frame[KEYS[1]])
                writer.write_table(table)
                rows += len(table)
        if digest.hexdigest() != captured["french_eligible_pair_order_sha256"]:
            raise ValueError("Merged French gate table changed pair order")
        item = {**variant, "path": str(path.resolve()), "sha256": sha256(path), "rows": rows,
                "pair_order_sha256": digest.hexdigest(), "reason_counts": dict(reasons[i]),
                "veto_rows": reasons[i]["street_contradiction"] + reasons[i]["distinctive_street_contradiction"],
                "provenance": provenance, "capture": captured}
        dump(directory / "manifest.json", item)
        tables.append(item)
    report = {"status": "complete", "interpretation": "Unlabelled diagnostics, never France accuracy or a links-count target",
              "coverage": dict(coverage), "tables": tables, "capture": captured, "provenance": provenance}
    dump(a.output / "report.json", report)
    print(json.dumps({"status": "complete", "coverage": dict(coverage),
                      "tables": [{k: t[k] for k in ("name", "rows", "veto_rows", "path", "sha256")} for t in tables]}))


if __name__ == "__main__":
    main()
