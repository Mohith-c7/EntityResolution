"""Streaming pair-keyed gate tables; memory is bounded by fitted state + batches.

One immutable config per run. Original scored chunks are read-only. Every
scored pair survives in the output, with a separate veto/reason; no probability
is replaced. Final merge validates the complete pair order when one is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from france_gate import DEFAULT_CONFIG, SIGNAL_VERSION, _country, _name, gate_veto, pair_signals, reduced_name
from build_raw_lookup import lookup_manifest, lookup_raw_records, open_raw_lookup, sha256

STATE = None
PAIR_GATE = gate_veto
PAIR_SIGNALS = pair_signals
SCHEMA = pa.schema([("source1_entity_id", pa.string()), ("candidate_entity_id", pa.string()),
                    ("gate_veto", pa.bool_()), ("gate_reason", pa.string())])
PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id"]


def initialize(stats_path, lookup_path, config, fingerprint, batch_size, reference_lookup_path=None, gate_rule="v2"):
    global STATE, PAIR_GATE, PAIR_SIGNALS
    PAIR_GATE = gate_veto
    PAIR_SIGNALS = pair_signals
    if gate_rule == "v3-house":
        from house_agreement_gate_v3 import gate_veto as gate_veto_v3
        PAIR_GATE = gate_veto_v3
    elif gate_rule == "v4-distinctive":
        from distinctive_street_gate_v4 import gate_veto as gate_veto_v4, pair_signals as pair_signals_v4
        PAIR_GATE, PAIR_SIGNALS = gate_veto_v4, pair_signals_v4
    elif gate_rule != "v2":
        raise ValueError("Unsupported versioned gate rule")
    pa.set_cpu_count(1)
    stats = json.loads(Path(stats_path).read_text())
    # Delegate config validation to the versioned pure gate, including empty
    # scored chunks where no complete pair signal would otherwise be computed.
    gate_veto({"version": SIGNAL_VERSION, "name_match": False}, config)
    config = {**DEFAULT_CONFIG, **(config or {})}
    if stats.get("version") != SIGNAL_VERSION:
        raise ValueError("Gate table state uses an unsupported signal version")
    if stats.get("provenance", {}).get("code_sha256", fingerprint["gate_code_sha256"]) != fingerprint["gate_code_sha256"]:
        raise ValueError("Fitted state was produced by different gate code")
    conn = open_raw_lookup(lookup_path)
    conn.execute("PRAGMA cache_size=-65536")
    refs = open_raw_lookup(reference_lookup_path) if reference_lookup_path else conn
    if refs is not conn:
        refs.execute("PRAGMA cache_size=-65536")
    STATE = stats, conn, config, fingerprint, batch_size, refs


def prepare_name_views(records):
    """Batch-local normalized names; no global cache of millions of targets."""
    views = {}
    for entity_id, record in records.items():
        name = _name(record)
        views[entity_id] = [_country(record), name, reduced_name(name)]
    return views


def batch_pair_decision(s1, target, records, views, stats, config):
    """Skip expensive addresses only on two exact pure-gate early returns.

    The full module remains the control for eligible pairs. Equivalence is
    checked in fixtures; this does not add a new model or gate threshold.
    """
    reference = views[s1]
    if stats["owner_keys"].get(s1) != reference:
        raise ValueError("Reference must belong unchanged to the fitted Source 1 universe")
    country, rname, rreduced = reference
    _, tname, treduced = views[target]
    name_equal = bool(rname and rname == tname) or bool(rreduced and rreduced == treduced)
    if not name_equal:
        return country, {"gate_veto": False, "gate_reason": "name_not_equal"}
    state = stats["countries"][country]
    def rivals(kind, name, reference_name):
        return state[kind].get(name, 0) - int(name == reference_name) if name else -1
    name_rivals = max(rivals("original_name_counts", rname, rname),
        rivals("original_name_counts", tname, rname), rivals("reduced_name_counts", rreduced, rreduced),
        rivals("reduced_name_counts", treduced, rreduced))
    if name_rivals < config["min_name_rivals"]:
        return country, {"gate_veto": False, "gate_reason": "name_not_ambiguous"}
    signals = PAIR_SIGNALS(records[s1], records[target], stats)
    return country, PAIR_GATE(signals, config)


def _update_pairs(digest, left, right):
    for s1, target in zip(left, right):
        if not isinstance(s1, str) or not s1 or not isinstance(target, str) or not target:
            raise ValueError("Scored pair IDs must be nonempty strings")
        digest.update((s1 + "\t" + target + "\n").encode("utf-8"))


def process_chunk(task):
    input_path, output_dir = map(Path, task)
    stats, conn, config, fingerprint, batch_size, refs = STATE
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / input_path.name
    marker = output.with_suffix(".json")
    source_marker = input_path.with_suffix(".json")
    if not source_marker.is_file():
        raise ValueError("Scored chunk has no complete source checkpoint")
    source_saved = json.loads(source_marker.read_text())
    input_hash = sha256(input_path)
    if source_saved.get("parquet_sha256") != input_hash:
        raise ValueError("Scored chunk differs from its complete source checkpoint")
    identity = {**fingerprint, "input_path": str(input_path.resolve()), "input_parquet_sha256": input_hash,
                "input_checkpoint_sha256": sha256(source_marker)}
    if marker.exists():
        saved = json.loads(marker.read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Gate checkpoint belongs to different inputs/configuration")
        if not output.exists() or saved["gate_parquet_sha256"] != sha256(output):
            raise ValueError("Gate checkpoint output hash mismatch")
        return saved
    if output.exists():
        raise ValueError("Gate parquet exists without its complete checkpoint")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise ValueError("Partial gate parquet exists; inspect before retrying")
    source = pq.ParquetFile(input_path)
    if not set(PAIR_COLUMNS) <= set(source.schema_arrow.names):
        raise ValueError("Scored parquet lacks pair-key columns")
    digest, rows = hashlib.sha256(), 0
    country_counts, reason_counts = Counter(), Counter()
    floor = fingerprint.get("eligibility_floor")
    columns = PAIR_COLUMNS + (["probability"] if floor is not None else [])
    if floor is not None and "probability" not in source.schema_arrow.names:
        raise ValueError("R1 eligibility skip requires immutable original probability column")
    with pq.ParquetWriter(partial, SCHEMA, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pydict()
            left, right = frame[PAIR_COLUMNS[0]], frame[PAIR_COLUMNS[1]]
            _update_pairs(digest, left, right)
            eligible = list(range(len(left)))
            if floor is not None:
                probabilities = frame.pop("probability")
                if any(value is None or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
                    raise ValueError("R1 scored probabilities must be finite and within zero/one")
                eligible = [i for i, probability in enumerate(probabilities) if probability >= floor]
            eligible_left, eligible_right = [left[i] for i in eligible], [right[i] for i in eligible]
            if refs is conn:
                records = lookup_raw_records(conn, [*eligible_left, *eligible_right])
            else:
                records = {**lookup_raw_records(refs, eligible_left), **lookup_raw_records(conn, eligible_right)}
            views = prepare_name_views(records)
            vetoes, reasons = [False] * len(left), ["ineligible_r1_score"] * len(left)
            for i, s1 in enumerate(left):
                owner_key = stats["owner_keys"].get(s1)
                if owner_key is None:
                    raise ValueError("Scored reference is absent from fitted owner universe")
                country_counts[owner_key[0] + "_pairs"] += 1
            reason_counts["ineligible_r1_score"] += len(left) - len(eligible)
            for i in eligible:
                s1, target = left[i], right[i]
                country, decision = batch_pair_decision(s1, target, records, views, stats, config)
                vetoes[i] = decision["gate_veto"]
                reasons[i] = decision["gate_reason"]
                country_counts[country + "_veto"] += decision["gate_veto"]
                reason_counts[decision["gate_reason"]] += 1
            writer.write_table(pa.Table.from_pydict({**frame, "gate_veto": vetoes, "gate_reason": reasons}, schema=SCHEMA))
            rows += len(left)
    if rows != source.metadata.num_rows:
        raise ValueError("Gate table row count differs from scored input")
    partial.rename(output)
    saved = {**identity, "status": "complete", "rows": rows, "pair_order_sha256": digest.hexdigest(),
             "gate_parquet_sha256": sha256(output), "country_counts": dict(country_counts),
             "reason_counts": dict(reason_counts), "output": str(output.resolve())}
    temporary = marker.with_name(marker.name + ".partial")
    temporary.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    temporary.rename(marker)
    return saved


def merge_gate_tables(chunks, output, expected_pair_order_sha256=None, batch_size=10000):
    output = Path(output)
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise ValueError("Merged gate table already exists; use a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    pair_digest, rows, countries, reasons = hashlib.sha256(), 0, Counter(), Counter()
    common_keys = ("stats_sha256", "config_sha256", "gate_code_sha256", "builder_code_sha256",
                   "normalization_sha256", "raw_lookup_sha256", "raw_inputs", "reference_lookup_sha256",
                   "reference_inputs", "execution_mode", "eligibility_floor", "eligibility_proof",
                   "gate_rule", "gate_wrapper_code_sha256", "parent_gate_wrapper_sha256", "gate_version")
    fingerprint = {key: chunks[0][key] for key in common_keys if key in chunks[0]} if chunks else {}
    with pq.ParquetWriter(partial, SCHEMA, compression="zstd") as writer:
        for checkpoint in chunks:
            if any(checkpoint.get(key) != value for key, value in fingerprint.items()):
                raise ValueError("Gate merge chunks use different input/configuration fingerprints")
            path = Path(checkpoint["output"])
            if checkpoint["status"] != "complete" or sha256(path) != checkpoint["gate_parquet_sha256"]:
                raise ValueError("Gate merge chunk is incomplete or its hash differs")
            chunk_digest, chunk_rows = hashlib.sha256(), 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                frame = batch.to_pydict()
                _update_pairs(chunk_digest, frame[PAIR_COLUMNS[0]], frame[PAIR_COLUMNS[1]])
                _update_pairs(pair_digest, frame[PAIR_COLUMNS[0]], frame[PAIR_COLUMNS[1]])
                writer.write_batch(batch)
                chunk_rows += len(batch)
            if chunk_rows != checkpoint["rows"] or chunk_digest.hexdigest() != checkpoint["pair_order_sha256"]:
                raise ValueError("Gate merge pair order/count differs from chunk checkpoint")
            rows += chunk_rows
            countries.update(checkpoint["country_counts"])
            reasons.update(checkpoint["reason_counts"])
    pair_hash = pair_digest.hexdigest()
    if expected_pair_order_sha256 and expected_pair_order_sha256 != pair_hash:
        raise ValueError("Merged gate pair order differs from verified score reuse manifest")
    partial.rename(output)
    result = {"status": "complete", "path": str(output.resolve()), "rows": rows,
        "pair_order_sha256": pair_hash, "sha256": sha256(output), "chunks": len(chunks),
        "country_counts": dict(countries), "reason_counts": dict(reasons), "gate_fingerprint": fingerprint}
    output.with_suffix(output.suffix + ".complete.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-chunks", required=True, type=Path)
    parser.add_argument("--raw-lookup", required=True, type=Path)
    parser.add_argument("--reference-lookup", type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gate-rule", choices=("v2", "v3-house", "v4-distinctive"), default="v2")
    parser.add_argument("--execution-mode", choices=("all-pairs", "R1"), default="all-pairs")
    parser.add_argument("--decision-config", type=Path,
                        help="Exact frozen R1 decision JSON with t_first/t_rest; do not reuse on boosted R2")
    parser.add_argument("--output-chunks", required=True, type=Path)
    parser.add_argument("--merge-output", type=Path)
    parser.add_argument("--expected-pair-order-sha256")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.batch_size < 1:
        raise ValueError("Use one to eight allocated workers and positive batch size")
    config = json.loads(args.config.read_text())
    if bool(args.decision_config) != (args.execution_mode == "R1"):
        raise ValueError("R1 skipping requires exact frozen decision config; all-pairs mode accepts none")
    raw_manifest = lookup_manifest(args.raw_lookup)
    raw_sha = sha256(args.raw_lookup)
    complete = json.loads(args.raw_lookup.with_name(args.raw_lookup.name + ".complete.json").read_text())
    if raw_sha != complete["sqlite_sha256"]:
        raise ValueError("Raw record lookup differs from its complete marker hash")
    fingerprint = {"stats_sha256": sha256(args.stats), "config_sha256": sha256(args.config),
        "gate_code_sha256": sha256(ROOT / "scripts/france_gate.py"), "builder_code_sha256": sha256(__file__),
        "normalization_sha256": sha256(ROOT / "code/business_entity_resolution/src/preprocessing/normalize.py"),
        "raw_lookup_sha256": raw_sha, "raw_inputs": raw_manifest["inputs"], "execution_mode": args.execution_mode,
        "gate_rule": args.gate_rule}
    if args.gate_rule == "v3-house":
        from house_agreement_gate_v3 import GATE_VERSION
        fingerprint.update(gate_wrapper_code_sha256=sha256(Path(__file__).with_name("house_agreement_gate_v3.py")),
                           gate_version=GATE_VERSION)
    elif args.gate_rule == "v4-distinctive":
        from distinctive_street_gate_v4 import GATE_VERSION
        fingerprint.update(gate_wrapper_code_sha256=sha256(Path(__file__).with_name("distinctive_street_gate_v4.py")),
                           parent_gate_wrapper_sha256=sha256(Path(__file__).with_name("house_agreement_gate_v3.py")),
                           gate_version=GATE_VERSION)
    if args.execution_mode == "R1":
        decision = json.loads(args.decision_config.read_text())
        thresholds = [decision["t_first"], decision["t_rest"]]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
            raise ValueError("Frozen decision thresholds must be finite probabilities")
        fingerprint["eligibility_floor"] = min(thresholds)
        fingerprint["eligibility_proof"] = {"decision_config_sha256": sha256(args.decision_config),
            "t_first": thresholds[0], "t_rest": thresholds[1], "score_column": "probability",
            "scope": "R1 only, immutable original scores; neither ordinary nor rank-one can accept below floor",
            "R2_reuse": "forbidden unless recomputed after updated scores or including every globally routed pair"}
    if args.reference_lookup:
        reference_manifest = lookup_manifest(args.reference_lookup)
        reference_sha = sha256(args.reference_lookup)
        complete = json.loads(args.reference_lookup.with_name(args.reference_lookup.name + ".complete.json").read_text())
        if reference_sha != complete["sqlite_sha256"]:
            raise ValueError("Reference lookup differs from its complete marker hash")
        fingerprint.update(reference_lookup_sha256=reference_sha, reference_inputs=reference_manifest["inputs"])
    paths = sorted(path for path in args.input_chunks.glob("*.parquet")
                   if path.stem.isdecimal() and path.with_suffix(".json").exists())
    if not paths:
        raise ValueError("No scored parquet chunks")
    tasks = [(str(path), str(args.output_chunks)) for path in paths]
    initargs = (str(args.stats), str(args.raw_lookup), config, fingerprint, args.batch_size,
                str(args.reference_lookup) if args.reference_lookup else None, args.gate_rule)
    if args.workers == 1:
        initialize(*initargs)
        checkpoints = [process_chunk(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=initargs) as pool:
            checkpoints = list(pool.map(process_chunk, tasks))
    if args.merge_output:
        result = merge_gate_tables(checkpoints, args.merge_output, args.expected_pair_order_sha256, args.batch_size)
    else:
        result = {"status": "complete", "chunks": len(checkpoints), "rows": sum(c["rows"] for c in checkpoints)}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
