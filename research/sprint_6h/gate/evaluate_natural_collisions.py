"""Natural owner-labelled diagnostic restricted to already-exposed references.

This is a text-neighborhood stress cohort, not frozen matcher predictions and
not an entity macro score. Truth is used only to label true target links and
different-owner target negatives, never to fit name/address transformations.
"""
import argparse
import hashlib
import importlib.util
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from france_gate import fit_signals, gate_veto, pair_signals, reduced_name, _name
import france_gate as gate_api


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def connection(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True, type=Path)
    owner_input = parser.add_mutually_exclusive_group(required=True)
    owner_input.add_argument("--owners", type=Path)
    owner_input.add_argument("--truth", type=Path, help="Already-exposed dev-only truth export")
    parser.add_argument("--s2-index", type=Path)
    parser.add_argument("--s3-index", type=Path)
    parser.add_argument("--target-records", type=Path)
    parser.add_argument("--signal-module", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--configs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    api = gate_api
    if args.signal_module:
        spec = importlib.util.spec_from_file_location("gate_diagnostic_version", args.signal_module)
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
    # A decision-only wrapper inherits unchanged frozen v2 text views without
    # having to export private normalization helpers as public gate API.
    name_view = getattr(api, "_name", _name)
    reduced_view = getattr(api, "reduced_name", reduced_name)
    references = json.loads(args.references.read_text())
    refs = {row["entity_id"]: row for row in references}
    if len(refs) != len(references):
        raise ValueError("Duplicate reference IDs")
    configurations = json.loads(args.configs.read_text())
    if not isinstance(configurations, list) or not 1 <= len(configurations) <= 6:
        raise ValueError("Predeclare one to six gate configurations")
    groups = defaultdict(list)
    for row in references:
        groups[row["country"]].append(row)
    stats = json.loads(args.stats.read_text()) if args.stats else api.fit_signals(groups)
    owners = {}
    if args.truth:
        truth = json.loads(args.truth.read_text())
        if set(truth) != set(refs):
            raise ValueError("Dev-only truth IDs do not exactly match the supplied exposed references")
        for owner, target_ids in truth.items():
            for target_id in target_ids:
                if not target_id.startswith(("S2", "S3")) or target_id in owners:
                    raise ValueError("Truth target IDs must be unique supplied S2/S3 rows")
                owners[target_id] = owner
    else:
        conn = connection(args.owners)
        ids = sorted(refs)
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            query = f"SELECT id,owner FROM owners WHERE owner IN ({','.join('?' for _ in part)})"
            for target_id, owner in conn.execute(query, part):
                if target_id.startswith(("S2", "S3")):
                    owners[target_id] = owner
    targets = {}
    if args.target_records:
        raw_targets = {row["entity_id"]: row for row in json.loads(args.target_records.read_text())}
        if set(raw_targets) != set(owners):
            raise ValueError("Target-record export does not exactly match selected owners' targets")
        targets = raw_targets
    else:
        if not args.s2_index or not args.s3_index:
            raise ValueError("Supply raw target-record export or both normalized source indexes")
        for source, path in (("S2", args.s2_index), ("S3", args.s3_index)):
            conn = connection(path)
            ids = sorted(target_id for target_id in owners if target_id.startswith(source))
            for i in range(0, len(ids), 500):
                part = ids[i:i + 500]
                query = f"SELECT id,name,address,country FROM records WHERE id IN ({','.join('?' for _ in part)})"
                for target_id, name, address, country in conn.execute(query, part):
                    targets[target_id] = {"entity_id": target_id, "name": name, "address": address, "country": country}
        if len(targets) != len(owners):
            raise ValueError("Missing selected owners' target records")
    neighborhoods = defaultdict(set)
    for reference_id, row in refs.items():
        country = stats["owner_keys"][reference_id][0]
        for name in (name_view(row), reduced_view(name_view(row))):
            if name:
                neighborhoods[(country, name)].add(reference_id)
    pairs = set()
    for target_id, target in targets.items():
        owner = owners[target_id]
        pairs.add((owner, target_id))
        country = stats["owner_keys"][owner][0]
        for name in (name_view(target), reduced_view(name_view(target))):
            if name:
                pairs.update((reference_id, target_id) for reference_id in neighborhoods[(country, name)])
    counters = [defaultdict(Counter) for _ in configurations]
    route = defaultdict(Counter)
    samples = [[] for _ in configurations]
    all_true_vetoes = [[] for _ in configurations]
    noisy_retained = {"positive": [], "negative": []}
    for reference_id, target_id in sorted(pairs):
        label = "positive" if owners[target_id] == reference_id else "negative"
        signals = api.pair_signals(refs[reference_id], targets[target_id], stats)
        country = signals["country"]
        route[country][label] += 1
        route[country][label + "_name_equal"] += signals["name_match"]
        route[country][label + "_reliable_street"] += signals["street_reliable"]
        route[country][label + "_missing_address"] += signals["reference_address_missing"] or signals["target_address_missing"]
        route[country][label + "_house_conflict"] += signals["house_number_conflict"]
        route[country][label + "_zero_street_overlap"] += signals["street_overlap"] == 0.0
        for i, config in enumerate(configurations):
            decision = api.gate_veto(signals, config)
            counters[i][country][label + ("_veto" if decision["veto"] else "_retain")] += 1
            counters[i][country][label + "_reason_" + decision["reason"]] += 1
            if decision["veto"] and label == "positive":
                all_true_vetoes[i].append({"source1_entity_id": reference_id, "candidate_entity_id": target_id,
                                          "known_owner": owners[target_id], "signals": signals})
            if decision["veto"] and len(samples[i]) < 30:
                samples[i].append({"source1_entity_id": reference_id, "candidate_entity_id": target_id,
                    "known_owner": owners[target_id], "label": label, "signals": signals})
        if len(noisy_retained[label]) < 15 and not api.gate_veto(signals, configurations[0])["veto"]:
            interesting = (signals["house_number_conflict"] or signals["street_overlap"] == 0.0) if label == "positive" else True
            if interesting:
                noisy_retained[label].append({"source1_entity_id": reference_id, "candidate_entity_id": target_id,
                    "known_owner": owners[target_id], "label": label,
                    "gate_reason": api.gate_veto(signals, configurations[0])["reason"], "signals": signals})
    report = {"cohort": "already-exposed references; all known true targets plus exact raw/reduced-name cross-owner neighbors",
        "interpretation": "pair stress diagnostic, not matcher or entity-level final-score evaluation",
        "reference_count": len(refs), "known_true_target_count": len(owners), "pair_count": len(pairs),
        "stats_owner_universe_count": len(stats["owner_keys"]),
        "target_text_source": (str(args.target_records) if args.target_records else
                               "existing normalized supplied-record indexes; punctuation already lost"),
        "truth_scope": ("Already-exposed dev-only truth export; exact reference-ID match" if args.truth else
            "owners WHERE owner IN explicitly selected already-exposed reference IDs only"),
        "route_counts": {country: dict(values) for country, values in route.items()},
        "retained_noisy_examples": noisy_retained,
        "variants": [{"config": config, "counts": {country: dict(values) for country, values in counters[i].items()},
                      "veto_examples": samples[i], "all_true_vetoes": all_true_vetoes[i]}
                     for i, config in enumerate(configurations)],
        "references_sha256": digest(args.references), "config_sha256": digest(args.configs),
        "truth_sha256": digest(args.truth) if args.truth else None,
        "code_sha256": digest(args.signal_module or ROOT / "scripts/france_gate.py"),
        "signal_code_sha256": digest(ROOT / "scripts/france_gate.py"),
        "stats_sha256": digest(args.stats) if args.stats else None,
        "diagnostic_code_sha256": digest(__file__),
        "gate_version": getattr(api, "GATE_VERSION", None),
        "signal_version": stats["version"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"variants", "retained_noisy_examples"}}, sort_keys=True))
    print(json.dumps([{key: value for key, value in variant.items() if key != "veto_examples"}
                      for variant in report["variants"]], sort_keys=True))


if __name__ == "__main__":
    main()
