"""Bounded label-free structural coverage scan; no truth files are opened."""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from france_gate import SIGNAL_VERSION, address_view, fit_signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-country", type=int, default=3000)
    parser.add_argument("--max-read", type=int, default=100000)
    parser.add_argument("--stats-output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    groups = defaultdict(list)
    seen = 0
    with args.source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            seen += 1
            country = row.get("country", "")
            if args.per_country == 0 or len(groups[country]) < args.per_country:
                groups[country].append(row)
            if args.max_read and seen >= args.max_read:
                break
    fit_started = time.monotonic()
    stats = fit_signals(groups)
    report = {"version": SIGNAL_VERSION, "source": str(args.source.resolve()),
              "selection": ("complete supplied Source 1 universe" if not args.max_read and not args.per_country
                  else "first capped rows per country within bounded prefix; not representative random sample"),
              "rows_read": seen, "records_selected": sum(map(len, groups.values())),
              "truth_opened": False, "fit_seconds": time.monotonic() - fit_started, "countries": {}}
    for raw_country, records in groups.items():
        country = stats["owner_keys"][records[0]["entity_id"]][0]
        state = stats["countries"][country]
        coverage = Counter()
        for row in records:
            view = address_view(row, state["street_templates"])
            coverage["address_nonempty"] += bool(view["folded"])
            coverage["street_reliable"] += view["street_reliable"]
            coverage["street_reliable_two_tokens"] += view["street_reliable"] and len(view["street_tokens"]) >= 2
            coverage["house_qualified"] += view["house_number"] is not None
        report["countries"][country] = {"records": len(records),
            "name_original_nonempty": sum(state["original_name_counts"].values()),
            "name_reduced_empty": state["empty_reduced"],
            "original_ambiguous_owners": sum(n for n in state["original_name_counts"].values() if n >= 3),
            "reduced_ambiguous_owners": sum(n for n in state["reduced_name_counts"].values() if n >= 3),
            **coverage,
            "street_templates": {key: state["street_templates"][key] for key in ("prefix", "suffix")},
            "learned_edge_tokens": state["learned_edge_tokens"]}
    report["elapsed_seconds"] = time.monotonic() - started
    report["code_sha256"] = hashlib.sha256((ROOT / "scripts/france_gate.py").read_bytes()).hexdigest()
    hasher = hashlib.sha256()
    with args.source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    report["source_sha256"] = hasher.hexdigest()
    if args.stats_output:
        stats["provenance"] = {key: report[key] for key in ("source", "source_sha256", "code_sha256", "selection")}
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(json.dumps(stats, sort_keys=True) + "\n")
        report["stats_path"] = str(args.stats_output)
        report["stats_sha256"] = hashlib.sha256(args.stats_output.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
