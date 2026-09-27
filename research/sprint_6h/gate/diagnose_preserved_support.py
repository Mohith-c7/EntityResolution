"""Read-only parser-loss diagnosis on exposed development records.

No gate implementation changes. Inspect whether the fixed rejected v4 street
contradictions ignore exact/near support in the other preserved full address.
Thresholds and DF definitions remain exactly those of the rejected ablation.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import distinctive_street_gate_v4 as v4
from build_raw_lookup import sha256


def preserved_support(signals):
    left = [[token, other] for token in signals["reference_distinctive_street_tokens"]
            for other in signals["target_address"]["tokens"] if v4.near_match(token, other)]
    right = [[token, other] for token in signals["target_distinctive_street_tokens"]
             for other in signals["reference_address"]["tokens"] if v4.near_match(token, other)]
    return left, right


def main():
    p = argparse.ArgumentParser()
    for key in ("references", "truth", "targets", "stats", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    refs = {row["entity_id"]: row for row in json.loads(a.references.read_text())}
    truth = json.loads(a.truth.read_text())
    if set(truth) != set(refs) or len(refs) != 50000:
        raise ValueError("Require exactly the already exposed development50k universe")
    owners = {target: owner for owner, values in truth.items() for target in values}
    if len(owners) != sum(map(len, truth.values())):
        raise ValueError("Known development targets have conflicting owners")
    targets = {row["entity_id"]: row for row in json.loads(a.targets.read_text())}
    if set(targets) != set(owners):
        raise ValueError("Raw target export differs from the exposed known-target universe")
    stats = json.loads(a.stats.read_text())
    neighbors = defaultdict(set)
    for entity, record in refs.items():
        country = stats["owner_keys"][entity][0]
        name = v4._name(record)
        for key in (name, v4.reduced_name(name)):
            if key:
                neighbors[country, key].add(entity)
    pairs = {(owner, target) for target, owner in owners.items()}
    for target, record in targets.items():
        country = stats["owner_keys"][owners[target]][0]
        name = v4._name(record)
        for key in (name, v4.reduced_name(name)):
            if key:
                pairs.update((owner, target) for owner in neighbors[country, key])
    counts, true_harms, remaining_negative_examples = defaultdict(Counter), [], []
    for owner, target in sorted(pairs):
        label = "positive" if owners[target] == owner else "negative"
        signals = v4.pair_signals(refs[owner], targets[target], stats)
        country = signals["country"]
        counts[country][label + "_pairs"] += 1
        if not v4.gate_veto(signals)["gate_veto"]:
            continue
        counts[country][label + "_v4_contradictions"] += 1
        left, right = preserved_support(signals)
        remaining = not left and not right
        counts[country][label + ("_remaining_contradictions" if remaining else "_preserved_support")] += 1
        item = {"source1_entity_id": owner, "candidate_entity_id": target, "known_owner": owners[target],
                "reference_distinctive_to_full_target_support": left,
                "target_distinctive_to_full_reference_support": right,
                "still_contradicts": remaining, "signals": signals}
        if label == "positive":
            true_harms.append(item)
        elif remaining and len(remaining_negative_examples) < 20:
            remaining_negative_examples.append(item)
    report = {"status": "read_only_diagnostic_complete", "interpretation": "Parser-loss diagnosis only; no gate or full-test variant changed",
              "definitions": {"df_fraction_max": v4.DF_FRACTION_MAX, "min_length": v4.MIN_LENGTH,
                              "near_match_normalized_edit_max": v4.NEAR_MATCH_MAX,
                              "near_match_min_length": v4.NEAR_MATCH_MIN_LENGTH},
              "counts": {country: dict(values) for country, values in counts.items()},
              "all_existing_true_harms": true_harms, "remaining_negative_examples": remaining_negative_examples,
              "pair_count": len(pairs), "known_true_targets": len(owners),
              "provenance": {"references_sha256": sha256(a.references), "truth_sha256": sha256(a.truth),
                             "targets_sha256": sha256(a.targets), "stats_sha256": sha256(a.stats),
                             "fixed_v4_code_sha256": sha256(v4.__file__), "diagnostic_code_sha256": sha256(__file__)}}
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"counts": report["counts"], "true_harms": len(true_harms)}))


if __name__ == "__main__":
    main()
