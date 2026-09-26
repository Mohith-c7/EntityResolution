"""Explain tuning losses without inspecting the fresh audit or test labels."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.evaluation.metrics import entity_f05, score_matches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("models/redesign/wide_tune_cache"))
    p.add_argument("--artifact", type=Path, default=Path("models/features_v3_audit01"))
    p.add_argument("--output-dir", type=Path, default=Path("models/redesign/error_analysis"))
    p.add_argument("--threshold", type=float, default=.75)
    args = p.parse_args()
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    if not manifest["complete"] or manifest["test"]:
        raise ValueError("Only a complete labelled tuning cache is eligible")
    all_refs = json.loads((args.artifact / "sampled_references.json").read_text())["tune"]
    refs = {r["entity_id"]: r for r in all_refs}
    entities = manifest["reference_ids"]
    if not set(entities) <= refs.keys(): raise ValueError("Not a tuning-only cache")
    labels = json.loads((args.artifact / "sampled_truth.json").read_text())
    truth = {e: set(labels[e]) for e in entities}
    pa.set_cpu_count(1)
    frame = pd.concat([pd.read_parquet(args.cache_dir / f) for f in manifest["files"]], ignore_index=True)
    frame["label"] = [c in truth[e] for e, c in frame[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None)]
    frame["selected"] = frame.probability >= args.threshold
    candidates = {e: set() for e in entities}; predictions = {e: set() for e in entities}
    for e, c, selected in frame[["source1_entity_id", "candidate_entity_id", "selected"]].itertuples(index=False, name=None):
        candidates[e].add(c)
        if selected: predictions[e].add(c)
    oracle = {e: truth[e] & candidates[e] for e in entities}
    tp_only = {e: truth[e] & predictions[e] for e in entities}
    recovered = {e: predictions[e] | oracle[e] for e in entities}
    countries = {e: refs[e]["country"] for e in entities}
    actual = score_matches(truth, predictions, countries)
    perfect_retrieved = score_matches(truth, oracle, countries)
    no_false_links = score_matches(truth, tp_only, countries)
    # Telescoping, order-dependent decomposition of the total macro loss.
    deficit = {
        "unretrieved_true_links": 1 - perfect_retrieved["macro_f05"],
        "retrieved_true_links_rejected": perfect_retrieved["macro_f05"] - no_false_links["macro_f05"],
        "false_links_on_singletons": sum(bool(predictions[e]) for e in entities if not truth[e]) / len(entities),
    }
    deficit["false_links_on_non_singletons"] = no_false_links["macro_f05"] - actual["macro_f05"] - deficit["false_links_on_singletons"]
    errors = frame[(frame.label & ~frame.selected) | (~frame.label & frame.selected)].copy()
    errors["error_type"] = np.where(errors.label, "rejected_true_match", "false_merge")
    errors["country"] = errors.source1_entity_id.map(countries)
    flags = {
        "candidate_address_missing": errors.candidate_address_missing == 1,
        "numeric_containment": errors.num_containment > 0,
        "numeric_substitution": errors.num_single_substitution > 0,
        "numeric_contradiction": errors.num_disjoint_contradiction > 0,
        "strong_phonetic_name": errors.phonetic_name_set >= .9,
        "strong_name_weak_address": (errors.name_token_set >= .9) & (errors.address_token_set < .6),
        "strong_address_weak_name": (errors.address_token_set >= .9) & (errors.name_token_set < .6),
        "alias_supported": errors.candidate_alias_confidence >= .98,
        "threshold_neighbourhood": errors.probability.between(args.threshold - .1, args.threshold + .1),
    }
    buckets = {name: errors[mask].groupby(["error_type", "country"]).size().to_dict() for name, mask in flags.items()}
    buckets = {name: {"/".join(k): int(v) for k, v in counts.items()} for name, counts in buckets.items()}
    missing = [{"source1_entity_id": e, "candidate_entity_id": c, "country": countries[e]}
        for e in entities for c in sorted(truth[e] - candidates[e])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors.to_parquet(args.output_dir / "matcher_errors.parquet", index=False)
    pd.DataFrame(missing).to_parquet(args.output_dir / "missed_links.parquet", index=False)
    result = {"split": "tuning_only", "threshold": args.threshold, "actual": actual,
        "retained_candidate_oracle": perfect_retrieved,
        "counterfactual_no_false_links": no_false_links,
        "counterfactual_recover_retrieved_true_links": score_matches(truth, recovered, countries),
        "macro_loss_decomposition": deficit, "decomposition_note": "Ordered counterfactual attribution, not independent achievable gains.",
        "missing_links": len(missing), "matcher_error_counts": errors.error_type.value_counts().to_dict(),
        "overlapping_error_slices": buckets, "fresh_holdout_evaluated": False}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    Path("reports/experiments/redesign/current_error_budget.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__": main()
