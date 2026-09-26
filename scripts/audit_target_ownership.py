"""Count ownership leverage for tuning false merges; do not fit on audit labels."""
import csv
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa


def main():
    os.nice(10); pa.set_cpu_count(1)
    artifact = Path("models/features_v3_audit01")
    refs = json.loads((artifact / "sampled_references.json").read_text())["tune"]
    ids = {r["entity_id"] for r in refs}
    truths = {e: set(v) for e, v in json.loads((artifact / "sampled_truth.json").read_text()).items() if e in ids}
    base = pd.read_parquet(artifact / "predictions_tune.parquet", columns=["source1_entity_id", "candidate_entity_id", "probability"])
    wrong_base = [(e, c) for e, c, p in base.itertuples(index=False, name=None) if p >= .76 and c not in truths[e]]
    current = pd.read_parquet("models/redesign/error_analysis/matcher_errors.parquet")
    wrong_current = list(current.loc[current.error_type == "false_merge", ["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None))
    wanted = {c for _, c in wrong_base + wrong_current}
    owners = {}
    with Path("dataset/train/train_ground_truth.tsv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            for candidate in row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else ():
                if candidate in wanted:
                    if candidate in owners and owners[candidate] != row["source1_entity_id"]:
                        raise ValueError("Multiple labelled owners encountered")
                    owners[candidate] = row["source1_entity_id"]
    results = {}
    for label, errors in (("original_v3", wrong_base), ("postings_v1", wrong_current)):
        counts = Counter()
        for e, c in errors:
            owner = owners.get(c)
            counts["false_positive_links"] += 1
            if owner == e: raise ValueError("A false positive is labelled as true")
            counts["target_has_another_labelled_owner" if owner else "target_has_no_labelled_owner"] += 1
            if owner:
                counts["true_owner_in_tuning_sample" if owner in ids else "true_owner_outside_tuning_sample"] += 1
                counts["false_link_on_singleton" if not truths[e] else "false_link_on_non_singleton"] += 1
        results[label] = dict(counts)
    out = Path("models/redesign/ownership_audit"); out.mkdir(parents=True, exist_ok=True)
    # Future audit samplers already exclude IDs recorded in sampled_references.json.
    (out / "sampled_references.json").write_text(json.dumps({"ownership_inspection": [
        {"entity_id": e} for e in sorted(set(owners.values()))]}, indent=2))
    result = {"scope": "Tuning false-positive target IDs looked up only in provided ground truth",
        "results": results,
        "interpretation": "Having another true owner is potential leverage, not a measured gain. That owner must be retrieved and scored reliably. Tuning samples omit most competitors.",
        "future_audit_exclusions": str(out / "sampled_references.json")}
    Path("reports/experiments/redesign/ownership_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__": main()
