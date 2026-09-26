"""Select an experimental threshold from cached tuning predictions only."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("models/features_v3_audit01"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    if not manifest["complete"] or manifest["test"]:
        raise ValueError("Threshold selection requires a complete labelled tuning cache")
    refs = json.loads((args.artifact / "sampled_references.json").read_text())["tune"]
    allowed = {r["entity_id"] for r in refs}
    entities = manifest["reference_ids"]
    if not set(entities) <= allowed or len(set(entities)) != len(entities):
        raise ValueError("Cached entities are not unique tuning references")
    labels = json.loads((args.artifact / "sampled_truth.json").read_text())
    truth = {eid: set(labels[eid]) for eid in entities}
    positions = {eid: i for i, eid in enumerate(entities)}
    pa.set_cpu_count(1)
    frame = pd.concat([pd.read_parquet(args.cache_dir / f, columns=[
        "source1_entity_id", "candidate_entity_id", "probability"]) for f in manifest["files"]], ignore_index=True)
    if len(frame) != manifest["rows_scored"] or frame.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise ValueError("Cached pair count/uniqueness mismatch")
    group = np.fromiter((positions[e] for e in frame.source1_entity_id), dtype=np.int32)
    positive = np.fromiter((c in truth[e] for e, c in frame[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None)), dtype=bool)
    expected = np.array([len(truth[e]) for e in entities])
    probabilities = frame.probability.to_numpy()
    if not np.isfinite(probabilities).all(): raise ValueError("Nonfinite predictions")
    baseline = json.loads((args.artifact / "report.json").read_text())["tune"]
    rows = []
    for threshold in np.r_[np.arange(.30, .951, .01), .96, .97, .98, .99, .995, .999]:
        selected = probabilities >= threshold
        predicted = np.bincount(group[selected], minlength=len(entities))
        tp = np.bincount(group[selected & positive], minlength=len(entities))
        scores = np.zeros(len(entities), dtype=np.float64)
        np.divide(1.25 * tp, .25 * expected + predicted, out=scores, where=(.25 * expected + predicted) > 0)
        scores[(expected == 0) & (predicted == 0)] = 1.
        rows.append({"threshold": round(float(threshold), 6), "macro_f05": float(scores.mean()),
            "micro_precision": float(tp.sum() / max(predicted.sum(), 1)),
            "micro_recall": float(tp.sum() / max(expected.sum(), 1)),
            "singleton_false_positives": int(((expected == 0) & (predicted > 0)).sum())})
    # Exact same reference set is necessary for the saved-baseline constraints.
    if set(entities) != allowed:
        raise ValueError("Run on the complete tuning set to compare with the saved tuning baseline")
    eligible = [row for row in rows if row["micro_precision"] >= baseline["micro_precision"] - .002
                and row["singleton_false_positives"] <= baseline["singleton_false_positives"]]
    chosen = max(eligible, key=lambda r: (r["macro_f05"], r["micro_precision"], r["threshold"])) if eligible else None
    result = {"status": "tuning_selection_requires_fresh_audit", "selection": chosen,
        "baseline_tuning": baseline, "sweep": rows, "cache": str(args.cache_dir)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sweep"}))


if __name__ == "__main__": main()
