"""Verify cached feature parity and alternate retrieval implementation outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("models/redesign/wide_tune_cache"))
    p.add_argument("--artifact", type=Path, default=Path("models/features_v3_audit01"))
    p.add_argument("--reference-pairs", type=Path, default=Path("models/redesign/postings_wide_k20_10k_pairs.json"))
    p.add_argument("--native-pairs", type=Path, default=Path("models/redesign/postings_hash_wide_10k_pairs.json"))
    p.add_argument("--output", type=Path, default=Path("reports/experiments/redesign/feature_compatibility.json"))
    args = p.parse_args(); pa.set_cpu_count(1)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if not manifest["complete"] or manifest["test"]: raise ValueError("Use a complete tuning cache")
    current = pd.concat([pd.read_parquet(args.cache / name) for name in manifest["files"]], ignore_index=True)
    previous = pd.read_parquet(args.artifact / "predictions_tune.parquet")
    keys = ["source1_entity_id", "candidate_entity_id"]
    columns = [c for c in previous.columns if c in current.columns and c not in keys + [
        "probability", "blocking_paths", "rank_within_source", "candidate_source", "blocking_score"]]
    common = current.merge(previous[keys + columns], on=keys, suffixes=("_new", "_old"), validate="1:1")
    differences = sum(int(np.count_nonzero(common[c + "_new"].to_numpy() != common[c + "_old"].to_numpy())) for c in columns)
    identical = json.loads(args.reference_pairs.read_text()) == json.loads(args.native_pairs.read_text())
    if differences or len(columns) != 64 or not identical or not len(common):
        raise ValueError("Feature/implementation parity failed")
    report = {"common_pairs": len(common), "unchanged_feature_columns": len(columns), "different_values": differences,
        "expected_changed_feature": "blocking_score (new retrieval provenance)", "native_hash_matches_blob_predictions": identical}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__": main()
