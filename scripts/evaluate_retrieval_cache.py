"""Compare two retrieval caches using one matcher and the same tuning entities."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

from train_scale_model import score, predict, write_json, digest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.features.registry import FEATURE_NAMES_V3


def load_cache(path, allow_recovery=False):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["test"]: raise ValueError("Only labelled tuning caches may be evaluated")
    if not manifest["complete"] and not allow_recovery: raise ValueError("Incomplete cache")
    refs = manifest["reference_ids"]
    if len(refs) != len(set(refs)): raise ValueError("Duplicate reference IDs")
    files = []; frames = []
    # benchmark_postings writes deterministic 25-reference chunks, even for empty candidates.
    for offset in range(0, len(refs), 25):
        ids = refs[offset:offset+25]
        name = hashlib.sha256("\n".join(ids).encode()).hexdigest() + ".parquet"
        f = pd.read_parquet(path / name)
        if not set(f.source1_entity_id) <= set(ids): raise ValueError("Chunk references mismatch")
        if not np.isfinite(f[list(FEATURE_NAMES_V3) + ["probability"]].to_numpy()).all(): raise ValueError("Invalid features or predictions")
        files.append(name); frames.append(f)
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["source1_entity_id", "candidate_entity_id"]).any(): raise ValueError("Duplicate pair")
    if not manifest["complete"]:
        # This run failed only in report generation. Require complete fixed-budget coverage to recover it.
        sizes = frame.groupby("source1_entity_id").size()
        if set(sizes.index) != set(refs) or not (sizes == 2 * manifest["top_k"]).all():
            raise ValueError("Cannot establish complete fixed-budget scoring")
        manifest.update(complete=True, rows_scored=len(frame), files=files,
            recovery="Verified all expected chunks, IDs, finite features, uniqueness and fixed-budget coverage after report-generation failure.")
        write_json(path / "manifest.json", manifest)
    if len(frame) != manifest["rows_scored"]: raise ValueError("Unexpected row count")
    return manifest, frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-cache", type=Path, required=True)
    p.add_argument("--baseline-cache", type=Path, default=Path("models/redesign/wide_tune_cache"))
    p.add_argument("--model", type=Path, default=Path("models/scale_v1_run/model_300k/model.txt"))
    p.add_argument("--references", type=Path, default=Path("models/features_v3_audit01"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--recover-complete-chunks", action="store_true")
    args = p.parse_args(); pa.set_cpu_count(1)
    current_manifest, current = load_cache(args.candidate_cache, args.recover_complete_chunks)
    old_manifest, old = load_cache(args.baseline_cache)
    refs = {r["entity_id"]: r for r in json.loads((args.references / "sampled_references.json").read_text())["tune"]}
    ids = current_manifest["reference_ids"]
    if set(ids) != set(old_manifest["reference_ids"]) or not set(ids) <= refs.keys(): raise ValueError("Not the same tuning references")
    truth = json.loads((args.references / "sampled_truth.json").read_text())
    truth = {e: set(truth[e]) for e in ids}
    positions = {e: i for i, e in enumerate(ids)}
    expected = np.array([len(truth[e]) for e in ids])
    countries = np.array([refs[e]["country"] for e in ids])
    model = lgb.Booster(model_file=str(args.model))
    if model.feature_name() != list(FEATURE_NAMES_V3): raise ValueError("Feature schema mismatch")
    model_sha = digest(args.model)
    data = {}
    candidates = {}
    for tag, frame, manifest in (("baseline", old, old_manifest), ("candidate", current, current_manifest)):
        probability = frame.probability.to_numpy() if manifest["model_sha256"] == model_sha else predict(model, frame[list(FEATURE_NAMES_V3)].to_numpy(dtype=np.float32), 4)
        group = frame.source1_entity_id.map(positions).to_numpy(dtype=np.int32)
        label = np.array([c in truth[e] for e,c in frame[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None)], dtype=np.uint8)
        fixed, values = score(probability, .75, label, group, expected, countries)
        oracle = score(label, .5, label, group, expected, countries)[0]
        trials = [score(probability, round(float(t), 6), label, group, expected, countries)[0]
                  for t in np.r_[np.arange(.30, .951, .01), .96, .97, .98, .99, .995]]
        data[tag] = {"at_frozen_threshold": fixed, "oracle": oracle, "threshold_sweep": trials}
        candidates[tag] = {(e,c) for e,c,positive in zip(frame.source1_entity_id,frame.candidate_entity_id,label) if positive}
    base = data["baseline"]["at_frozen_threshold"]
    eligible = [r for r in data["candidate"]["threshold_sweep"] if r["micro_precision"] >= base["micro_precision"] - .002
                and r["singleton_false_positives"] <= base["singleton_false_positives"]]
    chosen = max(eligible, key=lambda r: r["macro_f05"]) if eligible else None
    report = {"status": "tuning_only_not_promoted", "model_sha256": model_sha, "entities": len(ids),
        "baseline": data["baseline"], "candidate": data["candidate"], "selection": chosen,
        "macro_gain_at_frozen_threshold": data["candidate"]["at_frozen_threshold"]["macro_f05"] - base["macro_f05"],
        "new_true_candidates": len(candidates["candidate"] - candidates["baseline"]),
        "lost_true_candidates": len(candidates["baseline"] - candidates["candidate"]),
        "fresh_audit_evaluated": False, "submission_generated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True); write_json(args.output, report)
    print(json.dumps({"baseline": base, "candidate": data["candidate"]["at_frozen_threshold"],
        "candidate_oracle": data["candidate"]["oracle"], "selection": chosen,
        "new_true_candidates": report["new_true_candidates"], "lost_true_candidates": report["lost_true_candidates"]}), flush=True)


if __name__ == "__main__": main()
