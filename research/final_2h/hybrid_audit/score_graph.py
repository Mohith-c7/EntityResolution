"""Assemble label-free native8/main and native4/extra heldout probabilities."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "research/sprint_6h"),
                str(Path(__file__).resolve().parent)]
import lightgbm as lgb
import numpy as np
import pandas as pd
import neural_adapter as nn
from build_neural_last4_submission import FEATURES, assemble_fine, scatter_corrections
from preflight import POLICY, read, sha

KEYS = ["source1_entity_id", "candidate_entity_id"]


def scores(directory, input_manifest, head_manifest, *, old=False):
    sm, im = read(directory / "manifest.json"), read(input_manifest)
    field = "head_manifest_sha256" if old else "checkpoint_manifest_sha256"
    path = directory / "scores.jsonl"
    if (sm.get("status") != "complete" or sm.get("labels_read") is not False
            or im.get("status") != "complete" or im.get("labels_read") is not False
            or sm[field] != sha(head_manifest) or sm["input_sha256"] != im["input_sha256"]
            or sha(input_manifest.parent / "pairs.jsonl") != im["input_sha256"]
            or sha(path) != sm["scores_sha256"] or sm["rows"] != im["rows"]):
        raise ValueError("Neural score/input/head lineage mismatch")
    return pd.read_json(path, lines=True, dtype={k: str for k in KEYS}), sm


def add_old(frame36, values):
    index = pd.MultiIndex.from_frame(frame36[KEYS]); other = pd.MultiIndex.from_frame(values[KEYS])
    if index.has_duplicates or other.has_duplicates or len(index) != len(other):
        raise ValueError("Duplicate/incomplete old-head extra keys")
    positions = other.get_indexer(index)
    if (positions < 0).any() or not values.neural_probability.between(0, 1).all():
        raise ValueError("Old-head extra pair/probability mismatch")
    out = frame36.copy(); out["neural_probability"] = values.neural_probability.to_numpy()[positions]
    return out


def load_model(directory, head_manifest, old_head_manifest):
    report = read(directory / "report.json"); path = directory / "adapter.txt"
    if (sha(path) != report["model_sha256"] or report["features"] != FEATURES
            or report["fine_head_sha256"] != sha(head_manifest)
            or report["original_neural_head_sha256"] != sha(old_head_manifest)
            or report.get("calibration_logit_shift", 0) != 0
            or report.get("peer_head_sha256") is not None):
        raise ValueError("Native anchored38 adapter/head lineage mismatch")
    model = lgb.Booster(model_file=str(path))
    if model.feature_name() != FEATURES or model.num_feature() != 38:
        raise ValueError("Native adapter feature schema mismatch")
    return model, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "original_manifest", "blue", "main_features", "main_input_manifest",
                 "eight_scores", "eight_model", "eight_head_manifest", "four_model",
                 "four_head_manifest", "old_head_manifest", "extra_features", "extra_old_scores",
                 "extra_four_scores", "reservation", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args(); started = time.monotonic()
    if a.output.exists(): raise FileExistsError(a.output)
    source = read(a.original_manifest)
    original_sha, blue_sha = sha(a.original), sha(a.blue)
    if source["pairs"] != 13240480 or len(source["reference_ids"]) != 331012 or source["pairs_sha256"] != original_sha:
        raise ValueError("Wrong complete heldout original graph")
    route_marker = read(a.extra_features / "route_manifest.json")
    fm = read(a.extra_features / "manifest.json")
    extra_file = a.extra_features / "features.parquet"
    if (fm.get("status") != "complete" or fm.get("labels_read") is not False
            or fm["features_sha256"] != sha(extra_file) or fm["features"] != nn.reverse.FEATURES
            or fm["original_pairs_sha256"] != original_sha or fm["blue_pairs_sha256"] != blue_sha
            or fm["route_manifest_sha256"] != sha(a.extra_features / "route_manifest.json")
            or route_marker["route_sha256"] != sha(a.extra_features / "route.parquet")):
        raise ValueError("Disjoint extra feature/Blue route provenance mismatch")
    main37, main_marker = nn.load_features(a.main_features)
    if len(main37) != 123600: raise ValueError("Wrong heldout original neural route")
    fine8, eight_sm = scores(a.eight_scores, a.main_input_manifest, a.eight_head_manifest)
    main38 = assemble_fine(main37, fine8)
    extra36 = pd.read_parquet(extra_file)
    input_path = a.extra_features / "neural_inputs/manifest.json"
    im = read(input_path)
    if im["feature_manifest_sha256"] != sha(a.extra_features / "manifest.json") or im["feature_file_sha256"] != sha(extra_file):
        raise ValueError("Extra raw input used another feature graph")
    old_values, old_sm = scores(a.extra_old_scores, input_path, a.old_head_manifest, old=True)
    fine4, four_sm = scores(a.extra_four_scores, input_path, a.four_head_manifest)
    extra38 = assemble_fine(add_old(extra36, old_values), fine4)
    model8, report8 = load_model(a.eight_model, a.eight_head_manifest, a.old_head_manifest)
    model4, report4 = load_model(a.four_model, a.four_head_manifest, a.old_head_manifest)
    eight_head, four_head, old_head = [read(x) for x in (a.eight_head_manifest, a.four_head_manifest, a.old_head_manifest)]
    if set(eight_head["training_owners"]) != set(four_head["training_owners"]):
        raise ValueError("NN8 and fixed4 training-owner populations differ")
    reserved = set(read(a.reservation / "entity_ids.json"))
    fitted = set(report8["fit_owners"]) | set(report4["fit_owners"])
    fitted |= set(eight_head["training_owners"]) | set(four_head["training_owners"]) | set(old_head["training_owners"])
    if reserved.intersection(fitted): raise ValueError("Fitted owner entered reserved extension audit")
    joined = pd.concat([main38, extra38], ignore_index=True)
    if joined.duplicated(KEYS).any(): raise ValueError("Main and rescue pair routes overlap")
    margins = []
    for model, frame in ((model8, main38), (model4, extra38)):
        inputs = frame[FEATURES].to_numpy(dtype=np.float32)
        if not np.isfinite(inputs).all(): raise ValueError("Nonfinite native38 inputs")
        margins.append(model.predict(inputs, raw_score=True, num_threads=a.threads))
    original = pd.read_parquet(a.original)
    candidate = scatter_corrections(original, joined, np.concatenate(margins), base_verified=False)
    a.output.mkdir(parents=True)
    candidate.to_parquet(a.output / "pairs.parquet", index=False)
    main37[KEYS].to_parquet(a.output / "main_route.parquet", index=False)
    extra38[KEYS].to_parquet(a.output / "rescue_route.parquet", index=False)
    inputs = {name: sha(getattr(a, name)) for name in ("original", "original_manifest", "blue", "main_input_manifest",
        "eight_head_manifest", "four_head_manifest", "old_head_manifest")}
    marker = {"status": "scored_requires_complete_graph_decision_replay", "labels_read": False,
              "extension_labels_read": False, "rows": len(candidate), "owners": 331012,
              "main_pairs": len(main38), "rescue_pairs": len(extra38), "route_overlap": 0,
              "hybrid_policy": POLICY, "original_p2_anchored_once": True,
              "original_pairs_sha256": original_sha, "blue_pairs_sha256": blue_sha,
              "scored_pairs_sha256": sha(a.output / "pairs.parquet"),
              "main_route_sha256": sha(a.output / "main_route.parquet"),
              "rescue_route_sha256": sha(a.output / "rescue_route.parquet"),
              "native8_sha256": report8["model_sha256"], "native4_sha256": report4["model_sha256"],
              "input_sha256": inputs, "source_sha256": sha(__file__), "seconds": time.monotonic() - started}
    write_path = a.output / "manifest.json"
    write_path.write_text(json.dumps(marker, indent=2) + "\n")
    print(json.dumps(marker, indent=2), flush=True)


if __name__ == "__main__": main()
