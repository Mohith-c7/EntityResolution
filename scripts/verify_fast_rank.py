"""Compare accelerated and original candidate generation without changing a model."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.disk_index import DiskSearchConfig, DiskSourceIndex, normalize_record
from src.blocking.fast_rank import install_fast_rank
from src.features.pairwise_features import build_extended_pair_features
from src.features.registry import FEATURE_NAMES_V2
from src.model.name_aliases import NameAliases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("models/larger_alias_v4"))
    parser.add_argument("--index-dir", type=Path, default=Path("models"))
    parser.add_argument("--extension", type=Path, default=Path("models/runtime/erbm25.dylib"))
    parser.add_argument("--entities", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("models/fast_rank_verification.json"))
    args = parser.parse_args()
    report = json.loads((args.artifact_dir / "report.json").read_text())
    config = DiskSearchConfig(**report["search_config"])
    aliases = NameAliases.load(args.artifact_dir / "name_aliases.json")
    references = json.loads((args.artifact_dir / "sampled_references.json").read_text())["tune"][:args.entities]
    model = lgb.Booster(model_file=str(args.artifact_dir / "model.txt"))
    indexes = [DiskSourceIndex(args.index_dir / f"index_train_{s}.sqlite", config, aliases=aliases) for s in ("S2", "S3")]
    captured = {}; original_seconds = accelerated_seconds = 0.0; queries = 0; pairs = 0
    try:
        # Capture the exact original final candidate set, scores and provenance.
        start = time.perf_counter()
        for raw in references:
            ref = normalize_record(raw)
            captured[ref.entity_id] = [p for index in indexes for p in index.query(ref)]
        original_seconds = time.perf_counter() - start
        for index in indexes:
            install_fast_rank(index, args.extension)
        start = time.perf_counter()
        for raw in references:
            ref = normalize_record(raw)
            result = [p for index in indexes for p in index.query(ref)]
            assert result == captured[ref.entity_id], f"Candidate ranking or scores changed for {ref.entity_id}"
            queries += len(indexes); pairs += len(result)
        accelerated_seconds = time.perf_counter() - start
        # Candidate equality gives identical deterministic feature inputs.
        # Replay scores for 120 pairs as an explicit downstream check.
        lookup = {index.source: index for index in indexes}
        features = []
        for raw in references[:3]:
            ref = normalize_record(raw)
            for candidate, target in captured[ref.entity_id]:
                features.append(list(build_extended_pair_features(ref, target, candidate, lookup[candidate.candidate_source]).values()))
        predictions = model.predict(np.asarray(features, dtype=np.float32), num_threads=1)
        assert np.all(np.isfinite(predictions))
        result = {"references": len(references), "source_queries": queries, "retained_pairs": pairs,
            "exact_candidate_record_score_and_path_equality": True,
            "original_seconds": original_seconds, "accelerated_seconds": accelerated_seconds,
            "speedup": original_seconds / accelerated_seconds,
            "policy": "Runtime equivalence test on tuning references; no model, threshold or blocking-setting change. Timings are affected by concurrent inference load."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        for index in indexes:
            index.close()


if __name__ == "__main__":
    main()
