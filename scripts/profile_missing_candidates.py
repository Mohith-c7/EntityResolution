"""Locate known tuning retrieval misses; this is an error-only diagnostic, not an evaluation."""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.cheap_ranker import RANK_FEATURES, rank_features
from src.blocking.disk_index import DiskSearchConfig, normalize_record
from src.blocking.postings_index import PostingsConfig, PostingsSourceIndex
from src.features.pairwise_features import build_v3_evidence
from src.model.name_aliases import NameAliases


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/redesign/missing_candidate_profile"))
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); os.nice(15); pa.set_cpu_count(1)
    artifact = Path("models/features_v3_audit01")
    missed_path = Path("models/redesign/error_analysis/missed_links.parquet")
    missed = pd.read_parquet(missed_path)
    refs = {r["entity_id"]: r for r in json.loads((artifact / "sampled_references.json").read_text())["tune"]}
    if not set(missed.source1_entity_id) <= refs.keys(): raise ValueError("Only inspected tuning references are allowed")
    manifest = json.loads(Path("models/redesign/wide_tune_cache/manifest.json").read_text())
    if not manifest["complete"] or manifest["test"]: raise ValueError("Expected a completed tuning cache")
    config = DiskSearchConfig(**json.loads((artifact / "report.json").read_text())["search_config"])
    options = PostingsConfig(**manifest["config"])
    if config.top_k != manifest["top_k"]: raise ValueError("Candidate budget mismatch")
    aliases = NameAliases.load(artifact / "name_aliases.json")
    indexes = {s: PostingsSourceIndex(Path(f"models/index_train_{s}.sqlite"), config, aliases=aliases,
        postings_config=options, native_extension=Path("models/runtime/erpostings.dylib")) for s in ("S2", "S3")}
    ranker_path = Path("models/redesign/cheap_ranker_v1/model.txt")
    ranker = lgb.Booster(model_file=str(ranker_path))
    if ranker.feature_name() != list(RANK_FEATURES): raise ValueError("Ranker feature order mismatch")
    protocol = {"split": "previously_inspected_tuning_errors_only", "missed_links_sha256": sha(missed_path),
        "aliases_sha256": sha(artifact / "name_aliases.json"), "ranker_sha256": sha(ranker_path),
        "original_postings": asdict(options), "wider_postings": asdict(replace(options, fused_budget=128)),
        "final_candidates_per_source": config.top_k,
        "limitation": "Conditioned on existing misses. Does not measure lost previously retrieved links, precision, runtime, or overall macro F0.5. No promotion from this probe.",
        "fresh_audit_evaluated": False}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    missed["source"] = missed.candidate_entity_id.str[:2]
    results = []; started = time.perf_counter()
    for (eid, source), group in missed.groupby(["source1_entity_id", "source"], sort=True):
        index = indexes[source]; ref = normalize_record(refs[eid])
        index.postings_config = options
        original = index.query(ref, return_all=True)
        original_ranks = {c.candidate_entity_id: c.rank_within_source for c, _ in original}
        index.postings_config = replace(options, fused_budget=128)
        wider = index.query(ref, return_all=True)
        wider_ranks = {c.candidate_entity_id: c.rank_within_source for c, _ in wider}
        matrix = np.array([rank_features(ref, t, index) for _, t in wider], dtype=np.float32)
        probabilities = ranker.predict(matrix, num_threads=1) if len(matrix) else []
        ordered = sorted(zip(wider, probabilities), key=lambda x: (-x[1], x[0][0].candidate_entity_id))
        learned_ranks = {item[0][0].candidate_entity_id: i + 1 for i, item in enumerate(ordered)}
        for cid in group.candidate_entity_id:
            original_rank = original_ranks.get(cid, 0)
            if original_rank and original_rank <= config.top_k: raise ValueError("Saved miss is now retrieved; configuration drift")
            row = index.connection.execute("SELECT id,name,core,address,country,name_digits,address_digits,postcodes,folded FROM records WHERE id=?", (cid,)).fetchone()
            if row is None: raise ValueError("Unknown target")
            target = index.record_values(row)
            index.prefetch_frequencies([ref, target])
            features = build_v3_evidence(ref, target, index)
            results.append({"source1_entity_id": eid, "candidate_entity_id": cid, "country": ref.country,
                "original_rank": original_rank, "wider_rank": wider_ranks.get(cid, 0),
                "learned_rank": learned_ranks.get(cid, 0), "candidate_address_missing": not bool(target.address),
                "candidate_non_latin_name": any(c.isalpha() and not c.isascii() for c in target.name),
                "reference_name": ref.name, "candidate_name": target.name,
                "reference_address": ref.address, "candidate_address": target.address, **features})
        if len(results) % 100 < len(group):
            print(json.dumps({"misses_profiled": len(results), "total": len(missed), "seconds": time.perf_counter()-started}), flush=True)
    frame = pd.DataFrame(results)
    frame.to_parquet(args.output / "misses_profiled.parquet", index=False)
    def counts(part):
        return {"missed_links": len(part), "present_before_original_top20": int((part.original_rank > 0).sum()),
            "absent_from_original_shortlist": int((part.original_rank == 0).sum()),
            "present_in_wider_shortlist": int((part.wider_rank > 0).sum()),
            "recovered_by_wider_original_ranker": int(part.wider_rank.between(1, config.top_k).sum()),
            "recovered_by_wider_learned_ranker": int(part.learned_rank.between(1, config.top_k).sum())}
    report = {**protocol, "counts": counts(frame),
        "slices": {"missing_address": counts(frame[frame.candidate_address_missing]),
            "non_ascii_name": counts(frame[frame.candidate_non_latin_name]),
            **{f"country_{c}": counts(f) for c, f in frame.groupby("country")}},
        "seconds": time.perf_counter()-started}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("reports/experiments/redesign/missing_candidate_profile.json").write_text(json.dumps(report, indent=2) + "\n")
    for index in indexes.values(): index.close()
    print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
