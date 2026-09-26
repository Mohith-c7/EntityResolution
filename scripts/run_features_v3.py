"""Train v3 on cached train pairs, select on 10k tuning entities, audit once on fresh 5k.

Run from repository root. Full submission generation is deliberately separate.
"""
import argparse
import csv
import hashlib
import heapq
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.disk_index import DiskSearchConfig, DiskSourceIndex, normalize_record
from src.evaluation.metrics import entity_f05, score_matches, bootstrap_interval
from src.evaluation.validation import entity_fold, stable_hash
from src.features.pairwise_features import build_v3_evidence
from src.features.registry import FEATURE_NAMES_V2, FEATURE_NAMES_V3, FEATURE_NAMES_V3_EXTRA, FEATURE_VERSION_V3
from src.model.name_aliases import NameAliases
from src.model.train import train
from src.model.threshold import select_matches, tune_threshold
from src.pipeline.training import parallel_pairs, json_write


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def collect_labels(path, ids):
    truth = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["source1_entity_id"] in ids:
                truth[row["source1_entity_id"]] = set(filter(None, row["matched_entity_ids"].split(",")))
    if set(truth) != ids: raise ValueError("Missing selected labels")
    return truth


def prepare_references(args, output):
    refs_path = output / "sampled_references.json"
    if refs_path.exists():
        refs = json.loads(refs_path.read_text())
        if len(refs["tune"]) != args.tune_entities or len(refs["holdout"]) != args.audit_entities:
            raise ValueError("Cached reference counts differ; use a new artifact directory")
        return refs
    excluded = set()
    sources = []
    for path in sorted(Path("models").rglob("sampled_references.json")):
        if path == refs_path: continue
        data = json.loads(path.read_text())
        for rows in data.values(): excluded.update(row["entity_id"] for row in rows)
        sources.append({"path": str(path), "sha256": digest(path)})
    sizes = {"tune": args.tune_entities, "holdout": args.audit_entities}
    heaps = {fold: [] for fold in sizes}
    with (args.train_dir / "train_source1.tsv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            eid = row["entity_id"]
            fold = entity_fold(eid)
            if fold not in heaps or eid in excluded: continue
            priority = stable_hash(eid, "feature-v3-fresh-audit-20260926")
            heap = heaps[fold]
            item = (-priority, eid, row)
            if len(heap) < sizes[fold]: heapq.heappush(heap, item)
            elif priority < -heap[0][0]: heapq.heapreplace(heap, item)
    refs = {fold: [item[2] for item in sorted(heap, key=lambda x: x[1])] for fold, heap in heaps.items()}
    if any(len(refs[f]) != sizes[f] for f in sizes): raise ValueError("Insufficient fresh references")
    refs["train"] = json.loads((args.baseline / "sampled_references.json").read_text())["train"]
    json_write(refs_path, refs)
    json_write(output / "audit_protocol.json", {
        "feature_version": FEATURE_VERSION_V3, "seed": 42, "tuning_entities": sizes["tune"],
        "audit_entities": sizes["holdout"], "previous_reference_files_excluded": sources,
        "sampling_domain": "feature-v3-fresh-audit-20260926", "sampling_uses_labels": False,
        "training": "Reuse baseline train-only pair rows and aliases; no holdout identity trains the model.",
        "threshold_policy": "Maximize tuning macro F0.5 subject to precision >= frozen baseline minus .002 and singleton false positives <= baseline on these same tuning entities.",
        "audit_acceptance": "Paired entity bootstrap gain lower bound > 0, precision loss <= .002, singleton false positives do not rise. Audit may be used once; any later tuning requires a new audit.",
        "changes": "Add 15 pair evidence features; fix approximate romanizer word-boundary schwa. Keep primary normalization, aliases, retrieval, top K and all previous 50 feature definitions unchanged.",
        "france": "No labeled France estimate; no country-specific cardinality cap or score target.",
    })
    return refs


def augment_cached_train(args, output, refs, indexes):
    destination = output / "pairs_train.parquet"
    if destination.exists(): return
    frame = pd.read_parquet(args.baseline / "pairs_train.parquet")
    left = {r["entity_id"]: normalize_record(r) for r in refs["train"]}
    candidates = {}
    for index in indexes:
        wanted = sorted(set(frame.loc[frame.candidate_source == index.source, "candidate_entity_id"]))
        for start in range(0, len(wanted), 700):
            chunk = wanted[start:start + 700]
            sql = "SELECT id,name,core,address,country,name_digits,address_digits,postcodes,folded FROM records WHERE id IN (" + ",".join("?" for _ in chunk) + ")"
            for row in index.connection.execute(sql, chunk): candidates[row[0]] = index.record_values(row)
    lookup = {index.source: index for index in indexes}
    values = []
    for start in range(0, len(frame), 5000):
        chunk = frame.iloc[start:start + 5000]
        for index in indexes:
            rows = chunk.loc[chunk.candidate_source == index.source]
            index.prefetch_frequencies([left[e] for e in rows.source1_entity_id] + [candidates[c] for c in rows.candidate_entity_id])
        for e, c, source in chunk[["source1_entity_id", "candidate_entity_id", "candidate_source"]].itertuples(index=False, name=None):
            values.append(list(build_v3_evidence(left[e], candidates[c], lookup[source]).values()))
        if start % 100000 == 0: print(json.dumps({"stage": "augment_train", "pairs": start}), flush=True)
    extra = pd.DataFrame(values, columns=FEATURE_NAMES_V3_EXTRA, dtype=np.float32, index=frame.index)
    frame = pd.concat([frame, extra], axis=1)
    frame.to_parquet(destination, index=False)


def pair_fold(args, output, refs, indexes, config, fold):
    path = output / f"pairs_{fold}.parquet"
    truth = collect_labels(args.train_dir / "train_ground_truth.tsv", {r["entity_id"] for r in refs[fold]})
    if not path.exists():
        frame, countries, blocking = parallel_pairs(refs[fold], indexes, truth, fold, {}, output,
            config, args.workers, 42, True, FEATURE_VERSION_V3)
        frame.to_parquet(path, index=False)
        json_write(output / f"blocking_{fold}.json", blocking)
        json_write(output / f"truth_{fold}.json", {e: sorted(v) for e, v in truth.items()})
    return pd.read_parquet(path), truth, {r["entity_id"]: r["country"] for r in refs[fold]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("models/larger_alias_v4"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("models/features_v3_audit01"))
    parser.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--index-dir", type=Path, default=Path("models"))
    parser.add_argument("--tune-entities", type=int, default=10000)
    parser.add_argument("--audit-entities", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    start = time.perf_counter()
    output = args.artifact_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "audit_report.json").exists(): raise FileExistsError("Audit already evaluated; do not reuse for selection")
    baseline = json.loads((args.baseline / "report.json").read_text())
    config = DiskSearchConfig(**baseline["search_config"])
    for filename in ("name_aliases.json", "MODEL_LICENSE.txt"):
        if not (output / filename).exists(): shutil.copyfile(args.baseline / filename, output / filename)
    refs = prepare_references(args, output)
    aliases = NameAliases.load(output / "name_aliases.json")
    indexes = [DiskSourceIndex(args.index_dir / f"index_train_{source}.sqlite", config, aliases=aliases) for source in ("S2", "S3")]
    try:
        augment_cached_train(args, output, refs, indexes)
        tune, truth, countries = pair_fold(args, output, refs, indexes, config, "tune")
        base_model = lgb.Booster(model_file=str(args.baseline / "model.txt"))
        bp = base_model.predict(tune[list(FEATURE_NAMES_V2)].to_numpy(np.float32), num_threads=args.threads)
        base_tune = score_matches(truth, select_matches(tune, bp, baseline["threshold"], truth), countries)
        if not (output / "frozen_selection.json").exists():
            train_pairs = pd.read_parquet(output / "pairs_train.parquet")
            model = train(train_pairs, tune, threads=args.threads, feature_names=FEATURE_NAMES_V3)
            probs = model.predict_proba(tune[list(FEATURE_NAMES_V3)].astype(np.float32))[:, 1]
            _, sweep = tune_threshold(tune, probs, truth)
            eligible = [r for r in sweep if r["micro_precision"] >= base_tune["micro_precision"] - .002 and r["singleton_false_positives"] <= base_tune["singleton_false_positives"]]
            if not eligible: raise ValueError("No threshold passes tuning safety criteria")
            chosen = max(eligible, key=lambda r: (r["macro_f05"], r["micro_precision"], r["threshold"]))
            model.booster_.save_model(str(output / "model.txt"))
            json_write(output / "feature_registry.json", {"version": FEATURE_VERSION_V3, "features": FEATURE_NAMES_V3})
            json_write(output / "threshold_sweep.json", sweep)
            tune.assign(probability=probs, baseline_probability=bp).to_parquet(output / "predictions_tune.parquet", index=False)
            report = {"experiment": output.name, "feature_version": FEATURE_VERSION_V3, "threshold": chosen["threshold"],
                "search_config": asdict(config), "tune": chosen, "baseline_same_tune": base_tune,
                "model": {"library": "lightgbm", "version": lgb.__version__, "license": "MIT", "best_iteration": model.best_iteration_, "parameters": model.get_params()},
                "feature_gain": dict(zip(FEATURE_NAMES_V3, map(float, model.booster_.feature_importance("gain")))),
                "validation_protocol": "10k unused tuning entities; threshold frozen before scoring 5k unused holdout entities. No leaderboard estimate.",
                "leaderboard_score": None, "promoted": False}
            json_write(output / "report.json", report)
            sources = sorted(Path("code/business_entity_resolution/src").rglob("*.py")) + [Path(__file__)]
            json_write(output / "frozen_selection.json", {"model_sha256": digest(output / "model.txt"),
                "threshold": chosen["threshold"], "baseline_model_sha256": digest(args.baseline / "model.txt"),
                "alias_sha256": digest(output / "name_aliases.json"), "reference_manifest_sha256": digest(output / "sampled_references.json"),
                "source_sha256": {str(p): digest(p) for p in sources},
                "train_pairs_sha256": digest(output / "pairs_train.parquet"), "tune_pairs_sha256": digest(output / "pairs_tune.parquet")})
            print(json.dumps({"stage": "selection_frozen", "threshold": chosen["threshold"], "tune_macro": chosen["macro_f05"], "baseline_tune": base_tune["macro_f05"]}), flush=True)
            del train_pairs, model
        frozen = json.loads((output / "frozen_selection.json").read_text())
        if digest(output / "model.txt") != frozen["model_sha256"]: raise ValueError("Frozen model changed")
        if digest(args.baseline / "model.txt") != frozen["baseline_model_sha256"]: raise ValueError("Baseline changed")
        if digest(output / "name_aliases.json") != frozen["alias_sha256"]: raise ValueError("Aliases changed")
        if digest(output / "sampled_references.json") != frozen["reference_manifest_sha256"]: raise ValueError("References changed")
        if any(digest(p) != h for p, h in frozen["source_sha256"].items()): raise ValueError("Code changed after selection")
        audit, actual, countries = pair_fold(args, output, refs, indexes, config, "holdout")
        model = lgb.Booster(model_file=str(output / "model.txt"))
        new_p = model.predict(audit[list(FEATURE_NAMES_V3)].to_numpy(np.float32), num_threads=args.threads)
        old_p = base_model.predict(audit[list(FEATURE_NAMES_V2)].to_numpy(np.float32), num_threads=args.threads)
        new_matches = select_matches(audit, new_p, frozen["threshold"], actual)
        old_matches = select_matches(audit, old_p, baseline["threshold"], actual)
        new_score, old_score = (score_matches(actual, p, countries) for p in (new_matches, old_matches))
        delta = np.array([entity_f05(actual[e], new_matches[e]) - entity_f05(actual[e], old_matches[e]) for e in sorted(actual)])
        rng = np.random.default_rng(42)
        ci = np.quantile([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(2000)], [.025, .975]).tolist()
        gates = {"positive_gain_lower_bound": ci[0] > 0, "precision_loss_at_most_002": new_score["micro_precision"] >= old_score["micro_precision"] - .002,
                 "singleton_false_positives_do_not_rise": new_score["singleton_false_positives"] <= old_score["singleton_false_positives"]}
        result = {"baseline": old_score, "candidate": new_score, "paired_macro_gain": float(delta.mean()), "paired_gain_95": ci,
            "candidate_macro_95": bootstrap_interval(actual, new_matches), "acceptance_gates": gates, "accepted": all(gates.values()),
            "seconds": time.perf_counter() - start, "leaderboard_score": None}
        audit.assign(probability=new_p, baseline_probability=old_p).to_parquet(output / "predictions_holdout.parquet", index=False)
        json_write(output / "audit_report.json", result)
        report = json.loads((output / "report.json").read_text())
        report.update(holdout=new_score, audit=result, promoted=all(gates.values()))
        report["blocking"] = {fold: json.loads((output / f"blocking_{fold}.json").read_text()) for fold in ("tune", "holdout")}
        report["selection_status"] = "quality_verified_runtime_pending" if all(gates.values()) else "audit_gate_failed"
        json_write(output / "report.json", report)
        # Keep the standard diagnostic interface used by analyze_tuning_errors.
        train_truth = json.loads((args.baseline / "sampled_truth.json").read_text())
        selected_truth = {r["entity_id"]: train_truth[r["entity_id"]] for r in refs["train"]}
        for fold in ("tune", "holdout"):
            selected_truth.update(json.loads((output / f"truth_{fold}.json").read_text()))
        json_write(output / "sampled_truth.json", selected_truth)
        print(json.dumps({"stage": "audit_complete", **result}), flush=True)
    finally:
        for index in indexes: index.close()


if __name__ == "__main__": main()
