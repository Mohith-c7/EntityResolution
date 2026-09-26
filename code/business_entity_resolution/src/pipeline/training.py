"""Reproducible pilot using complete target indexes and entity-disjoint labels."""

import csv
import gc
import json
import time
import multiprocessing
import hashlib
import shutil
import lightgbm
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..blocking.disk_index import DiskSearchConfig, DiskSourceIndex, build_disk_index, normalize_record
from ..blocking.anchor_index import build_anchor_index
from ..evaluation.metrics import bootstrap_interval, score_matches
from ..evaluation.validation import entity_fold, sample_references
from ..features.pairwise_features import build_pair_features, build_extended_pair_features
from ..features.registry import DEFINITIONS, FEATURE_NAMES, FEATURE_VERSION, FEATURE_NAMES_V2, FEATURE_VERSION_V2
from ..model.name_aliases import fit_name_aliases
from ..model.predict import predict
from ..model.threshold import select_matches, tune_threshold
from ..model.train import train


def json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_labels(path: Path, selected: set[str], seed: int = 42):
    truth = {}
    target_folds = {}
    counts = Counter()
    source_cardinalities = {"S2": Counter(), "S3": Counter()}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            eid = row["source1_entity_id"]
            matches = set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
            fold = entity_fold(eid, seed)
            counts[fold] += 1
            for mid in matches:
                old = target_folds.setdefault(mid, fold)
                if old != fold:
                    raise ValueError("Target ownership crosses folds; rebuild component-level splits")
            for source in source_cardinalities:
                source_cardinalities[source][sum(mid.startswith(source + "-") for mid in matches)] += 1
            if eid in selected:
                truth[eid] = matches
    if set(truth) != selected:
        raise ValueError("Selected reference IDs are missing from ground truth")
    return truth, target_folds, {"s1_fold_counts": dict(counts), "matched_target_count": len(target_folds),
        "source_match_cardinalities": {source: dict(sorted(hist.items())) for source, hist in source_cardinalities.items()}}


def build_pairs(references: list[dict], indexes, truth: dict, fold: str, target_folds: dict, seed: int = 42, *, extended=False, filter_training=True, log_progress=True):
    start = time.perf_counter()
    rows, candidate_map, countries = [], {}, {}
    filtered = 0
    path_counts = Counter()
    for number, raw in enumerate(references, 1):
        ref = normalize_record(raw)
        countries[ref.entity_id] = ref.country
        candidates = [pair for index in indexes for pair in index.query(ref)]
        index_by_source = {index.source: index for index in indexes} if extended else {}
        candidate_map[ref.entity_id] = [candidate.candidate_entity_id for candidate, target in candidates]
        for candidate, target in candidates:
            # Protect held-out counterpart identities even when they occur as
            # distractors for a training anchor. Validation is never filtered.
            if filter_training and fold == "train" and target_folds.get(target.entity_id, entity_fold(target.entity_id, seed)) != "train":
                filtered += 1
                continue
            path_counts.update(candidate.blocking_paths)
            rows.append({"source1_entity_id": ref.entity_id, "candidate_entity_id": target.entity_id,
                "candidate_source": candidate.candidate_source, "rank_within_source": candidate.rank_within_source,
                "blocking_paths": ",".join(candidate.blocking_paths),
                "label": int(target.entity_id in truth[ref.entity_id]),
                **(build_extended_pair_features(ref, target, candidate, index_by_source[candidate.candidate_source])
                   if extended else build_pair_features(ref, target, candidate))})
        if log_progress and number % 100 == 0:
            print(json.dumps({"stage": "pair_features", "fold": fold, "references": number,
                "pairs": len(rows), "seconds": round(time.perf_counter() - start, 1)}), flush=True)
    names = FEATURE_NAMES_V2 if extended else FEATURE_NAMES
    columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", "rank_within_source", "blocking_paths", "label", *names]
    frame = pd.DataFrame(rows, columns=columns)
    for name in names:
        frame[name] = frame[name].astype(np.float32)
    available = {eid: set(cids) & truth[eid] for eid, cids in candidate_map.items()}
    ceiling = score_matches({eid: truth[eid] for eid in candidate_map}, available, countries)
    summary = {"entities": len(references), "scored_pairs": len(frame), "retrieved_pairs": sum(map(len, candidate_map.values())),
        "training_pairs_excluded_for_entity_isolation": filtered,
        "blocking_recall": ceiling["micro_recall"], "oracle_macro_f05": ceiling["macro_f05"],
        "country_oracle": ceiling["country_macro_f05"], "path_pair_counts": dict(path_counts),
        "mean_candidates": float(np.mean([len(v) for v in candidate_map.values()])),
        "p95_candidates": float(np.percentile([len(v) for v in candidate_map.values()], 95)),
        "seconds": time.perf_counter() - start}
    return frame, countries, summary


_TRAIN_INDEXES = None


def initialize_training_worker(paths, config, alias_path):
    global _TRAIN_INDEXES
    from ..model.name_aliases import NameAliases
    aliases = NameAliases.load(alias_path) if alias_path else None
    _TRAIN_INDEXES = [DiskSourceIndex(path, config, aliases=aliases) for path in paths]


def training_chunk(task):
    raw, truth, fold, seed, extended = task
    return build_pairs(raw,_TRAIN_INDEXES,truth,fold,{},seed,extended=extended,filter_training=False,log_progress=False)


def parallel_pairs(references,indexes,truth,fold,target_folds,artifact_dir,config,workers,seed,extended):
    if workers == 1:
        return build_pairs(references,indexes,truth,fold,target_folds,seed,extended=extended)
    start = time.perf_counter()
    tasks=[]
    for offset in range(0,len(references),250):
        raw=references[offset:offset+250]
        tasks.append((raw,{row["entity_id"]:truth[row["entity_id"]] for row in raw},fold,seed,extended))
    frames,countries,paths=[],{},Counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_training_worker,initargs=([idx.path for idx in indexes],config,artifact_dir/"name_aliases.json" if extended else None)) as executor:
        completed=0
        for frame,country_map,summary in executor.map(training_chunk,tasks):
            frames.append(frame); countries.update(country_map); paths.update(summary["path_pair_counts"])
            completed += summary["entities"]
            print(json.dumps({"stage":"pair_features","fold":fold,"references":completed,"workers":workers,
                "seconds":round(time.perf_counter()-start,1)}),flush=True)
    frame=pd.concat(frames,ignore_index=True)
    counts=frame.groupby("source1_entity_id").size().reindex(countries,fill_value=0).to_numpy()
    available={eid:set() for eid in countries}
    for eid,cid in frame.loc[frame["label"]==1,["source1_entity_id","candidate_entity_id"]].itertuples(index=False,name=None):
        available[eid].add(cid)
    oracle=score_matches({eid:truth[eid] for eid in countries},available,countries)
    retrieved=len(frame)
    if fold=="train":
        allowed=np.fromiter((target_folds.get(cid,entity_fold(cid,seed))=="train" for cid in frame["candidate_entity_id"]),dtype=bool,count=len(frame))
        if frame.loc[~allowed,"label"].sum():
            raise ValueError("Positive training counterpart crosses folds")
        frame=frame.loc[allowed].reset_index(drop=True)
    summary={"entities":len(references),"scored_pairs":len(frame),"retrieved_pairs":retrieved,
        "training_pairs_excluded_for_entity_isolation":retrieved-len(frame),"blocking_recall":oracle["micro_recall"],
        "oracle_macro_f05":oracle["macro_f05"],"country_oracle":oracle["country_macro_f05"],"path_pair_counts":dict(paths),
        "mean_candidates":float(np.mean(counts)),"p95_candidates":float(np.percentile(counts,95)),"seconds":time.perf_counter()-start}
    return frame,countries,summary


def run_training(train_dir: Path, index_dir: Path, artifact_dir: Path, *, train_entities=2000, tune_entities=500,
                 holdout_entities=500, config: DiskSearchConfig | None = None, threads=4, seed=42,workers=4):
    if workers < 1:
        raise ValueError("workers must be positive")
    config = config or DiskSearchConfig()
    train_dir, index_dir, artifact_dir = Path(train_dir), Path(index_dir), Path(artifact_dir)
    if (artifact_dir / "model.txt").exists():
        raise FileExistsError(f"Completed experiment exists at {artifact_dir}; choose a new artifact directory")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    indexes = []
    try:
        aliases = fit_name_aliases(train_dir, artifact_dir / "name_aliases.json", seed=seed) if config.use_name_aliases else None
        extended = config.use_name_aliases
        names, version = (FEATURE_NAMES_V2, FEATURE_VERSION_V2) if extended else (FEATURE_NAMES, FEATURE_VERSION)
        for source, number in (("S2", 2), ("S3", 3)):
            path = index_dir / f"index_train_{source}.sqlite"
            build_disk_index(train_dir / f"train_source{number}.tsv", path, source)
            if config.retrieval_mode == "anchored":
                build_anchor_index(path, aliases)
            indexes.append(DiskSourceIndex(path, config, aliases=aliases))
        sizes = {"train": train_entities, "tune": tune_entities, "holdout": holdout_entities}
        references, split = sample_references(train_dir / "train_source1.tsv", sizes, artifact_dir / "split_manifest.tsv", seed)
        selected = {row["entity_id"] for rows in references.values() for row in rows}
        truth, target_folds, labels_meta = load_labels(train_dir / "train_ground_truth.tsv", selected, seed)
        json_write(artifact_dir / "sampled_references.json", references)
        json_write(artifact_dir / "sampled_truth.json", {eid: sorted(matches) for eid, matches in truth.items()})
        json_write(artifact_dir / "split.json", {**split, **labels_meta})
        print(json.dumps({"stage": "split_ready", **split["sample_counts"]}), flush=True)
        frames, country_maps, blocking = {}, {}, {}
        for fold, records in references.items():
            frames[fold], country_maps[fold], blocking[fold] = parallel_pairs(records,indexes,truth,fold,target_folds,artifact_dir,config,workers,seed,extended)
            frames[fold].to_parquet(artifact_dir / f"pairs_{fold}.parquet", index=False)
            json_write(artifact_dir / "blocking_progress.json", blocking)
        del target_folds
        gc.collect()
        model = train(frames["train"], frames["tune"], threads=threads, seed=seed, feature_names=names)
        tune_truth = {raw["entity_id"]: truth[raw["entity_id"]] for raw in references["tune"]}
        holdout_truth = {raw["entity_id"]: truth[raw["entity_id"]] for raw in references["holdout"]}
        tune_prob = predict(model, frames["tune"])
        threshold, sweep = tune_threshold(frames["tune"], tune_prob, tune_truth)
        holdout_prob = predict(model, frames["holdout"])
        tune_pred = select_matches(frames["tune"], tune_prob, threshold, tune_truth)
        holdout_pred = select_matches(frames["holdout"], holdout_prob, threshold, holdout_truth)
        model.booster_.save_model(str(artifact_dir / "model.txt"))
        shutil.copyfile(Path(__file__).parents[1]/"model/MODEL_LICENSE.txt",artifact_dir/"MODEL_LICENSE.txt")
        package=Path(__file__).parents[2]
        code_files=sorted(package.glob("src/**/*.py"))+sorted(package.glob("run_*.py"))+[package/"requirements.txt"]
        provenance={"source_sha256":{str(path.relative_to(package)):hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files},
            "input_files":{path.name:{"size":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns}
                for path in sorted(train_dir.glob("train_*.tsv"))},
            "index_metadata":{index.source:index.meta for index in indexes}}
        json_write(artifact_dir/"reproduction.json",provenance)
        for fold, probs in (("tune", tune_prob), ("holdout", holdout_prob)):
            frames[fold].assign(probability=probs).to_parquet(artifact_dir / f"predictions_{fold}.parquet", index=False)
        json_write(artifact_dir / "threshold_sweep.json", sweep)
        json_write(artifact_dir / "feature_registry.json", {"version": version, "features": names, "definitions": DEFINITIONS,
            "extended": "Canonical name similarities use train-only aliases; family count and IDF evidence are target-local. Ordered numbers have a zero-stripped alternate view, while original numeric strings are preserved." if extended else None})
        json_write(artifact_dir / "holdout_matches.json", holdout_pred)
        importance = dict(zip(names, map(float, model.booster_.feature_importance(importance_type="gain"))))
        errors = []
        raw_holdout = {row["entity_id"]: row for row in references["holdout"]}
        for eid, expected in holdout_truth.items():
            predicted = set(holdout_pred[eid])
            if predicted != expected:
                errors.append({"reference": raw_holdout[eid], "missing": sorted(expected - predicted), "false_matches": sorted(predicted - expected)})
        json_write(artifact_dir / "holdout_errors.json", errors)
        report = {
            "experiment": artifact_dir.name, "seed": seed, "feature_version": version,
            "name_alias_model": aliases.metadata if aliases else None,
            "search_config": asdict(config), "split": split, "threshold": threshold,
            "model": {"library": "lightgbm", "version": lightgbm.__version__, "license": "MIT", "artifact_license":"MODEL_LICENSE.txt", "best_iteration": model.best_iteration_, "parameters": model.get_params()},
            "target_pool": {idx.source: idx.record_count for idx in indexes},
            "validation_protocol": "Full training target-pool stress validation. S1 folds are disjoint; held-out target identities are excluded from all supervised training examples. Corpus-local index frequencies are label-free. This is not a French test estimate or a private leaderboard score.",
            "blocking": blocking,
            "tune": score_matches(tune_truth, tune_pred, country_maps["tune"]),
            "holdout": score_matches(holdout_truth, holdout_pred, country_maps["holdout"]),
            "holdout_macro_f05_bootstrap_95": bootstrap_interval(holdout_truth, holdout_pred, seed=seed),
            "feature_gain": importance, "seconds": time.perf_counter() - start,
        }
        json_write(artifact_dir / "report.json", report)
        print(json.dumps({"stage": "training_complete", "threshold": threshold, "holdout": report["holdout"], "artifact_dir": str(artifact_dir)}), flush=True)
        return report
    finally:
        for index in indexes:
            index.close()
