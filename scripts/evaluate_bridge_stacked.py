"""Score saved bridge-retrieval candidates of the selection references with a stacked second stage.

The saved pair features are re-scored with the current first stage, then context
and competition features are rebuilt exactly as for the original candidates.
Only the selection references' labels are read, for scoring.
"""
import argparse
import csv
import json
import multiprocessing
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

from build_context_features import CONTEXT_NAMES, clean_record, pair_context
from train_scale_model import FEATURE_NAMES_V3, predict, score, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.disk_index import normalize_record
from src.features.competition import COMPETITION_NAMES, KEY_KINDS, competition_features, count_universe, record_keys

STATE = None


def init(adapter, first_stage):
    global STATE
    pa.set_cpu_count(1)
    refs = {}
    for raw in json.loads((Path(adapter) / "sampled_references.json").read_text())["tune"]:
        ref = normalize_record(raw); refs[ref.entity_id] = clean_record(ref.name, ref.address)
    conns = {s: sqlite3.connect(Path(f"models/index_train_{s}.sqlite").resolve().as_uri() + "?mode=ro", uri=True) for s in ("S2", "S3")}
    STATE = refs, conns, lgb.Booster(model_file=first_stage)


def chunk(path):
    refs, conns, booster = STATE
    frame = pd.read_parquet(path, columns=[*FEATURE_NAMES_V3, "source1_entity_id", "candidate_entity_id"])
    first = booster.predict(frame[list(FEATURE_NAMES_V3)].to_numpy(dtype=np.float32), num_threads=1)
    targets = {}
    for source, conn in conns.items():
        ids = sorted(set(frame.loc[frame.candidate_entity_id.str.startswith(source), "candidate_entity_id"]))
        for start in range(0, len(ids), 500):
            part = ids[start:start + 500]
            for cid, name, address in conn.execute(f"SELECT id,name,address FROM records WHERE id IN ({','.join('?' * len(part))})", part):
                targets[cid] = clean_record(name, address)
    context = np.empty((len(frame), len(CONTEXT_NAMES)), dtype=np.float32)
    for eid, positions in frame.groupby("source1_entity_id", sort=False).indices.items():
        context[positions] = pair_context(refs[eid], targets, frame.candidate_entity_id.iloc[positions].tolist(), first[positions])
    for i, name in enumerate(CONTEXT_NAMES): frame[name] = context[:, i]
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("models/next_round/bridge_selection_cache"))
    p.add_argument("--adapter", type=Path, default=Path("models/next_round/context_selection_adapter"))
    p.add_argument("--first-stage", type=Path, default=Path("models/next_round/extended_model/model.txt"))
    p.add_argument("--models", type=Path, nargs="*", default=[], help="Without models, only the feature cache is built")
    p.add_argument("--universe", type=Path, default=Path("dataset/train/train_source1.tsv"))
    p.add_argument("--baseline-threshold", type=float, default=.76)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--features-output", type=Path, default=Path("models/next_round/bridge_selection_stacked_features.parquet"))
    p.add_argument("--report-output", type=Path, default=Path("reports/experiments/round3/bridge_stacked.json"))
    args = p.parse_args()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if not manifest["complete"] or manifest["test"]: raise ValueError("Completed selection cache required")
    raw = {r["entity_id"]: r for r in json.loads((args.adapter / "sampled_references.json").read_text())["tune"]}
    ids = list(manifest["reference_ids"])
    if set(ids) != set(raw): raise ValueError("Adapter and cache references differ")
    if args.features_output.exists():
        frame = pd.read_parquet(args.features_output)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init,
                                 initargs=(str(args.adapter), str(args.first_stage))) as pool:
            frame = pd.concat(pool.map(chunk, [str(args.cache / f) for f in manifest["files"]]), ignore_index=True)
        if len(frame) != manifest["rows_scored"] or frame.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
            raise ValueError("Pair count mismatch")
        reference_keys = {e: record_keys(r.name, r.core, r.address) for e, r in ((e, normalize_record(raw[e])) for e in ids)}
        target_keys = {}
        for source in ("S2", "S3"):
            conn = sqlite3.connect(Path(f"models/index_train_{source}.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
            wanted_ids = sorted({c for c in frame.candidate_entity_id if c.startswith(source + "-")})
            for i in range(0, len(wanted_ids), 500):
                part = wanted_ids[i:i + 500]
                for cid, name, core, address in conn.execute(f"SELECT id,name,core,address FROM records WHERE id IN ({','.join('?' * len(part))})", part):
                    target_keys[cid] = record_keys(name, core, address)
        wanted = {kind: set() for kind in KEY_KINDS}
        for keys in (*reference_keys.values(), *target_keys.values()):
            for kind in KEY_KINDS:
                if keys[kind] is not None: wanted[kind].add(keys[kind])

        def universe():
            with args.universe.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    record = normalize_record(row)
                    yield record_keys(record.name, record.core, record.address)
        counts = count_universe(universe(), wanted)
        frame[list(COMPETITION_NAMES)] = np.asarray([competition_features(reference_keys[s], target_keys[c], counts)
            for s, c in zip(frame.source1_entity_id, frame.candidate_entity_id)], dtype=np.float32)
        frame.to_parquet(args.features_output, index=False)
    if not args.models: return
    truth = {e: set(v) for e, v in json.loads((args.adapter / "sampled_truth.json").read_text()).items()}
    positions = {e: i for i, e in enumerate(ids)}
    groups = frame.source1_entity_id.map(positions).to_numpy(dtype=np.int32)
    label = np.array([c in truth[e] for e, c in zip(frame.source1_entity_id, frame.candidate_entity_id)], dtype=np.uint8)
    expected = np.array([len(truth[e]) for e in ids])
    countries = np.array([raw[e]["country"].casefold() for e in ids])
    base = frame.ctx_probability.to_numpy()
    baseline, _ = score(base, args.baseline_threshold, label, groups, expected, countries)
    report = {"status": "exploratory_selection_only", "first_stage_bridge": baseline,
              "oracle": score(label, .5, label, groups, expected, countries)[0], "models": {}}
    for model_dir in args.models:
        chosen = json.loads((model_dir / "report.json").read_text())["protocol_selection"]
        model = lgb.Booster(model_file=str(model_dir / "model.txt"))
        stacked = predict(model, frame[model.feature_name()].to_numpy(dtype=np.float32), args.workers)
        fixed = score((1 - chosen["context_weight"]) * base + chosen["context_weight"] * stacked, chosen["threshold"],
                      label, groups, expected, countries)[0]
        trials = []
        for weight in (0., .25, .5, .75, 1.):
            blended = (1 - weight) * base + weight * stacked
            for threshold in np.r_[np.arange(.30, .951, .01), .96, .97, .98, .99]:
                trials.append({"context_weight": weight, **score(blended, round(float(threshold), 6), label, groups, expected)[0]})
        limits = json.loads((model_dir / "report.json").read_text())["first_stage_baseline"]
        eligible = [r for r in trials if r["micro_precision"] >= limits["micro_precision"] - .002
                    and r["singleton_false_positives"] <= limits["singleton_false_positives"]]
        report["models"][str(model_dir)] = {"original_candidate_selection": chosen, "same_rule_bridge": fixed,
            "reselected_bridge": max(eligible, key=lambda r: r["macro_f05"]) if eligible else None}
        np.save(model_dir / "bridge_selection_probabilities.npy", stacked)
    write_json(args.report_output, report)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
