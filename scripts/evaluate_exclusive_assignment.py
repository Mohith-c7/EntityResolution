"""Development check of one-owner-per-target assignment.

The training ground truth never gives one Source 2/3 record to two Source 1
records. At inference every Source 1 record is scored, so a target claimed by
several references can go to its most probable claimant. This reproduces that
for the development selection references: other Source 1 records sharing a
competition key with a predicted target are found without labels and scored
with the same retrieval, first-stage, context and competition stages. Only the
selection references' labels are read, for scoring.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
import build_scale_cache as bsc
from build_context_features import CONTEXT_NAMES, clean_record, pair_context
from train_scale_model import FEATURE_NAMES_V3, digest, predict, score, write_json
from src.blocking.disk_index import normalize_record
from src.blocking.postings_index import PostingsConfig
from src.evaluation.validation import entity_fold
from src.features.competition import COMPETITION_NAMES, KEY_KINDS, competition_features, count_universe, record_keys
from src.features.pairwise_features import build_versioned_pair_features
from src.features.registry import FEATURE_VERSION_V3
from src.model.crossfit_aliases import inner_fold

WORKER = None
ALIASES = Path("models/scale_v1_plan/aliases")


def init(alias_path, first_stage, bridge=None):
    global WORKER
    search = json.loads(Path("models/features_v3_audit01/report.json").read_text())["search_config"]
    bsc.init(str(Path("models").resolve()), alias_path, str((ALIASES / "counts.sqlite").resolve()), search,
             asdict(PostingsConfig(max_postings=100000, fused_budget=128 if bridge else 64)),
             str(Path("models/runtime/erpostings.dylib").resolve()), "competitors", "/tmp", bridge)
    conns = {s: sqlite3.connect(Path(f"models/index_train_{s}.sqlite").resolve().as_uri() + "?mode=ro", uri=True) for s in ("S2", "S3")}
    WORKER = lgb.Booster(model_file=first_stage), conns


def score_chunk(raw_rows):
    """Retrieval, v3 features, first-stage probability and context features; no labels are read."""
    indexes, _, aliases, _, _ = bsc.STATE
    booster, conns = WORKER
    excluded = aliases.metadata["excluded_inner_fold"]
    features, meta, cleaned = [], [], {}
    for raw in raw_rows:
        e = raw["entity_id"]
        if (entity_fold(e) == "train") != (excluded is not None) or (excluded is not None and inner_fold(e) != excluded):
            raise ValueError("Alias state does not exclude the reference's own supervision")
        reference = normalize_record(raw)
        cleaned[e] = clean_record(reference.name, reference.address)
        for candidate, target, index in bsc.retrieve(reference, indexes):
            values = build_versioned_pair_features(reference, target, candidate, index, FEATURE_VERSION_V3)
            features.append([values[n] for n in FEATURE_NAMES_V3]); meta.append((e, candidate.candidate_entity_id))
    matrix = np.asarray(features, dtype=np.float32).reshape(-1, len(FEATURE_NAMES_V3))
    frame = pd.DataFrame(matrix, columns=FEATURE_NAMES_V3)
    frame["source1_entity_id"] = [m[0] for m in meta]; frame["candidate_entity_id"] = [m[1] for m in meta]
    if not len(frame): return frame
    first = booster.predict(matrix, num_threads=1)
    targets = {}
    for source, conn in conns.items():
        ids = sorted({c for c in frame.candidate_entity_id if c.startswith(source)})
        for start in range(0, len(ids), 500):
            part = ids[start:start + 500]
            for cid, name, address in conn.execute(f"SELECT id,name,address FROM records WHERE id IN ({','.join('?' * len(part))})", part):
                targets[cid] = clean_record(name, address)
    context = np.empty((len(frame), len(CONTEXT_NAMES)), dtype=np.float32)
    for e, positions in frame.groupby("source1_entity_id", sort=False).indices.items():
        context[positions] = pair_context(cleaned[e], targets, frame.candidate_entity_id.iloc[positions].tolist(), first[positions])
    for i, n in enumerate(CONTEXT_NAMES): frame[n] = context[:, i]
    return frame


def stream_universe(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            yield row, normalize_record(row)


def target_keys_for(ids):
    keys = {}
    for source in ("S2", "S3"):
        conn = sqlite3.connect(Path(f"models/index_train_{source}.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
        part_ids = sorted(t for t in ids if t.startswith(source + "-"))
        for i in range(0, len(part_ids), 500):
            part = part_ids[i:i + 500]
            for cid, name, core, address in conn.execute(f"SELECT id,name,core,address FROM records WHERE id IN ({','.join('?' * len(part))})", part):
                keys[cid] = record_keys(name, core, address)
        conn.close()
    return keys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("models/next_round/competition_context_model"))
    p.add_argument("--first-stage", type=Path, default=Path("models/next_round/extended_model"))
    p.add_argument("--cache", type=Path, default=Path("models/next_round/extended_context_cache"))
    p.add_argument("--competition", type=Path, default=Path("models/next_round/competition_v1"))
    p.add_argument("--universe", type=Path, default=Path("dataset/train/train_source1.tsv"))
    p.add_argument("--output", type=Path, default=Path("models/next_round/exclusive_dev"))
    p.add_argument("--report-output", type=Path, default=Path("reports/experiments/round2/exclusive_assignment_dev.json"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--bridge-cache", type=Path, help="Bridge-retrieval cache whose manifest fixes the competitor retrieval")
    args = p.parse_args()
    bridge = json.loads((args.bridge_cache / "manifest.json").read_text())["bridge"] if args.bridge_cache else None
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    log = lambda stage, **kw: print(json.dumps({"stage": stage, **kw, "seconds": round(time.perf_counter() - start, 1)}), flush=True)
    chosen = json.loads((args.model / "report.json").read_text())["protocol_selection"]
    weight, threshold = chosen["context_weight"], chosen["threshold"]
    context_model = lgb.Booster(model_file=str(args.model / "model.txt"))
    if context_model.feature_name() != [*FEATURE_NAMES_V3, *CONTEXT_NAMES, *COMPETITION_NAMES]: raise ValueError("Unexpected context model features")

    manifest = json.loads((args.cache / "manifest.json").read_text()); source = Path(manifest["source_cache"])
    refs, frames = {}, []
    for n in range(manifest["chunks"]):
        stem = f"{n:06d}"
        for row in json.loads((source / (stem + ".json")).read_text())["entity_rows"]: refs[row["entity_id"]] = row
        frames.append(pd.concat([pd.read_parquet(args.cache / (stem + ".parquet"), columns=["source1_entity_id", "candidate_entity_id", "label", "ctx_probability"]),
                                 pd.read_parquet(args.competition / (stem + ".parquet"), columns=list(COMPETITION_NAMES))], axis=1))
    ids = sorted(refs, key=lambda e: hashlib.sha256(("context-split-v1:" + e).encode()).digest())
    context_training, select_ids = set(ids[:30000]), ids[30000:]
    selection = pd.concat(frames, ignore_index=True); del frames
    selection = selection[selection.source1_entity_id.isin(select_ids)].reset_index(drop=True)
    selection["prob"] = (1 - weight) * selection.ctx_probability + weight * np.load(args.model / "selection_probabilities.npy")
    links = selection[selection.prob >= threshold]
    log("selection_loaded", pairs=len(selection), links=len(links))

    wanted_targets = target_keys_for(set(links.candidate_entity_id))
    wanted = {kind: defaultdict(set) for kind in KEY_KINDS}
    for t, keys in wanted_targets.items():
        for kind in KEY_KINDS:
            if keys[kind] is not None: wanted[kind][keys[kind]].add(t)
    claimants, raw_rows = defaultdict(set), {}
    for row, record in stream_universe(args.universe):
        keys = record_keys(record.name, record.core, record.address)
        for kind in KEY_KINDS:
            for t in wanted[kind].get(keys[kind], ()) if keys[kind] is not None else ():
                claimants[t].add(record.entity_id); raw_rows[record.entity_id] = row
    selected = set(select_ids)
    competitors = sorted({b for bs in claimants.values() for b in bs} - selected)
    log("competitors_found", competitors=len(competitors))

    raw_tune = {r["entity_id"]: r for r in json.loads(Path("models/scale_v1_plan/sampled_references.json").read_text())["tune"]}
    replay = sorted(select_ids, key=lambda e: hashlib.sha256(("replay:" + e).encode()).digest())[:250]
    groups = defaultdict(list)
    groups[None].extend(raw_tune[e] for e in replay)
    for e in competitors:
        groups[inner_fold(e) if entity_fold(e) == "train" else None].append(raw_rows[e])
    parts = []
    for fold, rows in sorted(groups.items(), key=lambda kv: -1 if kv[0] is None else kv[0]):
        alias_path = str((ALIASES / ("full.json" if fold is None else f"exclude_{fold}.json")).resolve())
        batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init,
                                 initargs=(alias_path, str(args.first_stage / "model.txt"), bridge)) as pool:
            parts.extend(pool.map(score_chunk, batches))
        log("competitors_scored", alias_fold=fold, references=len(rows))
    scored = pd.concat(parts, ignore_index=True); del parts

    raw_rows.update({e: raw_tune[e] for e in replay})
    reference_keys = {e: record_keys(r.name, r.core, r.address) for e, r in ((e, normalize_record(raw_rows[e])) for e in set(scored.source1_entity_id))}
    target_keys = target_keys_for(set(scored.candidate_entity_id))
    count_wanted = {kind: set() for kind in KEY_KINDS}
    for keys in (*reference_keys.values(), *target_keys.values()):
        for kind in KEY_KINDS:
            if keys[kind] is not None: count_wanted[kind].add(keys[kind])
    counts = count_universe((record_keys(r.name, r.core, r.address) for _, r in stream_universe(args.universe)), count_wanted)
    scored[list(COMPETITION_NAMES)] = np.asarray([competition_features(reference_keys[s], target_keys[c], counts)
        for s, c in zip(scored.source1_entity_id, scored.candidate_entity_id)], dtype=np.float32)
    names = context_model.feature_name()
    scored["prob"] = (1 - weight) * scored.ctx_probability + weight * predict(context_model, scored[names].to_numpy(dtype=np.float32), args.workers)
    replayed = scored[scored.source1_entity_id.isin(set(replay))].merge(
        selection[["source1_entity_id", "candidate_entity_id", "ctx_probability", "prob"]],
        on=["source1_entity_id", "candidate_entity_id"], how="outer", suffixes=("_replay", "_cached"), indicator=True)
    replayed = replayed[replayed.source1_entity_id.isin(set(replay))]
    both = replayed[replayed._merge == "both"]
    replay_check = {"references": len(replay), "pairs_matched": len(both), "pairs_only_replay": int((replayed._merge == "left_only").sum()),
                    "pairs_only_cached": int((replayed._merge == "right_only").sum()),
                    "max_first_stage_difference": float((both.ctx_probability_replay - both.ctx_probability_cached).abs().max()),
                    "max_final_difference": float((both.prob_replay - both.prob_cached).abs().max())}
    log("replay_check", **replay_check)
    scored = scored[~scored.source1_entity_id.isin(selected)]
    scored[["source1_entity_id", "candidate_entity_id", "prob"]].to_parquet(args.output / "competitor_scores.parquet", index=False)
    log("competitor_probabilities", pairs=len(scored))

    first_stage_training = set(json.loads((args.first_stage / "train_references.json").read_text()))
    claims = scored[scored.candidate_entity_id.isin(set(links.candidate_entity_id)) & (scored.prob >= threshold)]
    claims = pd.concat([claims[["source1_entity_id", "candidate_entity_id", "prob"]].assign(
                            in_sample=claims.source1_entity_id.isin(first_stage_training | context_training).to_numpy()),
                        links[["source1_entity_id", "candidate_entity_id", "prob"]].assign(in_sample=False)], ignore_index=True)
    positions = {e: i for i, e in enumerate(select_ids)}
    group = selection.source1_entity_id.map(positions).to_numpy(dtype=np.int32)
    expected = np.array([len(refs[e]["true_ids"]) for e in select_ids])
    countries = np.array([refs[e]["country"] for e in select_ids])
    label = selection.label.to_numpy(dtype=np.uint8)
    base_result, base_values = score(selection.prob.to_numpy(), threshold, label, group, expected, countries)
    report = {"status": "exploratory_selection_only", "model": str(args.model), "context_weight": weight, "threshold": threshold,
              "competitors_scored": len(competitors), "competitor_pairs": len(scored), "replay_check": replay_check,
              "without_exclusivity": base_result}
    rng = np.random.default_rng(42)
    for variant, pool_claims in (("all_competitors", claims), ("out_of_sample_competitors", claims[~claims.in_sample])):
        rivals = defaultdict(list)
        for s, c, pr in zip(pool_claims.source1_entity_id, pool_claims.candidate_entity_id, pool_claims.prob): rivals[c].append((s, pr))
        def loses(row):
            return any(pr > row.prob for s, pr in rivals.get(row.candidate_entity_id, ()) if s != row.source1_entity_id)
        lost = np.zeros(len(selection), dtype=bool)
        link_positions = np.flatnonzero(selection.prob.to_numpy() >= threshold)
        lost[link_positions] = [loses(r) for r in selection.iloc[link_positions].itertuples()]
        adjusted = np.where(lost, 0., selection.prob.to_numpy())
        np.save(args.output / f"adjusted_{variant}.npy", adjusted)
        result, values = score(adjusted, threshold, label, group, expected, countries)
        delta = values - base_values
        interval = np.quantile([rng.choice(delta, len(delta), replace=True).mean() for _ in range(1000)], [.025, .975]).tolist()
        report[variant] = {**result, "links_removed": int(lost.sum()), "true_links_removed": int((lost & (label == 1)).sum()),
                           "false_links_removed": int((lost & (label == 0)).sum()), "gain": float(delta.mean()), "gain_95pct_ci": interval}
        selection[f"lost_{variant}"] = lost
    selection.loc[selection.prob >= threshold].to_parquet(args.output / "selection_links.parquet", index=False)
    report.update({"seconds": time.perf_counter() - start, "competitor_labels_read": False, "fresh_audit_evaluated": False,
                   "submission_generated": False, "caveat": "Competitors are found by exact competition keys only; claimants without a shared key are missed."})
    write_json(args.output / "report.json", report); write_json(args.report_output, report)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
