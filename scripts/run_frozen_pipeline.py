"""Run the frozen bridge pipeline end to end on Source 1 references.

Stages per reference: bridge retrieval, v3 pair features, first stage, candidate
context, Source 1 competition counts and the second stage. Across references:
one owner per target (a link is dropped when another Source 1 record claims the
same target with a strictly higher score), then the rank-one rule.

Commands:
  freeze  write the frozen configuration with hashes of every model and rule source
  replay  rescore development selection references and compare with saved scores
  audit   evaluate once on a reserved audit (competitors found by shared keys)
  test    score every test Source 1 record and write the submission files
"""
import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
import build_scale_cache as bsc
from build_context_features import CONTEXT_NAMES, clean_record, pair_context
from train_scale_model import FEATURE_NAMES_V3, digest, write_json
from src.blocking.disk_index import DiskSearchConfig, normalize_record
from src.blocking.postings_index import PostingsConfig, PostingsSourceIndex
from src.evaluation.validation import entity_fold
from src.features.competition import COMPETITION_NAMES, KEY_KINDS, record_keys
from src.features.pairwise_features import build_versioned_pair_features
from src.features.registry import FEATURE_VERSION_V3
from src.model.crossfit_aliases import inner_fold
from src.model.name_aliases import NameAliases

WORKER = None
ALIASES = Path("models/scale_v1_plan/aliases")
NATIVE = Path("models/runtime/erpostings.dylib")


def log(stage, **values):
    print(json.dumps({"time": time.strftime("%H:%M:%S"), "stage": stage, **values}), flush=True)


def read_tsv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


# ---------------------------------------------------------------- universe counts

def _count_batch(rows):
    counts = {kind: Counter() for kind in KEY_KINDS}
    for row in rows:
        record = normalize_record(row)
        for kind, key in record_keys(record.name, record.core, record.address).items():
            if key is not None: counts[kind][key] += 1
    return counts


def build_universe_counts(universe, path, workers):
    """Every key of every Source 1 record in the split; identical to counting only wanted keys."""
    if path.exists(): return
    rows = read_tsv(universe)
    batches = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]
    total = {kind: Counter() for kind in KEY_KINDS}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for part in pool.map(_count_batch, batches):
            for kind in KEY_KINDS: total[kind].update(part[kind])
    temporary = path.with_suffix(".partial")
    temporary.unlink(missing_ok=True)
    with sqlite3.connect(temporary) as conn:
        conn.execute("CREATE TABLE counts(kind TEXT, key TEXT, n INTEGER, PRIMARY KEY(kind, key)) WITHOUT ROWID")
        for kind in KEY_KINDS:
            conn.executemany("INSERT INTO counts VALUES (?,?,?)", ((kind, k, n) for k, n in total[kind].items()))
        conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('universe', ?)", (json.dumps({"path": str(universe), "sha256": digest(universe), "records": len(rows)}),))
    temporary.replace(path)
    log("universe_counted", records=len(rows))


# ---------------------------------------------------------------- per-reference scoring

def init(config, alias_path, index_prefix, counts_path):
    global WORKER
    os.nice(5); pa.set_cpu_count(1)
    aliases = NameAliases.load(alias_path)
    bridge = config["bridge"]
    ranker = lgb.Booster(model_file=bridge["ranking_model"])
    bsc.BRIDGE = ranker, {k: bridge[k] for k in ("min_probability", "max_seeds", "informative_only")}
    indexes = [PostingsSourceIndex(Path(f"{index_prefix}_{s}.sqlite"), DiskSearchConfig(**config["search"]), aliases=aliases,
                                   postings_config=PostingsConfig(**config["postings"]), native_extension=str(NATIVE.resolve()),
                                   ranking_model=ranker) for s in ("S2", "S3")]
    counts = sqlite3.connect(Path(counts_path).resolve().as_uri() + "?mode=ro", uri=True)
    first = lgb.Booster(model_file=config["first_stage"]["path"])
    second = lgb.Booster(model_file=config["second_stage"]["path"])
    WORKER = indexes, first, second, counts, config, aliases


def lookup_counts(conn, wanted):
    counts = {kind: defaultdict(int) for kind in KEY_KINDS}
    for kind, keys in wanted.items():
        keys = sorted(keys)
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            for key, n in conn.execute(f"SELECT key, n FROM counts WHERE kind=? AND key IN ({','.join('?' * len(part))})", [kind, *part]):
                counts[kind][key] = n
    return counts


def others(counts, kind, key, reference_key):
    if key is None: return -1
    own = int(key == reference_key)
    if counts[kind][key] < own: raise ValueError("The reference must belong to the counted Source 1 universe")
    return counts[kind][key] - own


def score_chunk(task):
    """Scores one batch of references and writes (source1, candidate, probabilities) plus candidate order."""
    number, rows, output = task
    output = Path(output)
    marker = output / f"{number:07d}.json"
    reference_ids = [r["entity_id"] for r in rows]
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["input_ids_sha256"] != hashlib.sha256("\n".join(reference_ids).encode()).hexdigest():
            raise ValueError("Checkpoint belongs to different references")
        return saved
    indexes, first, second, counts_conn, config, aliases = WORKER
    excluded = aliases.metadata["excluded_inner_fold"]
    features, meta, cleaned, keys = [], [], {}, {}
    for raw in rows:
        e = raw["entity_id"]
        if excluded is not None and inner_fold(e) != excluded:
            raise ValueError("Alias state does not exclude the reference's own supervision")
        reference = normalize_record(raw)
        cleaned[e] = clean_record(reference.name, reference.address)
        keys[e] = record_keys(reference.name, reference.core, reference.address)
        for candidate, target, index in bsc.retrieve(reference, indexes):
            values = build_versioned_pair_features(reference, target, candidate, index, FEATURE_VERSION_V3)
            features.append([values[n] for n in FEATURE_NAMES_V3]); meta.append((e, candidate.candidate_entity_id))
    matrix = np.asarray(features, dtype=np.float32).reshape(-1, len(FEATURE_NAMES_V3))
    frame = pd.DataFrame({"source1_entity_id": [m[0] for m in meta], "candidate_entity_id": [m[1] for m in meta]})
    if len(frame):
        p1 = first.predict(matrix, num_threads=1)
        targets, target_keys = {}, {}
        for index in indexes:
            candidate_ids = sorted({c for c in frame.candidate_entity_id if c.startswith(index.source)})
            for i in range(0, len(candidate_ids), 500):
                part = candidate_ids[i:i + 500]
                for cid, name, core, address in index.connection.execute(
                        f"SELECT id,name,core,address FROM records WHERE id IN ({','.join('?' * len(part))})", part):
                    targets[cid] = clean_record(name, address); target_keys[cid] = record_keys(name, core, address)
        context = np.empty((len(frame), len(CONTEXT_NAMES)), dtype=np.float32)
        for e, positions in frame.groupby("source1_entity_id", sort=False).indices.items():
            context[positions] = pair_context(cleaned[e], targets, frame.candidate_entity_id.iloc[positions].tolist(), p1[positions])
        wanted = {kind: set() for kind in KEY_KINDS}
        for k in (*keys.values(), *target_keys.values()):
            for kind in KEY_KINDS:
                if k[kind] is not None: wanted[kind].add(k[kind])
        counts = lookup_counts(counts_conn, wanted)
        competition = np.asarray([[others(counts, "core", target_keys[c]["core"], keys[s]["core"]),
            others(counts, "core_set", target_keys[c]["core_set"], keys[s]["core_set"]),
            others(counts, "phonetic_set", target_keys[c]["phonetic_set"], keys[s]["phonetic_set"]),
            others(counts, "core_number", target_keys[c]["core_number"], keys[s]["core_number"]),
            others(counts, "address_set", target_keys[c]["address_set"], keys[s]["address_set"]),
            others(counts, "core", keys[s]["core"], keys[s]["core"]),
            others(counts, "address_set", keys[s]["address_set"], keys[s]["address_set"])]
            for s, c in zip(frame.source1_entity_id, frame.candidate_entity_id)], dtype=np.float32)
        full = np.hstack([matrix, context, competition])
        if not np.isfinite(full).all(): raise ValueError("Nonfinite features")
        p2 = second.predict(full, num_threads=1)
        weight = config["second_stage"]["context_weight"]
        frame["first_stage"] = p1
        frame["second_stage"] = p2
        frame["probability"] = (1 - weight) * p1 + weight * p2
    else:
        frame["first_stage"] = frame["second_stage"] = frame["probability"] = np.zeros(0)
    path = output / f"{number:07d}.parquet"
    temporary = path.with_suffix(".partial.parquet")
    frame.to_parquet(temporary, index=False); temporary.replace(path)
    result = {"chunk": number, "entities": len(rows), "pairs": len(frame),
              "input_ids_sha256": hashlib.sha256("\n".join(reference_ids).encode()).hexdigest(), "parquet_sha256": digest(path)}
    write_json(marker, result)
    return result


def score_references(config, groups, index_prefix, counts_path, output, workers, batch_size=250):
    """groups: alias path -> reference rows. Chunks are resumable and verified on reuse."""
    output.mkdir(parents=True, exist_ok=True)
    number, results = 0, []
    for alias_path, rows in groups.items():
        tasks = []
        for i in range(0, len(rows), batch_size):
            tasks.append((number, rows[i:i + batch_size], str(output))); number += 1
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=init,
                                 initargs=(config, str(alias_path), index_prefix, str(counts_path))) as pool:
            done = 0
            for result in pool.map(score_chunk, tasks):
                results.append(result); done += result["entities"]
                if done % 25000 < batch_size: log("scored", alias=Path(alias_path).name, entities=done, of=len(rows))
    frames = []
    for result in sorted(results, key=lambda r: r["chunk"]):
        path = output / f"{result['chunk']:07d}.parquet"
        if digest(path) != result["parquet_sha256"]: raise ValueError(f"Corrupt chunk {path}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- decision rules

def decide(frame, claims, config):
    """frame: rows to decide, in candidate order per reference. claims: every scored (reference, target, probability)."""
    threshold, t_first, t_rest = config["threshold"], config["t_first"], config["t_rest"]
    strong = claims[claims.probability >= threshold].sort_values("probability", ascending=False, kind="stable")
    top = strong.groupby("candidate_entity_id", sort=False).head(2)
    best = {}
    for c, s, p in zip(top.candidate_entity_id, top.source1_entity_id, top.probability):
        best.setdefault(c, []).append((s, p))
    probability = frame.probability.to_numpy()
    lost = np.zeros(len(frame), dtype=bool)
    for i in np.flatnonzero(probability >= threshold):
        s, c = frame.source1_entity_id.iat[i], frame.candidate_entity_id.iat[i]
        rival = max((p for owner, p in best.get(c, ()) if owner != s), default=-1.)
        lost[i] = rival > probability[i]
    adjusted = np.where(lost, 0., probability)
    codes, _ = pd.factorize(frame.source1_entity_id)
    order = np.lexsort((-adjusted, codes))
    first = np.zeros(len(frame), dtype=bool)
    first[order[np.r_[True, codes[order][1:] != codes[order][:-1]]]] = True
    chosen = (adjusted >= t_rest) | (first & (adjusted >= t_first))
    return chosen, lost


def macro(frame, chosen, truth, ids):
    predicted = defaultdict(set)
    for s, c, keep in zip(frame.source1_entity_id, frame.candidate_entity_id, chosen):
        if keep: predicted[s].add(c)
    values = []
    for e in ids:
        actual, pred = truth[e], predicted.get(e, set())
        if not actual and not pred: values.append(1.); continue
        tp = len(actual & pred)
        values.append(1.25 * tp / (1.25 * tp + (len(pred) - tp) + .25 * (len(actual) - tp)) if tp else 0.)
    values = np.asarray(values)
    tp = sum(len(truth[e] & predicted.get(e, set())) for e in ids)
    n_pred = sum(len(predicted.get(e, ())) for e in ids)
    return values, {"macro_f05": float(values.mean()), "entities": len(ids), "predicted_links": n_pred, "true_positive_links": tp,
                    "micro_precision": tp / max(1, n_pred), "micro_recall": tp / max(1, sum(len(truth[e]) for e in ids)),
                    "singleton_false_positives": sum(1 for e in ids if not truth[e] and predicted.get(e)),
                    "non_singleton_empty_predictions": sum(1 for e in ids if truth[e] and not predicted.get(e))}


def interval(delta, seed=42):
    rng = np.random.default_rng(seed)
    return np.quantile([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)], [.025, .975]).tolist()


# ---------------------------------------------------------------- commands

def load_frozen(path):
    config = json.loads(Path(path).read_text())
    if config["status"] != "frozen_before_audit": raise ValueError("Pipeline is not frozen")
    for part in ("first_stage", "second_stage"):
        if digest(config[part]["path"]) != config[part]["sha256"]: raise ValueError(f"Frozen {part} changed")
    if digest(config["bridge"]["ranking_model"]) != config["bridge"]["ranking_model_sha256"]: raise ValueError("Ranker changed")
    if digest(config["aliases"]["path"]) != config["aliases"]["sha256"]: raise ValueError("Aliases changed")
    for path, expected in config["code_sha256"].items():
        if digest(path) != expected: raise ValueError(f"Frozen code changed: {path}")
    return config


def freeze(args):
    if args.output.exists(): raise FileExistsError(args.output)
    stage2 = json.loads(Path("models/bridge_v1/stacked_l255s42/report.json").read_text())["protocol_selection"]
    rules_path = Path("reports/experiments/round3/bridge_decision_rules_exclusive_stacked_l255s42.json")
    rules = json.loads(rules_path.read_text())
    first_report = json.loads(Path("models/bridge_v1/first_stage/report.json").read_text())
    ranker = Path("models/redesign/cheap_ranker_v1/model.txt")
    root = Path(__file__).resolve().parents[1]
    code = [Path(__file__), Path("scripts/build_scale_cache.py"), Path("scripts/build_context_features.py"),
            *sorted((root / "code/business_entity_resolution/src").rglob("*.py"))]
    config = {"status": "frozen_before_audit", "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "first_stage": {"path": "models/bridge_v1/first_stage/model.txt", "sha256": digest("models/bridge_v1/first_stage/model.txt"),
                        "tuned_threshold": first_report["chosen"]["threshold"]},
        "second_stage": {"path": "models/bridge_v1/stacked_l255s42/model.txt", "sha256": digest("models/bridge_v1/stacked_l255s42/model.txt"),
                         "context_weight": stage2["context_weight"]},
        "threshold": stage2["threshold"], "t_first": rules["rank_one_rule"]["t_first"], "t_rest": rules["rank_one_rule"]["t_rest"],
        "rules_source": {"path": str(rules_path), "sha256": digest(rules_path)},
        "search": first_report["search_config"], "postings": first_report["retrieval"]["options"],
        "bridge": {"ranking_model": str(ranker), "ranking_model_sha256": digest(ranker), "min_probability": .99, "max_seeds": 2, "informative_only": False},
        "aliases": {"path": str(ALIASES / "full.json"), "sha256": digest(ALIASES / "full.json")},
        "feature_version": FEATURE_VERSION_V3,
        "code_sha256": {str(p.relative_to(root) if p.is_absolute() else p): digest(p) for p in code},
        "selection_evidence": "Chosen on the 20k development selection references (local 0.97982); not yet audited."}
    if stage2["threshold"] != rules["single_threshold"]["threshold"]: raise ValueError("Rule source uses another threshold")
    args.output.mkdir(parents=True)
    write_json(args.output / "frozen.json", config)
    print(json.dumps({k: config[k] for k in ("threshold", "t_first", "t_rest")}))


def replay(args):
    """Rescore development selection references with train indexes; the second-stage scores must match the saved ones."""
    config = load_frozen(args.frozen)
    tune = json.loads(Path("models/scale_v1_plan/sampled_references.json").read_text())["tune"]
    order = sorted(tune, key=lambda r: hashlib.sha256(r["entity_id"].encode()).digest())
    split = sorted((r["entity_id"] for r in tune), key=lambda e: hashlib.sha256(("context-split-v1:" + e).encode()).digest())
    selected = set(split[30000:])
    selection_rows = [r for r in order if r["entity_id"] in selected]
    saved = np.load(Path(config["second_stage"]["path"]).parent / "selection_probabilities.npy")
    if len(saved) != 40 * len(selection_rows): raise ValueError("Saved selection rows are not 40 per reference")
    chosen = selection_rows[:args.references]
    counts = args.output / "universe_counts_train.sqlite"
    args.output.mkdir(parents=True, exist_ok=True)
    build_universe_counts(Path("dataset/train/train_source1.tsv"), counts, args.workers)
    frame = score_references(config, {config["aliases"]["path"]: chosen}, "models/index_train", counts, args.output / "chunks", args.workers)
    expected = saved[:40 * len(chosen)]
    result = {"references": len(chosen), "pairs": len(frame), "expected_pairs": len(expected)}
    if len(frame) == len(expected):
        result["max_second_stage_difference"] = float(np.abs(frame.second_stage.to_numpy() - expected).max())
    write_json(args.output / "replay.json", result)
    print(json.dumps(result))


def truth_for(ids):
    conn = sqlite3.connect((ALIASES / "counts.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    truth = {e: set() for e in ids}
    ids = sorted(ids)
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        for target, owner in conn.execute(f"SELECT id, owner FROM owners WHERE owner IN ({','.join('?' * len(part))})", part):
            truth[owner].add(target)
    return truth


def audit(args):
    config = load_frozen(args.frozen)
    report_path = args.output / "report.json"
    if report_path.exists(): raise FileExistsError("Audit already evaluated; do not tune against it")
    rows = json.loads((args.audit / "sampled_references.json").read_text())["holdout"]
    if any(entity_fold(r["entity_id"]) != "holdout" for r in rows): raise ValueError("Audit references must be outer holdout")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "audit_started.json", {"frozen_sha256": digest(args.frozen), "audit_plan_sha256": digest(args.audit / "plan.json")})
    universe = Path("dataset/train/train_source1.tsv")
    counts = args.output / "universe_counts_train.sqlite"
    build_universe_counts(universe, counts, args.workers)
    frame = score_references(config, {config["aliases"]["path"]: rows}, "models/index_train", counts, args.output / "chunks", args.workers)
    log("references_scored", pairs=len(frame))

    # Competitors: other Source 1 records sharing a competition key with a target these references would link.
    links = frame[frame.probability >= config["threshold"]]
    wanted = {kind: defaultdict(set) for kind in KEY_KINDS}
    conns = {s: sqlite3.connect(Path(f"models/index_train_{s}.sqlite").resolve().as_uri() + "?mode=ro", uri=True) for s in ("S2", "S3")}
    targets = sorted(set(links.candidate_entity_id))
    for s, conn in conns.items():
        part_ids = [t for t in targets if t.startswith(s + "-")]
        for i in range(0, len(part_ids), 500):
            part = part_ids[i:i + 500]
            for cid, name, core, address in conn.execute(f"SELECT id,name,core,address FROM records WHERE id IN ({','.join('?' * len(part))})", part):
                for kind, key in record_keys(name, core, address).items():
                    if key is not None: wanted[kind][key].add(cid)
    audit_ids = {r["entity_id"] for r in rows}
    competitors = {}
    for row in read_tsv(universe):
        if row["entity_id"] in audit_ids: continue
        record = normalize_record(row)
        k = record_keys(record.name, record.core, record.address)
        if any(k[kind] is not None and k[kind] in wanted[kind] for kind in KEY_KINDS): competitors[row["entity_id"]] = row
    groups = defaultdict(list)
    for e, row in sorted(competitors.items()):
        groups[ALIASES / (f"exclude_{inner_fold(e)}.json" if entity_fold(e) == "train" else "full.json")].append(row)
    log("competitors_found", competitors=len(competitors))
    rivals = score_references(config, dict(sorted(groups.items(), key=lambda kv: str(kv[0]))), "models/index_train", counts,
                              args.output / "competitor_chunks", args.workers)
    claims = pd.concat([frame, rivals], ignore_index=True)[["source1_entity_id", "candidate_entity_id", "probability"]]
    chosen, lost = decide(frame, claims, config)

    ids = sorted(audit_ids)
    truth = truth_for(ids)
    final_values, final = macro(frame, chosen, truth, ids)
    first_values, first = macro(frame, frame.first_stage.to_numpy() >= config["first_stage"]["tuned_threshold"], truth, ids)
    single_values, single = macro(frame, frame.probability.to_numpy() >= config["threshold"], truth, ids)
    countries = {r["entity_id"]: r["country"] for r in rows}
    by_country = {c: float(np.mean([v for e, v in zip(ids, final_values) if countries[e] == c])) for c in sorted(set(countries.values()))}
    report = {"status": "fresh_audit_complete", "frozen_sha256": digest(args.frozen), "audit_entities": len(ids),
              "final_pipeline": {**final, "country_macro_f05": by_country, "links_removed_by_exclusivity": int(lost.sum())},
              "second_stage_single_threshold": single, "first_stage_only": first,
              "gain_vs_first_stage": float((final_values - first_values).mean()), "gain_vs_first_stage_95pct_ci": interval(final_values - first_values),
              "competitors_scored": len(competitors), "competitor_labels_read": False,
              "france": "No labeled France records; country generalization remains unmeasured."}
    baseline_path = args.audit / "submission03_predictions.json"
    if baseline_path.exists():
        saved = json.loads(baseline_path.read_text())
        base = pd.DataFrame([(e, c) for e, v in saved.items() for c in v["predictions"]], columns=["source1_entity_id", "candidate_entity_id"])
        base_values, base_result = macro(base, np.ones(len(base), dtype=bool), truth, ids)
        delta = final_values - base_values
        ci = interval(delta)
        report["submission_03_pipeline"] = {**base_result, "country_macro_f05": {c: float(np.mean([v for e, v in zip(ids, base_values) if countries[e] == c]))
                                                                                 for c in sorted(set(countries.values()))}}
        report["gain_vs_submission_03"] = float(delta.mean())
        report["gain_vs_submission_03_95pct_ci"] = ci
        report["ship_decision"] = {"accepted": bool(ci[0] > 0 and final["micro_precision"] >= base_result["micro_precision"] - .002),
                                   "rule": "Paired gain lower bound > 0 and precision drop <= .002 against submission_03 on the same audit."}
    np.save(args.output / "final_entity_scores.npy", final_values)
    write_json(args.output / "entity_ids.json", ids)
    write_json(report_path, report)
    print(json.dumps(report, indent=1))


def test(args):
    config = load_frozen(args.frozen)
    test_dir = args.test_dir
    rows = read_tsv(test_dir / "test_source1.tsv")
    args.output.mkdir(parents=True, exist_ok=True)
    counts = args.output / "universe_counts_test.sqlite"
    build_universe_counts(test_dir / "test_source1.tsv", counts, args.workers)
    frame = score_references(config, {config["aliases"]["path"]: rows}, "models/index_test", counts, args.output / "chunks", args.workers)
    log("test_scored", pairs=len(frame))
    chosen, lost = decide(frame, frame, config)
    candidates, predictions = defaultdict(list), defaultdict(list)
    for s, c, keep in zip(frame.source1_entity_id, frame.candidate_entity_id, chosen):
        candidates[s].append(c)
        if keep: predictions[s].append(c)
    for name, header, values in (("matching_results.tsv", "source1_entity_id\tmatched_entity_ids\n", predictions),
                                 ("candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids\n", candidates)):
        temporary = args.output / (name + ".partial")
        with temporary.open("w") as out:
            out.write(header)
            for row in rows:
                out.write(row["entity_id"] + "\t" + ",".join(values.get(row["entity_id"], ())) + "\n")
        temporary.replace(args.output / name)
    from src.pipeline.export import validate_submission
    matching, candidate = args.output / "matching_results.tsv", args.output / "candidate_pairs.tsv"
    strict = validate_submission(matching, candidate, test_dir)
    official = Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src/utils/organizer_validate_submission.py"
    with (args.output / "official_validation.log").open("w") as stream:
        returncode = subprocess.run([sys.executable, "-S", str(official), "--matching", str(matching), "--candidate", str(candidate),
                                     "--test-dir", str(test_dir), "--check-ids"], stdout=stream, stderr=subprocess.STDOUT).returncode
    country = Counter(); links = Counter(); empty = Counter()
    for row in rows:
        c = row["country"].strip().casefold(); country[c] += 1
        links[c] += len(predictions.get(row["entity_id"], ())); empty[c] += not predictions.get(row["entity_id"])
    summary = {"status": "complete" if not strict and not returncode else "validation_failed",
               "validation": {"strict_issues": strict[:20], "official_returncode": returncode},
               "entities": len(rows), "scored_candidates": len(frame),
               "predicted_links": int(chosen.sum()), "links_removed_by_exclusivity": int(lost.sum()),
               "empty_predictions": sum(empty.values()), "frozen_sha256": digest(args.frozen),
               "country": {c: {"entities": country[c], "predicted_links": links[c], "empty_predictions": empty[c]} for c in sorted(country)},
               "output_sha256": {n: digest(args.output / n) for n in ("matching_results.tsv", "candidate_pairs.tsv")}}
    write_json(args.output / "submission_report.json", summary)
    print(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--output", type=Path, default=Path("models/frozen_v4"))
    for name in ("replay", "audit", "test"):
        s = sub.add_parser(name)
        s.add_argument("--frozen", type=Path, default=Path("models/frozen_v4/frozen.json"))
        s.add_argument("--workers", type=int, default=60)
        if name == "replay":
            s.add_argument("--references", type=int, default=250)
            s.add_argument("--output", type=Path, default=Path("models/frozen_v4/replay"))
        if name == "audit":
            s.add_argument("--audit", type=Path, default=Path("models/audit_v2"))
            s.add_argument("--output", type=Path, default=Path("models/audit_v2/run"))
        if name == "test":
            s.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
            s.add_argument("--output", type=Path, default=Path("output/submission_04"))
    args = p.parse_args()
    {"freeze": freeze, "replay": replay, "audit": audit, "test": test}[args.command](args)


if __name__ == "__main__":
    main()
