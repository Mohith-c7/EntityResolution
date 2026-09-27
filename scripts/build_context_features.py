"""Build candidate-context evidence from out-of-sample tuning predictions only."""
import argparse
import hashlib
import json
import multiprocessing
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.disk_index import normalize_record
from src.preprocessing.normalize import accent_fold
from train_scale_model import digest, write_json

STATE = None
CONTEXT_NAMES = ["ctx_probability", "ctx_logit", "ctx_rank", "ctx_source_rank", "ctx_best_other_probability",
    "ctx_strong_peers", "ctx_opposite_source_strong_peers", "ctx_peer_name_set", "ctx_peer_name_sort",
    "ctx_peer_address_set", "ctx_peer_address_sort", "ctx_peer_number_jaccard", "ctx_peer_name_at_best_address",
    "ctx_peer_address_at_best_name", "independent_name_set", "independent_name_sort", "independent_name_ratio",
    "independent_name_missing", "reference_name_address_overlap", "candidate_name_address_overlap"]


def clean_record(name, address):
    name, address = accent_fold(name), accent_fold(address)
    nt, at = set(name.split()), set(address.split())
    independent = " ".join(t for t in name.split() if t not in at)
    digits = {t.lstrip("0") or "0" for t in at if t.isdecimal()}
    return name, address, independent, digits, len(nt & at) / max(1, len(nt))


def pair_context(reference, targets, ids, probabilities):
    """No labels enter peer selection or feature construction."""
    records = [targets[c] for c in ids]
    order = sorted(range(len(ids)), key=lambda i: (-probabilities[i], ids[i]))
    rank = {i: n+1 for n,i in enumerate(order)}
    source_rank = {}
    for source in ("S2", "S3"):
        source_rank.update({i: n+1 for n,i in enumerate(i for i in order if ids[i].startswith(source))})
    strong = [i for i in order if probabilities[i] >= .95]
    rows = []
    for i, record in enumerate(records):
        peers = [j for j in strong if j != i][:4]
        evidence = []
        for j in peers:
            peer = records[j]
            ns = fuzz.token_set_ratio(record[0], peer[0])/100 if record[0] and peer[0] else 0.
            no = fuzz.token_sort_ratio(record[0], peer[0])/100 if record[0] and peer[0] else 0.
            ads = fuzz.token_set_ratio(record[1], peer[1])/100 if record[1] and peer[1] else 0.
            ado = fuzz.token_sort_ratio(record[1], peer[1])/100 if record[1] and peer[1] else 0.
            numeric = len(record[3] & peer[3])/len(record[3] | peer[3]) if record[3] and peer[3] else 0.
            evidence.append((ns, no, ads, ado, numeric))
        p = float(np.clip(probabilities[i], 1e-6, 1-1e-6))
        rest = [probabilities[j] for j in order[:2] if j != i]
        independent_left, independent_right = reference[2], record[2]
        rows.append([p, np.log(p/(1-p)), rank[i], source_rank[i], max(rest, default=0.),
            len(strong) - int(i in strong), sum(ids[j][:2] != ids[i][:2] for j in strong),
            *[max((e[k] for e in evidence), default=0.) for k in range(5)],
            max(evidence, key=lambda e: (e[2], e[0]))[0] if evidence else 0.,
            max(evidence, key=lambda e: (e[0], e[2]))[2] if evidence else 0.,
            fuzz.token_set_ratio(independent_left, independent_right)/100 if independent_left and independent_right else 0.,
            fuzz.token_sort_ratio(independent_left, independent_right)/100 if independent_left and independent_right else 0.,
            fuzz.ratio(independent_left, independent_right)/100 if independent_left and independent_right else 0.,
            not bool(independent_left and independent_right), reference[4], record[4]])
    return np.asarray(rows, dtype=np.float32).reshape(-1, len(CONTEXT_NAMES))


def init(cache, probability_path, output, refs_path, split="tune"):
    global STATE
    os.nice(5); pa.set_cpu_count(1)
    conns = {s: sqlite3.connect(Path(f"models/index_train_{s}.sqlite").resolve().as_uri()+"?mode=ro", uri=True) for s in ("S2", "S3")}
    for conn in conns.values(): conn.execute("PRAGMA mmap_size=2147418112")
    raw_refs = json.loads(Path(refs_path).read_text())[split]
    refs = {}
    for r in raw_refs:
        n = normalize_record(r); refs[n.entity_id] = clean_record(n.name, n.address)
    STATE = Path(cache), np.load(probability_path, mmap_mode="r"), Path(output), conns, refs


def chunk(task):
    number, offset, length = task
    cache, probabilities, output, conns, refs = STATE
    path = cache / f"{number:06d}.parquet"
    marker = json.loads((cache / f"{number:06d}.json").read_text())
    if digest(path) != marker["parquet_sha256"]: raise ValueError("Source cache corrupt")
    frame = pd.read_parquet(path)
    if len(frame) != length: raise ValueError("Prediction/cache alignment mismatch")
    prob = probabilities[offset:offset+length]
    targets = {}
    for source, conn in conns.items():
        ids = sorted(set(frame.loc[frame.candidate_entity_id.str.startswith(source), "candidate_entity_id"]))
        for start in range(0, len(ids), 500):
            part = ids[start:start+500]; marks = ",".join("?" for _ in part)
            for cid,name,address in conn.execute(f"SELECT id,name,address FROM records WHERE id IN ({marks})", part):
                targets[cid] = clean_record(name,address)
    matrix = np.empty((len(frame), len(CONTEXT_NAMES)), dtype=np.float32)
    for eid, positions in frame.groupby("source1_entity_id", sort=False).indices.items():
        ids = frame.iloc[positions].candidate_entity_id.tolist()
        matrix[positions] = pair_context(refs[eid], targets, ids, prob[positions])
    if not np.isfinite(matrix).all(): raise ValueError("Nonfinite context features")
    for i, name in enumerate(CONTEXT_NAMES): frame[name] = matrix[:,i]
    dest = output / path.name; temp = dest.with_suffix(".partial")
    frame.to_parquet(temp, index=False); temp.replace(dest)
    write_json(output / f"{number:06d}.json", {"entities": marker["entities"], "pairs": len(frame), "sha256": digest(dest)})
    return marker["entities"], len(frame)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("models/next_round/context_cache"))
    p.add_argument("--cache", type=Path, default=Path("models/scale_v1_run/cache_tune"))
    p.add_argument("--first-stage-model", type=Path, default=Path("models/scale_v1_run/model_300k"))
    p.add_argument("--references", type=Path, default=Path("models/scale_v1_plan/sampled_references.json"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", choices=("tune", "train"), default="tune")
    p.add_argument("--probabilities", type=Path, help="Out-of-fold first-stage scores; required for the training split")
    args=p.parse_args()
    if args.split == "train" and not args.probabilities: p.error("Training references need out-of-fold scores")
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cache=args.cache; model=args.first_stage_model
    protocol=json.loads((model/"protocol.json").read_text())
    if digest(cache/"manifest.json") != protocol[f"{args.split}_manifest_sha256"]: raise ValueError("Source cache changed")
    progress=json.loads((cache/"progress.json").read_text())
    if progress["status"] != "complete": raise ValueError("Source cache incomplete")
    tasks=[];offset=0
    for number in range(progress["chunks"]):
        m=json.loads((cache/f"{number:06d}.json").read_text());tasks.append((number,offset,m["pairs"]));offset+=m["pairs"]
    probability_path=args.probabilities or model/"probabilities_tune.npy"
    if np.load(probability_path,mmap_mode="r").shape != (offset,): raise ValueError("Probability length mismatch")
    metadata={"status":"building","feature_names":CONTEXT_NAMES,"source_cache":str(cache),
        "probabilities_sha256":digest(probability_path),"source_manifest_sha256":digest(cache/"manifest.json"),
        "model_sha256":digest(model/"model.txt"),"source_script_sha256":digest(Path(__file__)),
        "split":"outer_tune_only" if args.split == "tune" else "outer_train_out_of_fold",
        "fresh_audit_evaluated":False,"submission_generated":False}
    write_json(args.output/"manifest.json",metadata)
    entities=pairs=0
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn"),initializer=init,
        initargs=(str(cache),str(probability_path),str(args.output),str(args.references),args.split)) as pool:
        for ne,npairs in pool.map(chunk,tasks):
            entities+=ne;pairs+=npairs
            if entities%5000==0: print(json.dumps({"entities":entities,"pairs":pairs}),flush=True)
    write_json(args.output/"manifest.json",{**metadata,"status":"complete","entities":entities,"pairs":pairs,"chunks":len(tasks)})


if __name__=="__main__":main()
