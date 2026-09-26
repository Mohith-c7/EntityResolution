"""Evaluate changed blocking with a frozen context model on its disjoint selection entities."""
import argparse
import hashlib
import json
import multiprocessing
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

from build_context_features import CONTEXT_NAMES, clean_record, pair_context, normalize_record
from train_scale_model import score, write_json, digest

STATE=None


def init(adapter,context_model):
    global STATE
    pa.set_cpu_count(1)
    refs={}
    for raw in json.loads((Path(adapter)/"sampled_references.json").read_text())["tune"]:
        ref=normalize_record(raw);refs[ref.entity_id]=clean_record(ref.name,ref.address)
    conns={s:sqlite3.connect(Path(f"models/index_train_{s}.sqlite").resolve().as_uri()+"?mode=ro",uri=True) for s in ("S2","S3")}
    for conn in conns.values():conn.execute("PRAGMA mmap_size=2147418112")
    model=lgb.Booster(model_file=str(Path(context_model)/"model.txt"))
    STATE=refs,conns,model


def chunk(path):
    refs,conns,model=STATE
    frame=pd.read_parquet(path);probabilities=frame.probability.to_numpy();targets={}
    for source,conn in conns.items():
        ids=sorted(set(frame.loc[frame.candidate_entity_id.str.startswith(source),"candidate_entity_id"]))
        for start in range(0,len(ids),500):
            part=ids[start:start+500];marks=",".join("?" for _ in part)
            for cid,name,address in conn.execute(f"SELECT id,name,address FROM records WHERE id IN ({marks})",part):
                targets[cid]=clean_record(name,address)
    matrix=np.empty((len(frame),len(CONTEXT_NAMES)),dtype=np.float32)
    for eid,positions in frame.groupby("source1_entity_id",sort=False).indices.items():
        matrix[positions]=pair_context(refs[eid],targets,frame.iloc[positions].candidate_entity_id.tolist(),probabilities[positions])
    for i,name in enumerate(CONTEXT_NAMES):frame[name]=matrix[:,i]
    values=frame[model.feature_name()].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():raise ValueError("Invalid context features")
    result=frame[["source1_entity_id","candidate_entity_id"]].copy()
    result["base_probability"]=matrix[:,0]
    result["context_probability"]=model.predict(values,num_threads=1)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=Path("models/next_round/bridge_selection_cache"))
    parser.add_argument("--adapter",type=Path,default=Path("models/next_round/context_selection_adapter"))
    parser.add_argument("--context-model",type=Path,default=Path("models/next_round/context_model"))
    parser.add_argument("--output",type=Path,default=Path("models/next_round/combined_context"))
    args=parser.parse_args()
    manifest=json.loads((args.cache/"manifest.json").read_text())
    if not manifest["complete"] or manifest["test"]:raise ValueError("Completed selection cache required")
    split=json.loads((args.context_model/"sampled_references.json").read_text())
    ids=[r["entity_id"] for r in split["stage2_select"]]
    if set(ids)!=set(manifest["reference_ids"]):raise ValueError("Candidate cache must contain only the entire context selection split")
    protocol=json.loads((args.context_model/"protocol.json").read_text())
    if manifest["model_sha256"]!=protocol["first_stage_model_sha256"]:raise ValueError("Wrong first-stage scores")
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn"),initializer=init,
        initargs=(str(args.adapter),str(args.context_model))) as pool:
        frames=list(pool.map(chunk,[str(args.cache/f) for f in manifest["files"]]))
    frame=pd.concat(frames,ignore_index=True)
    if len(frame)!=manifest["rows_scored"] or frame.duplicated(["source1_entity_id","candidate_entity_id"]).any():raise ValueError("Pair count mismatch")
    truth={e:set(v) for e,v in json.loads((args.adapter/"sampled_truth.json").read_text()).items()}
    refs={r["entity_id"]:r for r in json.loads((args.adapter/"sampled_references.json").read_text())["tune"]}
    positions={e:i for i,e in enumerate(ids)};groups=frame.source1_entity_id.map(positions).to_numpy(dtype=np.int32)
    label=np.array([c in truth[e] for e,c in frame[["source1_entity_id","candidate_entity_id"]].itertuples(index=False,name=None)],dtype=np.uint8)
    expected=np.array([len(truth[e]) for e in ids]);countries=np.array([refs[e]["country"].casefold() for e in ids])
    previous=json.loads((args.context_model/"report.json").read_text());frozen=previous["selection"]
    probability=(1-frozen["context_weight"])*frame.base_probability.to_numpy()+frozen["context_weight"]*frame.context_probability.to_numpy()
    fixed=score(probability,frozen["threshold"],label,groups,expected,countries)[0]
    trials=[]
    for weight in (0.,.25,.5,.75,1.):
        values=(1-weight)*frame.base_probability.to_numpy()+weight*frame.context_probability.to_numpy()
        for threshold in np.r_[np.arange(.50,.951,.01),.96,.97,.98,.99]:
            result,_=score(values,round(float(threshold),6),label,groups,expected,countries)
            trials.append({"context_weight":weight,**result})
    eligible=[r for r in trials if r["micro_precision"]>=frozen["micro_precision"]-.002 and r["singleton_false_positives"]<=frozen["singleton_false_positives"]]
    selected=max(eligible,key=lambda r:r["macro_f05"]) if eligible else None
    report={"status":"tuning_only_requires_new_audit","baseline_context_original_retrieval":frozen,
        "same_context_rule_new_retrieval":fixed,"selection":selected,
        "macro_gain":selected["macro_f05"]-frozen["macro_f05"] if selected else None,
        "oracle":score(label,.5,label,groups,expected,countries)[0],
        "model_sha256":digest(args.context_model/"model.txt"),"candidate_manifest_sha256":digest(args.cache/"manifest.json"),
        "fresh_audit_evaluated":False,"submission_generated":False}
    frame["label"]=label;frame.to_parquet(args.output/"predictions.parquet",index=False)
    write_json(args.output/"report.json",report);write_json(args.output/"threshold_sweep.json",trials)
    write_json(Path("reports/experiments/round2/combined_context.json"),report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
