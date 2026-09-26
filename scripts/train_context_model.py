"""Train a second-stage matcher on disjoint references scored by the frozen first stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

from build_context_features import CONTEXT_NAMES
from train_scale_model import FEATURE_NAMES_V3, digest, write_json, score, predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=Path("models/next_round/context_cache"))
    p.add_argument("--output",type=Path,default=Path("models/next_round/context_model"))
    p.add_argument("--threads",type=int,default=8)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);os.nice(5);pa.set_cpu_count(1)
    start=time.perf_counter();manifest=json.loads((args.cache/"manifest.json").read_text())
    if manifest["status"]!="complete" or manifest["split"]!="outer_tune_only":raise ValueError("Wrong context cache")
    names=[*FEATURE_NAMES_V3,*CONTEXT_NAMES];source=Path(manifest["source_cache"])
    refs={};frames=[]
    for number in range(manifest["chunks"]):
        stem=f"{number:06d}";path=args.cache/(stem+".parquet")
        meta=json.loads((args.cache/(stem+".json")).read_text())
        if digest(path)!=meta["sha256"]:raise ValueError("Corrupt context cache")
        for row in json.loads((source/(stem+".json")).read_text())["entity_rows"]:refs[row["entity_id"]]=row
        frames.append(pd.read_parquet(path,columns=[*names,"label","sample_weight","source1_entity_id"]))
    frame=pd.concat(frames,ignore_index=True);del frames
    ids=sorted(refs,key=lambda e:hashlib.sha256(("context-split-v1:"+e).encode()).digest())
    if len(ids)!=50000:raise ValueError("Expected the fixed 50k development set")
    train_ids=ids[:20000];early_ids=ids[20000:30000];select_ids=ids[30000:]
    if set(train_ids)&set(early_ids) or set(train_ids)&set(select_ids):raise ValueError("Stage-two split overlap")
    write_json(args.output/"sampled_references.json",{k:[{"entity_id":e} for e in values]
        for k,values in (("stage2_train",train_ids),("stage2_earlystop",early_ids),("stage2_select",select_ids))})
    protocol={"first_stage_model_sha256":manifest["model_sha256"],"cache_manifest_sha256":digest(args.cache/"manifest.json"),
        "feature_names":names,"stage2_train_entities":20000,"early_stopping_entities":10000,"selection_entities":20000,
        "supervision":"All development probabilities are out of sample with respect to the frozen first-stage training labels. Second-stage train, early-stop and selection references are disjoint.",
        "caveat":"Previously inspected outer tuning data; only a new reserved audit can establish a final gain. The prior 10k audit is not used.",
        "selection":"Maximize macro F0.5 over blend and threshold; precision >= frozen baseline minus .002; singleton errors <= frozen baseline.",
        "fresh_audit_evaluated":False,"submission_generated":False}
    write_json(args.output/"protocol.json",protocol)
    training=frame[frame.source1_entity_id.isin(train_ids)];early=frame[frame.source1_entity_id.isin(early_ids)]
    selection=frame[frame.source1_entity_id.isin(select_ids)];del frame
    model=lgb.LGBMClassifier(objective="binary",n_estimators=1200,learning_rate=.035,num_leaves=31,
        min_child_samples=100,reg_lambda=5.,colsample_bytree=.9,subsample=.9,subsample_freq=1,
        max_bin=127,random_state=42,n_jobs=args.threads,deterministic=True,force_col_wise=True,verbosity=-1)
    model.fit(training[names].to_numpy(dtype=np.float32),training.label,sample_weight=training.sample_weight,
        feature_name=names,eval_set=[(early[names].to_numpy(dtype=np.float32),early.label)],
        eval_sample_weight=[early.sample_weight],callbacks=[lgb.early_stopping(75),lgb.log_evaluation(100)])
    model.booster_.save_model(str(args.output/"model.txt"))
    probabilities=predict(model.booster_,selection[names].to_numpy(dtype=np.float32),args.threads)
    base=selection.ctx_probability.to_numpy();label=selection.label.to_numpy(dtype=np.uint8)
    positions={e:i for i,e in enumerate(select_ids)};groups=selection.source1_entity_id.map(positions).to_numpy(dtype=np.int32)
    expected=np.array([len(refs[e]["true_ids"]) for e in select_ids]);countries=np.array([refs[e]["country"] for e in select_ids])
    baseline,baseline_values=score(base,.75,label,groups,expected,countries)
    trials=[]
    for weight in (0.,.25,.5,.75,1.):
        blended=(1-weight)*base+weight*probabilities
        for threshold in np.r_[np.arange(.30,.951,.01),.96,.97,.98,.99,.995]:
            result,_=score(blended,round(float(threshold),6),label,groups,expected)
            trials.append({"context_weight":weight,**result})
    eligible=[r for r in trials if r["micro_precision"]>=baseline["micro_precision"]-.002
              and r["singleton_false_positives"]<=baseline["singleton_false_positives"]]
    chosen=max(eligible,key=lambda r:r["macro_f05"])
    weight=chosen["context_weight"]
    chosen_detail,values=score((1-weight)*base+weight*probabilities,chosen["threshold"],label,groups,expected,countries)
    delta=values-baseline_values;rng=np.random.default_rng(42)
    interval=np.quantile([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)],[.025,.975]).tolist()
    report={"status":"exploratory_selection_only","baseline":baseline,"selection":{"context_weight":weight,**chosen_detail},
        "macro_gain":float(delta.mean()),"exploratory_paired_gain_95pct_ci":interval,"best_iteration":model.best_iteration_,
        "seconds":time.perf_counter()-start,"fresh_audit_evaluated":False,"submission_generated":False,
        "retained_candidate_oracle":score(label,.5,label,groups,expected,countries)[0]}
    np.save(args.output/"selection_probabilities.npy",probabilities)
    np.save(args.output/"baseline_selection_probabilities.npy",base)
    write_json(args.output/"threshold_sweep.json",trials);write_json(args.output/"report.json",report)
    write_json(Path("reports/experiments/round2/context_model.json"),report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
