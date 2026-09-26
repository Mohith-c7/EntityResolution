"""Tuning-only feature ablations using frozen, entity-isolated pair tables."""

import json
import time
import shutil
from pathlib import Path

import pandas as pd

from .train import train
from .predict import predict
from .threshold import tune_threshold,select_matches
from ..evaluation.metrics import score_matches
from ..features.registry import names_for_version


def run_experiments(artifact_dir,output_dir,*,threads=2,mode="features"):
    source,output=Path(artifact_dir),Path(output_dir)
    if (output/"report.json").exists():
        raise FileExistsError("Completed experiment exists; choose a fresh output directory")
    output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("MODEL_LICENSE.txt"),output/"MODEL_LICENSE.txt")
    baseline=json.loads((source/"report.json").read_text())
    registry=names_for_version(baseline["feature_version"])
    train_pairs=pd.read_parquet(source/"pairs_train.parquet")
    tune_pairs=pd.read_parquet(source/"pairs_tune.parquet")
    references=json.loads((source/"sampled_references.json").read_text())["tune"]
    labels=json.loads((source/"sampled_truth.json").read_text())
    truth={row["entity_id"]:set(labels[row["entity_id"]]) for row in references}
    countries={row["entity_id"]:row["country"].casefold().strip() for row in references}
    aliases={name for name in registry if name.startswith("canonical_name_")}
    aliases.update(("canonical_core_exact","candidate_alias_confidence","candidate_alias_log_support"))
    if mode=="features":
        variants={"without_blocking_score":({"blocking_score"},{}),
            "without_blocking_or_alias_scores":({"blocking_score"}|aliases,{})}
    elif mode=="capacity":
        variants={"longer_lower_regularization":(set(),{"n_estimators":3000,"reg_lambda":.2}),
            "more_leaves":(set(),{"n_estimators":2400,"num_leaves":63,"reg_lambda":1.0})}
    else:
        raise ValueError("mode must be features or capacity")
    rows=[]
    for name,(excluded,overrides) in variants.items():
        start=time.perf_counter()
        names=tuple(feature for feature in registry if feature not in excluded)
        model=train(train_pairs,tune_pairs,threads=threads,seed=baseline["seed"],feature_names=names,overrides=overrides)
        probabilities=predict(model,tune_pairs)
        threshold,sweep=tune_threshold(tune_pairs,probabilities,truth)
        predictions=select_matches(tune_pairs,probabilities,threshold,truth)
        row={"name":name,"features":names,"parameter_overrides":overrides,"threshold":threshold,"tune":score_matches(truth,predictions,countries),
            "best_iteration":model.best_iteration_,"seconds":time.perf_counter()-start}
        model.booster_.save_model(str(output/f"{name}.txt"))
        tune_pairs.loc[:,["source1_entity_id","candidate_entity_id","label"]].assign(probability=probabilities).to_parquet(output/f"{name}_tune.parquet",index=False)
        rows.append(row)
        (output/"progress.json").write_text(json.dumps(rows,indent=2)+"\n")
        print(json.dumps({"stage":"ablation_complete",**row}),flush=True)
    report={"baseline_artifact":source.name,"baseline_tune":baseline["tune"],"mode":mode,"rows":rows,
        "policy":"Development comparison only; holdout entities do not contribute to fitting or selection. These experiment model files require explicit integration and independent evaluation before production inference."}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report
