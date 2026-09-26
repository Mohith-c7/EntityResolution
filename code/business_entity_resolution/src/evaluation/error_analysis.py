"""Separate retrieval losses from matcher errors on saved tuning predictions."""

from pathlib import Path
import json

import pandas as pd

from .metrics import score_matches, entity_f05
from ..model.threshold import select_matches


def analyze_tuning(artifact_dir):
    directory=Path(artifact_dir)
    report=json.loads((directory/"report.json").read_text())
    references=json.loads((directory/"sampled_references.json").read_text())["tune"]
    labels=json.loads((directory/"sampled_truth.json").read_text())
    truth={row["entity_id"]:set(labels[row["entity_id"]]) for row in references}
    countries={row["entity_id"]:row["country"].casefold().strip() for row in references}
    frame=pd.read_parquet(directory/"predictions_tune.parquet")
    threshold=report["threshold"]
    predicted=select_matches(frame,frame.probability.to_numpy(),threshold,truth)
    available={eid:set() for eid in truth}
    for eid,cid in frame.loc[frame.label==1,["source1_entity_id","candidate_entity_id"]].itertuples(index=False,name=None):
        available[eid].add(cid)
    cleaned={eid:set(predicted[eid])&truth[eid] for eid in truth}
    recovered={eid:set(predicted[eid])|available[eid] for eid in truth}
    selected=frame.probability>=threshold
    positive=frame.label==1
    frame=frame.assign(outcome="true_negative")
    frame.loc[selected&positive,"outcome"]="true_positive"
    frame.loc[selected&~positive,"outcome"]="false_positive"
    frame.loc[~selected&positive,"outcome"]="rejected_true_link"
    metrics=score_matches(truth,predicted,countries)
    counterfactuals={name:score_matches(truth,values,countries)["macro_f05"] for name,values in (
        ("remove_all_false_predictions",cleaned),
        ("recover_all_retrieved_true_links",recovered),
        ("perfect_matcher_on_retained_candidates",available))}
    slices={}
    masks={"candidate_address_missing":frame.candidate_address_missing==1,
        "weak_raw_name_similarity":frame.name_token_sort<.5,
        "high_address_token_similarity":frame.address_token_set>=.9}
    if "candidate_alias_confidence" in frame:
        masks["no_learned_alias_evidence"]=frame.candidate_alias_confidence==0
        masks["learned_alias_evidence_present"]=frame.candidate_alias_confidence>0
        masks["first_numeric_agreement_absent"]=frame.first_numeric_canonical_exact==0
    for name,mask in masks.items():
        part=frame.loc[mask]
        slices[name]={"pairs":len(part),"outcomes":{str(k):int(v) for k,v in part.outcome.value_counts().items()}}
    by_source={}
    for source in ("S2","S3"):
        expected={eid:{cid for cid in ids if cid.startswith(source+"-")} for eid,ids in truth.items()}
        found={eid:{cid for cid in ids if cid.startswith(source+"-")} for eid,ids in predicted.items()}
        by_source[source]=score_matches(expected,found)
    mistakes=frame.loc[frame.outcome.isin(("false_positive","rejected_true_link"))].copy()
    mistakes["distance_to_threshold"]=(mistakes.probability-threshold).abs()
    entity_errors=[]
    for eid in truth:
        p=set(predicted[eid]);t=truth[eid]
        if p!=t:
            entity_errors.append({"source1_entity_id":eid,"country":countries[eid],
                "macro_loss":1-entity_f05(t,p),"blocked_true_links":sorted(t-available[eid]),
                "rejected_true_links":sorted(available[eid]-p),"false_predictions":sorted(p-t)})
    entity_errors.sort(key=lambda row:(-row["macro_loss"],row["source1_entity_id"]))
    summary={"artifact":directory.name,"fold":"tune","threshold":threshold,
        "policy":"Tuning diagnostics only. Counterfactual scores require ground truth and are not deployable predictions. Slices overlap.",
        "actual":metrics,"blocked_true_links":sum(len(truth[eid]-available[eid]) for eid in truth),
        "rejected_true_links":int((~selected&positive).sum()),"false_positive_links":int((selected&~positive).sum()),
        "counterfactual_macro_f05":counterfactuals,"slices":slices,"source_metrics":by_source,
        "errors_within_005_of_threshold":int((mistakes.distance_to_threshold<=.05).sum()),
        "total_matcher_errors":len(mistakes)}
    (directory/"tuning_error_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (directory/"tuning_entity_errors.json").write_text(json.dumps(entity_errors,indent=2)+"\n")
    mistakes.sort_values(["outcome","distance_to_threshold"]).to_parquet(directory/"tuning_pair_errors.parquet",index=False)
    return summary
