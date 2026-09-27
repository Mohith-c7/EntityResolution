"""Describe fixed fine-head confidence on consumed audit pairs; no rescoring."""
from collections import Counter
import json
from pathlib import Path
import unicodedata

import pandas as pd
from preflight import read, sha


def main():
    lane = Path("research/final_2h/hybrid_audit")
    truth = {e:set(t) for e,t in read(lane / "extension_evaluation_v1/truth.json").items()}
    pred = {e:set(t) for e,t in read(lane / "heldout_predictions_v2/candidate_predictions.json").items()}
    refs = {r["entity_id"]:r for r in read("research/final_2h/extension_audit_last8_v1/references.json")}
    frame = pd.read_parquet(lane / "heldout_scored_v1/pairs.parquet", filters=[("source1_entity_id","in",list(truth))])
    fine = pd.read_json("models/sprint_6h/neural/last8_heldout331k/scores.jsonl",lines=True).rename(columns={"neural_probability":"fine8"})
    old = pd.read_parquet("research/sprint_6h/sibling/neural_v2_features_heldout331k/features.parquet",columns=["source1_entity_id","candidate_entity_id","neural_probability"]).rename(columns={"neural_probability":"old_head"})
    joined = frame.merge(fine,on=["source1_entity_id","candidate_entity_id"],how="inner",validate="one_to_one").merge(old,on=["source1_entity_id","candidate_entity_id"],validate="one_to_one")
    joined["true"] = [t in truth[e] for e,t in joined[["source1_entity_id","candidate_entity_id"]].itertuples(index=False,name=None)]
    joined["selected"] = [t in pred[e] for e,t in joined[["source1_entity_id","candidate_entity_id"]].itertuples(index=False,name=None)]
    counts = {}
    for cut in (.9,.95):
        pocket=joined.loc[(joined.fine8>=cut)&(joined.probability<.79)&~joined.selected]
        counts[str(cut)]={"unselected_below_079_pairs":len(pocket),"true":int(pocket.true.sum()),"false":int((~pocket.true).sum()),"owners":int(pocket.source1_entity_id.nunique()),"countries":pocket.groupby("country").true.agg(["count","sum"]).to_dict(),"original_p2_quantiles":pocket.probability_original.quantile([0,.25,.5,.75,1]).to_dict(),"old_head_quantiles":pocket.old_head.quantile([0,.25,.5,.75,1]).to_dict()}
    missed=joined.loc[joined.true&~joined.selected]; wrong=joined.loc[~joined.true&joined.selected]
    for role,data in (("routed_true_rejected",missed),("routed_false_accepted",wrong)):
        counts[role]={"pairs":len(data),"fine8_quantiles":data.fine8.quantile([0,.25,.5,.75,1]).to_dict(),"old_head_quantiles":data.old_head.quantile([0,.25,.5,.75,1]).to_dict(),"original_p2_quantiles":data.probability_original.quantile([0,.25,.5,.75,1]).to_dict(),"corrected_p2_quantiles":data.probability.quantile([0,.25,.5,.75,1]).to_dict()}
    traits=Counter()
    for e,gold in truth.items():
        if gold==pred[e]:continue
        r=refs[e];country=r["country"]
        traits[country+"_error_owners"]+=1
        traits[country+"_source_address_empty"]+=not (r["business_address"] or "").strip()
        scripts={unicodedata.name(c,"").split()[0] for c in (r["business_name"] or "") if c.isalpha()}
        traits[country+"_non_latin_name"]+=bool(scripts-{"LATIN"})
    out={"status":"consumed_audit_fixed_neural_confidence_description","cohort_consumed":True,"no_rescoring_or_alternative_predictions":True,"original_nn8_route_pairs_in_consumed_cohort":len(joined),"confidence_counts":counts,"source1_error_traits":dict(traits),"interpretation_limit":"These known-label pockets describe suppression/risk and cannot establish a prospective override gain. No thresholds or outputs changed.","source_sha256":sha(__file__)}
    path=Path("reports/final_2h/hybrid_consumed_neural_confidence_v1.json")
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))


if __name__=="__main__":main()
