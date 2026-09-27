"""Read-only traits of consumed audit errors; no alternative prediction rule."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "code/business_entity_resolution")]
from evaluate_sprint import entity_f05
from preflight import read, sha
from src.preprocessing.normalize import normalize_name, normalize_address
from rapidfuzz.fuzz import token_sort_ratio
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--private-examples", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.private_examples.exists(): raise FileExistsError("Unused diagnostic paths required")
    lane = Path("research/final_2h/hybrid_audit")
    truth = {e: set(v) for e,v in read(lane / "extension_evaluation_v1/truth.json").items()}
    pred = {e: set(v) for e,v in read(lane / "heldout_predictions_v2/candidate_predictions.json").items()}
    refs = {r["entity_id"]: r for r in read("research/final_2h/extension_audit_last8_v1/references.json")}
    frame = pd.read_parquet(lane / "heldout_scored_v1/pairs.parquet", filters=[("source1_entity_id","in",list(truth))])
    if len(frame) != 400000: raise ValueError("Wrong consumed audit pair scope")
    for name, column in (("rank_p1","first_stage"),("rank_original_p2","probability_original"),("rank_corrected_p2","probability")):
        ordered = frame.sort_values(["source1_entity_id",column,"candidate_entity_id"], ascending=[True,False,True])
        frame[name] = (ordered.groupby("source1_entity_id").cumcount()+1).reindex(frame.index)
    main = set(pd.read_parquet(lane / "heldout_scored_v1/main_route.parquet").itertuples(index=False,name=None))
    extra = set(pd.read_parquet(lane / "heldout_scored_v1/rescue_route.parquet").itertuples(index=False,name=None))
    indexed = {(r.source1_entity_id,r.candidate_entity_id):r for r in frame.itertuples(index=False)}
    missed = [(e,t) for e,g in truth.items() for t in sorted(g-pred[e]) if (e,t) in indexed]
    false = [(e,t) for e,predicted in pred.items() for t in sorted(predicted-truth[e])]
    targets = {}
    allids = {t for _,t in missed+false}
    for source in (2,3):
        ids = sorted(t for t in allids if t.startswith(f"S{source}-"))
        path = (ROOT / f"models/index_train_S{source}.sqlite").resolve()
        with sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as conn:
            for start in range(0,len(ids),500):
                group=ids[start:start+500]
                for e,n,ad in conn.execute("SELECT id,name,address FROM records WHERE id IN ("+",".join("?"*len(group))+")",group): targets[e] = (n or "",ad or "")
    if set(targets) != allids: raise ValueError("Missing supplied target records")
    stats = {}; examples = []
    pockets = {"unrouted_high_p1": [], "unrouted_very_high_name_similarity": [], "near_acceptance_threshold": []}
    for role,pairs in (("retrieved_true_rejected",missed),("accepted_false",false)):
        count = Counter(); quantiles = []
        for e,t in pairs:
            row=indexed[e,t]; route="main8" if (e,t) in main else "extra4" if (e,t) in extra else "unrouted"
            ref=refs[e]; name,address=targets[t]
            rn=normalize_name(ref["business_name"]); ra=normalize_address(ref["business_address"])
            tn=normalize_name(name); ta=normalize_address(address)
            ns=token_sort_ratio(rn,tn)/100; ads=token_sort_ratio(ra,ta)/100
            nd=set(re.findall(r"\d+",ra)); td=set(re.findall(r"\d+",ta)); conflict=bool(nd and td and not nd&td)
            count["route_"+route]+=1; count["owner_already_nonempty" if pred[e] else "owner_empty"]+=1
            count["p1_rank_le4"] += row.rank_p1<=4; count["original_p2_rank_le4"] += row.rank_original_p2<=4
            count["name_similarity_ge95"]+=ns>=.95; count["address_missing_either"]+=not ra or not ta
            count["number_conflict"]+=conflict; count["current_p_ge_079"]+=row.probability>=.79
            count["current_p_05_to_079"]+=.5<=row.probability<.79
            count["current_p_below_003"]+=row.probability<.03
            quantiles.append(float(row.probability))
            if role=="retrieved_true_rejected":
                if route=="unrouted" and row.rank_p1<=4: pockets["unrouted_high_p1"].append((e,t))
                if route=="unrouted" and ns>=.95 and not conflict: pockets["unrouted_very_high_name_similarity"].append((e,t))
                if .5<=row.probability<.79: pockets["near_acceptance_threshold"].append((e,t))
            if len([v for v in examples if v["role"]==role])<6 and (route=="unrouted" or role=="accepted_false"):
                examples.append({"role":role,"source1_entity_id":e,"candidate_entity_id":t,"source1_name":ref["business_name"],"source1_address":ref["business_address"],"target_name":name,"target_address":address,"route":route,"original_p2":float(row.probability_original),"corrected_p2":float(row.probability),"first_stage":float(row.first_stage),"rank_p1":int(row.rank_p1),"rank_original_p2":int(row.rank_original_p2),"name_similarity":ns,"address_similarity":ads,"number_conflict":conflict})
        stats[role]={"pairs":len(pairs),"counts":dict(count),"corrected_probability_quantiles":pd.Series(quantiles).quantile([0,.25,.5,.75,1]).to_dict()}
    ceilings = {}
    for name,pairs in pockets.items():
        additions = {}
        for e,t in pairs: additions.setdefault(e,set()).add(t)
        gain=sum(entity_f05(truth[e],pred[e]|ts)-entity_f05(truth[e],pred[e]) for e,ts in additions.items())/len(truth)
        ceilings[name]={"true_pairs":len(pairs),"owners":len(additions),"perfect_true_only_addition_macro_upper_bound":gain}
    out={"status":"consumed_audit_pair_traits_only","cohort_consumed":True,"no_new_predictions_or_candidate_selection":True,"pair_traits":stats,"conditional_oracle_pockets":ceilings,"oracle_pockets_are_not_tested_gains":True,"raw_examples_private_not_for_git":True,"source_sha256":sha(__file__)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n")
    a.private_examples.write_text(json.dumps(examples,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
