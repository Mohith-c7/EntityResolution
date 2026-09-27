"""Describe consumed extension errors; no alternative decisions or evaluation.

Uses already sealed truth/predictions only after their completed single-use
audit. Outputs aggregate counts and deficit accounting, never entity records.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_sprint import entity_f05
from preflight import read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    lane = Path("research/final_2h/hybrid_audit")
    audit = lane / "extension_evaluation_v1"
    report = read(audit / "report.json")
    if report["status"] != "paired_extension_audit_complete": raise ValueError("Audit not consumed")
    truth_path = audit / "truth.json"
    evidence = read(lane / "heldout_predictions_v2/manifest.json")
    if sha(truth_path) != report["input_sha256"]["truth"]: raise ValueError("Consumed truth changed")
    truth = {e: set(t) for e, t in read(truth_path).items()}
    pred = {e: set(t) for e, t in read(evidence["candidate_path"]).items()}
    if set(truth) != set(pred): raise ValueError("Owner mismatch")
    import pandas as pd
    keys = pd.read_parquet("models/sprint_6h/heldout_cohort/pairs.parquet",
                          columns=["source1_entity_id", "candidate_entity_id"])
    keys = keys.loc[keys.source1_entity_id.isin(truth)]
    candidates = keys.groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
    if set(candidates) != set(truth) or any(len(t) != 40 for t in candidates.values()): raise ValueError("Original40 candidate scope differs")
    groups = {}
    retrieval_deficit = score_deficit = 0.0
    missing_links = retrieved_rejected = false_links = 0
    missing_owners = 0
    for e, gold in truth.items():
        chosen = pred[e]; fp, fn = len(chosen-gold), len(gold-chosen)
        if gold == chosen: category = "perfect"
        elif not gold: category = "singleton_false_positive"
        elif not chosen: category = "non_singleton_empty"
        elif fp and fn: category = "mixed_false_positive_and_false_negative"
        elif fp: category = "false_positive_only"
        else: category = "false_negative_only"
        observed = entity_f05(gold, chosen)
        available = gold & candidates[e]
        ceiling = entity_f05(gold, available)
        if ceiling + 1e-12 < observed: raise ValueError("Oracle upper bound violated")
        entry = groups.setdefault(category, {"entities": 0, "macro_deficit": 0.0, "false_links": 0, "missed_true_links": 0})
        entry["entities"] += 1; entry["macro_deficit"] += (1-observed)/len(truth)
        entry["false_links"] += fp; entry["missed_true_links"] += fn
        retrieval_deficit += (1-ceiling)/len(truth)
        score_deficit += (ceiling-observed)/len(truth)
        missing_links += len(gold-candidates[e]); missing_owners += bool(gold-candidates[e])
        retrieved_rejected += len((gold-chosen) & candidates[e]); false_links += fp
    total = sum(v["macro_deficit"] for v in groups.values())
    if abs(total-(1-report["candidate"]["macro_f05"])) > 1e-10: raise ValueError("Loss accounting differs from frozen audit")
    out = {"status": "consumed_extension_audit_descriptive_loss_accounting",
           "cohort_consumed": True, "no_candidate_selection_or_retuning": True,
           "entities": len(truth), "candidate_macro_f05": report["candidate"]["macro_f05"],
           "total_macro_deficit": total, "error_categories": groups,
           "true_links_absent_from_original40": missing_links, "owners_with_absent_true_candidates": missing_owners,
           "retrieved_true_links_rejected": retrieved_rejected, "accepted_false_links": false_links,
           "retrieval_upper_bound_macro_f05": 1-retrieval_deficit,
           "candidate_absence_macro_deficit": retrieval_deficit,
           "scoring_or_decision_macro_deficit": score_deficit,
           "upper_bound_note": "Perfect truth selection within the existing40 candidates is an oracle ceiling, not a tested matcher gain. Remaining scoring/decision loss does not identify a deployable rule.",
           "country_scope": ["US", "India"], "france_accuracy_unknown": True,
           "gap_to_user_qualification_cutoff_0_989279": .989279-report["candidate"]["macro_f05"],
           "source_sha256": sha(__file__), "input_sha256": {"audit_report": sha(audit / "report.json"), "consumed_truth": sha(truth_path), "prediction_evidence": sha(lane / "heldout_predictions_v2/manifest.json")}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__": main()
