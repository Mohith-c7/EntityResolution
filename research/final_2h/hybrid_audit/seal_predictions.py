"""Replay a promoted hybrid and Blue05 on ALL heldout owners before slicing.

Reads no labels. All inputs must be newly pinned in a hybrid freeze. This is
separate from the audited four-layer sealer and does not change release05.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(Path(__file__).resolve().parent)]
from preflight import (GROUPS, POLICY, POST_POLICY, bound_file, read, sha, verify_blue_control,
                       verify_own_source, verify_promotion, verify_reservation)

KEYS = ["source1_entity_id", "candidate_entity_id", "candidate_order"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("promotion", "freeze", "reservation", "original", "original_manifest",
                 "baseline", "candidate", "main_route", "rescue_route", "universe", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Require unused hybrid prediction output")
    reservation = verify_reservation(ROOT, a.reservation)
    frozen = read(a.freeze); frozen_sha = sha(a.freeze)
    verify_promotion(read(a.promotion), frozen_sha, reservation["reservation_plan_sha256"])
    verify_blue_control(ROOT, frozen)
    if (frozen.get("status") != "candidate_frozen" or frozen.get("hybrid_policy") != POLICY
            or frozen.get("candidate_post_selection_policy") != POST_POLICY
            or frozen.get("split_ledger_sha256") != reservation["reservation_plan_sha256"]
            or frozen.get("audit_sampled_ids_sha256") != reservation["references_sha256"]):
        raise ValueError("Wrong hybrid freeze/reservation/policy")
    for group in GROUPS:
        if not frozen.get(group): raise ValueError("Missing freeze group: " + group)
        for path, expected in frozen[group].items():
            bound_file(ROOT, {"path": path, "sha256": expected})
    verify_own_source(ROOT, frozen, __file__)
    verify_own_source(ROOT, frozen, ROOT / "research/final_2h/hybrid_audit/preflight.py")
    for name in ("original", "original_manifest", "baseline", "candidate", "main_route", "rescue_route", "universe"):
        if sha(getattr(a, name)) != frozen["prediction_input_sha256"][name]:
            raise ValueError("Hybrid prediction input changed: " + name)
    import numpy as np
    import pandas as pd
    from evaluate_sprint import canonical_sha, decide_control, predictions_for
    sys.path.insert(0, str(ROOT / "research/final_2h"))
    from post_selection_exclusivity import exclusive_selected
    source = read(a.original_manifest)
    if source["pairs"] != 13240480 or len(source["reference_ids"]) != 331012 or source["pairs_sha256"] != sha(a.original):
        raise ValueError("Wrong sealed complete heldout graph")
    if source["frozen_sha256"] != frozen["original_frozen_sha256"]:
        raise ValueError("Original p2 graph came from a different baseline freeze")
    original, baseline, candidate = [pd.read_parquet(x) for x in (a.original, a.baseline, a.candidate)]
    if (len(original) != 13240480 or len(original) != len(baseline) or len(original) != len(candidate)
            or any(not original[KEYS].equals(f[KEYS]) for f in (baseline, candidate))
            or original.duplicated(KEYS[:2]).any()
            or original.groupby(KEYS[0]).size().ne(40).any()
            or set(original.source1_entity_id) != set(source["reference_ids"])):
        raise ValueError("Full graph pair/owner coverage or candidate order differs")
    for frame in (baseline, candidate):
        if (not np.array_equal(frame.probability_original.to_numpy(), original.probability.to_numpy())
                or not np.isfinite(frame.probability).all() or not frame.probability.between(0, 1).all()):
            raise ValueError("Frozen original p2 anchor or corrected probabilities differ")
    main, rescue = [pd.read_parquet(x, columns=KEYS[:2]) for x in (a.main_route, a.rescue_route)]
    if any(f.duplicated(KEYS[:2]).any() for f in (main, rescue)):
        raise ValueError("Repeated routed candidate key")
    main_keys = set(main.itertuples(index=False, name=None)); rescue_keys = set(rescue.itertuples(index=False, name=None))
    if main_keys.intersection(rescue_keys): raise ValueError("Main and rescue route overlap")
    positions = pd.MultiIndex.from_frame(original[KEYS[:2]]).get_indexer(pd.MultiIndex.from_frame(pd.concat([main, rescue], ignore_index=True)))
    if (positions < 0).any(): raise ValueError("Routed pair absent from complete graph")
    outside = np.ones(len(original), dtype=bool); outside[positions] = False
    if not np.array_equal(candidate.probability.to_numpy()[outside], original.probability.to_numpy()[outside]):
        raise ValueError("Outside-union probability changed")
    ids = read(a.reservation / "entity_ids.json")
    if not set(ids) <= set(source["reference_ids"]): raise ValueError("Reserved owner outside full claim universe")
    predictions = []
    exclusivity_removed = 0
    for role, frame, config in (("baseline", baseline, frozen["baseline_decision_config"]),
                                ("candidate", candidate, frozen["decision_config"])):
        chosen, _ = decide_control(frame, frame, config)
        if role == "candidate" and frozen.get("candidate_post_selection_policy") is not None:
            if frozen["candidate_post_selection_policy"] != "selected_only_probability_then_source1_id":
                raise ValueError("Unknown candidate post-selection policy")
            chosen, removed = exclusive_selected(frame, chosen)
            exclusivity_removed = int(removed.sum())
        keep = frame.source1_entity_id.isin(ids).to_numpy()
        predictions.append(predictions_for(frame.loc[keep], chosen[keep], ids))
    a.output.mkdir(parents=True)
    report = {"status": "complete_graph_predictions_frozen_no_labels_read", "labels_read": False,
              "predicted_at": datetime.now(timezone.utc).isoformat(),
              "full_claim_graph": {"references": 331012, "pairs": 13240480},
              "candidate_frozen_sha256": frozen_sha,
              "baseline_frozen_sha256": frozen["baseline_release_freeze_sha256"],
              "references_sha256": reservation["references_sha256"],
              "hybrid_policy": POLICY, "original_p2_anchored_once": True,
              "original_candidate_order_preserved": True, "main_rescue_pair_overlap": 0,
              "main_pairs": len(main), "rescue_pairs": len(rescue),
              "decision_config_sha256": canonical_sha(frozen["decision_config"]),
              "baseline_decision_config_sha256": canonical_sha(frozen["baseline_decision_config"]),
              "baseline_decision_config": frozen["baseline_decision_config"],
              "candidate_post_selection_policy": frozen.get("candidate_post_selection_policy"),
              "post_selection_removed_claims": exclusivity_removed,
              "decision_config": frozen["decision_config"], "source_sha256": sha(__file__)}
    for name, values in zip(("baseline", "candidate"), predictions):
        path = a.output / (name + "_predictions.json")
        path.write_text(json.dumps({e: sorted(values[e]) for e in ids}) + "\n")
        report[name + "_path"] = str(path.resolve()); report[name + "_sha256"] = sha(path)
    report["universe_path"] = str(a.universe.resolve()); report["universe_sha256"] = sha(a.universe)
    (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
