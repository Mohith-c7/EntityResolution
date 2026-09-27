"""Independent anchored-adapter pilot: complete30k claims, early10k truth only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from decision_policy_v2 import PAIR_KEYS
from evaluate_sprint import decide_control, paired_evaluate, predictions_for
from export_sprint_workbench import sha


def run(args):
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    candidate_manifest = json.loads((args.candidate / "manifest.json").read_text())
    model_manifest = json.loads((args.model / "manifest.json").read_text())
    if source_manifest.get("status") != "complete" or source_manifest.get("new_component_fit_owners_excluded") is not True or source_manifest.get("audit_labels_read") is not False:
        raise ValueError("Pilot requires the verified residual-fit-excluded development graph")
    if candidate_manifest.get("status") != "scored_requires_global_decision_replay" or candidate_manifest.get("adapter_strength") != 1.:
        raise ValueError("Only the single authorized full-strength adapter pilot is allowed")
    if sha(args.frozen) != source_manifest["frozen_sha256"]:
        raise ValueError("Pilot control freeze differs")
    if candidate_manifest["adapter_model_sha256"] != model_manifest["model_sha256"] or sha(args.model / "adapter.txt") != model_manifest["model_sha256"]:
        raise ValueError("Pilot model differs from trained artifact")
    ids = json.loads((args.source / "reference_ids.json").read_text())
    if len(ids) != 30000 or set(ids) & set(model_manifest["training_owner_ids"]):
        raise ValueError("Pilot claim graph includes fitted adapter owners")
    early_ids = set(model_manifest["early_owner_ids"])
    if len(early_ids) != 10000 or not early_ids <= set(ids):
        raise ValueError("Pilot early cohort differs from stopping cohort")
    # Only the explicitly supplied early10k label file is ever opened here.
    truth = json.loads(args.early_truth.read_text())
    countries = json.loads(args.early_countries.read_text())
    if set(truth) != early_ids or set(countries) != early_ids or sha(args.early_truth) != model_manifest["early_truth_sha256"]:
        raise ValueError("Pilot truth/country map differs from authorized early10k")
    tables = []
    for directory, manifest in ((args.source, source_manifest), (args.candidate, candidate_manifest)):
        path = directory / "pairs.parquet"
        if sha(path) != manifest["pairs_sha256"]:
            raise ValueError("Pilot score table checksum mismatch")
        tables.append(pd.read_parquet(path, use_threads=False))
    baseline_claims, candidate_claims = tables
    if not baseline_claims[PAIR_KEYS + ["candidate_order"]].equals(candidate_claims[PAIR_KEYS + ["candidate_order"]]):
        raise ValueError("Pilot score keys/order differ")
    if not np.array_equal(baseline_claims.probability.to_numpy(), candidate_claims.probability_original.to_numpy()):
        raise ValueError("Pilot baseline anchor probabilities differ")
    # Independently reconstruct inference from the persisted native model; do
    # not reuse the adapter's composition helper for this check.
    import lightgbm as lgb
    feature_manifest = json.loads((args.features / "manifest.json").read_text())
    feature_proof_key = "reverse_feature_manifest_sha256" if args.family in ("reverse", "neural") else "raw_feature_manifest_sha256"
    correction_column = "reverse_adapter_correction" if args.family in ("reverse", "neural") else "adapter_correction"
    if args.family == "reverse" and (len(model_manifest["features"]) != 36 or model_manifest["features"][-3:] != ["first_stage", "probability", "candidate_rank"] or not all(v.startswith("reverse_") for v in model_manifest["features"][:33])):
        raise ValueError("Reverse pilot must contain exactly the declared33 reverse features plus3 stored-score inputs")
    if args.family == "neural" and (len(model_manifest["features"]) != 37 or model_manifest["features"][-4:] != ["first_stage", "probability", "candidate_rank", "neural_probability"] or not all(v.startswith("reverse_") for v in model_manifest["features"][:33])):
        raise ValueError("Neural pilot must contain exactly the fixed36 inputs plus neural probability")
    if sha(args.features / "manifest.json") != candidate_manifest[feature_proof_key] or sha(args.features / "features.parquet") != feature_manifest["features_sha256"]:
        raise ValueError("Pilot feature artifact differs from score lineage")
    if feature_manifest["input_pairs_sha256"] != source_manifest["pairs_sha256"]:
        raise ValueError("Pilot features were built against another baseline")
    features = pd.read_parquet(args.features / "features.parquet", use_threads=False)
    positions = pd.MultiIndex.from_frame(baseline_claims[PAIR_KEYS]).get_indexer(pd.MultiIndex.from_frame(features[PAIR_KEYS]))
    if (positions < 0).any() or len(set(positions)) != len(positions):
        raise ValueError("Pilot feature keys are foreign or repeated")
    model = lgb.Booster(model_file=str(args.model / "adapter.txt"))
    if model.feature_name() != model_manifest["features"]:
        raise ValueError("Pilot native feature order differs")
    correction = model.predict(features[model_manifest["features"]].to_numpy(dtype=np.float32), raw_score=True, num_threads=1)
    if not np.isfinite(correction).all() or not np.array_equal(correction, candidate_claims[correction_column].to_numpy()[positions]):
        raise ValueError("Persisted raw model correction differs from score output")
    base = baseline_claims.probability.to_numpy()
    expected = base.copy()
    clip = model_manifest["base_probability_clip"]
    clipped = np.clip(base[positions], clip, 1 - clip)
    margin = np.log(clipped) - np.log1p(-clipped) + correction
    expected[positions] = np.where(correction == 0, base[positions], 1 / (1 + np.exp(-np.clip(margin, -50, 50))))
    if not np.array_equal(expected, candidate_claims.probability.to_numpy()):
        raise ValueError("Anchor was omitted, applied twice, or changed outside routed keys")
    config = json.loads(args.frozen.read_text())
    config = {k: config[k] for k in ("threshold", "t_first", "t_rest")}
    mask = baseline_claims.source1_entity_id.isin(early_ids).to_numpy()
    baseline_frame = baseline_claims.loc[mask].reset_index(drop=True)
    candidate_frame = candidate_claims.loc[mask].reset_index(drop=True)
    base_chosen, _ = decide_control(baseline_frame, baseline_claims, config)
    trial_chosen, _ = decide_control(candidate_frame, candidate_claims, config)
    baseline = predictions_for(baseline_frame, base_chosen, early_ids)
    trial = predictions_for(candidate_frame, trial_chosen, early_ids)
    universe = {"kind": "complete_development", "complete": True, "scoring_complete": True,
                "declared_references": len(ids), "scored_references": len(ids), "pairs": len(baseline_claims),
                "supervised_training_excluded": True, "adapter_fit_owners_excluded": True,
                "scope": "Complete early10k+selection20k claims; score only early10k stopping cohort. This is tuning evidence, not independent finalist or fresh confirmation evidence."}
    report = paired_evaluate(truth, baseline, trial, countries, universe=universe, role="development", iterations=2000)
    full_base, _ = decide_control(baseline_claims, baseline_claims, config)
    full_trial, _ = decide_control(candidate_claims, candidate_claims, config)
    if not np.array_equal(full_base[mask], base_chosen) or not np.array_equal(full_trial[mask], trial_chosen):
        raise ValueError("Early decisions differ from complete-graph decision slicing")
    def owner_groups(frame, accepted):
        return frame.loc[accepted].groupby("candidate_entity_id").source1_entity_id.agg(lambda values: frozenset(values)).to_dict()
    base_owners = owner_groups(baseline_claims, full_base)
    trial_owners = owner_groups(candidate_claims, full_trial)
    report["complete_graph_collision_stress"] = {
        "claimants": len(ids), "baseline_collision_targets": sum(len(v) > 1 for v in base_owners.values()),
        "candidate_collision_targets": sum(len(v) > 1 for v in trial_owners.values()),
        "owner_targets_changed": sum(base_owners.get(t, frozenset()) != trial_owners.get(t, frozenset()) for t in set(base_owners) | set(trial_owners)),
        "baseline_accepted_links": int(full_base.sum()), "candidate_accepted_links": int(full_trial.sum()),
        "early_slice_decisions_equal_complete_graph": True,
        "scope": "Complete30k decision-only stress; label quality measured only on early10k."}
    report.update(pilot=("reverse_s1_anchored_strength1_early10k" if args.family == "reverse" else "anchored_adapter_strength1_early10k"), evaluated_at=datetime.now(timezone.utc).isoformat(),
                  stopping_labels_reused_for_pilot_metrics=True, selected20k_adapter_metric_inspected=False,
                  independent_inference_reconstruction_exact=True,
                  fresh_labels_read=False, decision_policy="unchanged_original", decision_config=config,
                  input_hashes={"source_manifest": sha(args.source / "manifest.json"),
                                "candidate_manifest": sha(args.candidate / "manifest.json"),
                                "source_pairs": source_manifest["pairs_sha256"], "candidate_pairs": candidate_manifest["pairs_sha256"],
                                "model_manifest": sha(args.model / "manifest.json"), "model": model_manifest["model_sha256"],
                                "early_truth": sha(args.early_truth), "early_countries": sha(args.early_countries), "frozen": sha(args.frozen)})
    args.output.mkdir(parents=True, exist_ok=False)
    for name, predictions in (("baseline", baseline), ("candidate", trial)):
        (args.output / (name + "_predictions.json")).write_text(json.dumps({e: sorted(v) for e, v in predictions.items()}) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["paired_macro_f05_delta", "paired_delta_95pct_ci", "country_deltas", "links", "acceptance"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "candidate", "model", "features", "early_truth", "early_countries", "frozen", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--family", choices=("anchored", "reverse", "neural"), default="anchored")
    run(p.parse_args())


if __name__ == "__main__": main()
