"""Create the 30k learned-model claimant universe, excluding residual-fit owners."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import train_sibling_matcher as sibling
from run_frozen_pipeline import decide


def make_learned(frame, manifest, truth, countries, claimant_ids, evaluation_ids, excluded_ids, config, output):
    if set(claimant_ids) & set(excluded_ids) or not set(evaluation_ids) <= set(claimant_ids):
        raise ValueError("Residual-fit owners entered learned-model claimant population")
    part = frame[frame.source1_entity_id.isin(claimant_ids)].reset_index(drop=True)
    part["accepted_source50k"] = part.accepted
    part["accepted"], _ = decide(part, part, config)
    actual = {e: truth[e] for e in claimant_ids}
    values = sibling.entity_values(part, part.accepted.to_numpy(dtype=bool), actual)
    output = Path(output)
    output.mkdir(parents=True)
    part.to_parquet(output / "pairs.parquet", index=False)
    sibling.write_json(output / "truth.json", actual)
    sibling.write_json(output / "countries.json", {e: countries[e] for e in claimant_ids})
    sibling.write_json(output / "reference_ids.json", claimant_ids)
    sibling.write_json(output / "evaluation_reference_ids.json", evaluation_ids)
    result = {**manifest, "status": "complete", "split": "development", "role": "learned_development",
        "reference_ids": claimant_ids, "entities": len(claimant_ids), "pairs": len(part),
        "evaluation_reference_ids": evaluation_ids, "evaluation_entities": len(evaluation_ids),
        "expected_macro_f05": float(np.mean(list(values.values()))),
        "evaluation_expected_macro_f05": float(np.mean([values[e] for e in evaluation_ids])),
        "evaluation_expected_country_macro_f05": {c: float(np.mean([values[e] for e in evaluation_ids if countries[e] == c]))
            for c in sorted({countries[e] for e in evaluation_ids})},
        "evaluation_reference_ids_sha256": hashlib.sha256("\n".join(sorted(evaluation_ids)).encode()).hexdigest(),
        "excluded_residual_training_reference_ids_sha256": hashlib.sha256("\n".join(sorted(excluded_ids)).encode()).hexdigest(),
        "pairs_sha256": sibling.digest(output / "pairs.parquet"), "pair_order_sha256": sibling.pair_order_hash(part),
        "source50k_pairs_sha256": manifest["pairs_sha256"],
        "claims_universe_entities": len(claimant_ids),
        "decision_scope": "Original incumbent policy recomputed over complete declared early+selection claimant population, excluding all residual-training owners. External rival coverage remains incomplete.",
        "new_component_fit_owners_excluded": True, "historical_full_ownership_replayed": False,
        "accepted_changes_from_source50k": int((part.accepted != part.accepted_source50k).sum()),
        "labels_separate": True, "truth_sha256": sibling.digest(output / "truth.json")}
    result["claims_universe_pairs_sha256"] = result["pairs_sha256"]
    result["routing_universe_pair_order_sha256"] = result["pair_order_sha256"]
    result.pop("chunks", None)
    sibling.write_json(output / "manifest.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--role-root", type=Path, required=True)
    p.add_argument("--decision-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    pa.set_cpu_count(1)
    started = time.perf_counter()
    frame, manifest = sibling.load_pairs(args.pairs, args.manifest)
    if manifest["split"] != "development" or len(manifest["reference_ids"]) != 50000:
        raise ValueError("Need verified current complete 50k development input")
    if sibling.digest(args.decision_config) != manifest["frozen_sha256"]:
        raise ValueError("Decision config differs from verified incumbent")
    get = lambda role, name: json.loads((args.role_root / role / name).read_text())
    early, selection, excluded = [get(role, "reference_ids.json") for role in ("early_stop", "selection", "residual_train")]
    truth, countries = {}, {}
    for role in ("early_stop", "selection"):
        truth.update(get(role, "truth.json")); countries.update(get(role, "countries.json"))
    result = make_learned(frame, manifest, truth, countries, early+selection, selection, excluded,
        json.loads(args.decision_config.read_text()), args.output)
    sibling.write_json(args.output / "lineage.json", {"source_manifest_sha256": sibling.digest(args.manifest),
        "decision_config_sha256": sibling.digest(args.decision_config), "code_sha256": sibling.digest(__file__),
        "seconds": time.perf_counter()-started})
    print(json.dumps({k:result[k] for k in ("entities", "pairs", "evaluation_entities", "evaluation_expected_macro_f05",
        "evaluation_expected_country_macro_f05", "accepted_changes_from_source50k", "pairs_sha256", "pair_order_sha256")}), flush=True)


if __name__ == "__main__": main()
