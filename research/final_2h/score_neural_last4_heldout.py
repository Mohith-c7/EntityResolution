"""Keyed 38-feature inference on the complete sealed heldout claim graph.

No labels are read. Decisions and reserved-cohort evaluation are separate stages.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/sprint_6h"), str(ROOT / "research/final_2h")]
import neural_adapter as neural
import neural_peer as peer

KEYS = neural.sibling.KEYS
FEATURES = [*neural.FEATURES, "neural_fine_probability"]


def scatter_corrections(frame, features, correction):
    """Apply one anchored correction only to verified, pair-keyed route rows."""
    if frame.duplicated(KEYS).any() or features.duplicated(KEYS).any():
        raise ValueError("Duplicate pair keys")
    if len(features) != len(correction) or not np.isfinite(correction).all():
        raise ValueError("Nonfinite or incomplete correction")
    positions = pd.MultiIndex.from_frame(frame[KEYS]).get_indexer(
        pd.MultiIndex.from_frame(features[KEYS])
    )
    if (positions < 0).any():
        raise ValueError("Fine features contain a foreign pair key")
    base = frame.probability.to_numpy()[positions]
    if (not np.array_equal(base, features.probability.to_numpy())
            or not np.array_equal(frame.first_stage.to_numpy()[positions], features.first_stage.to_numpy())
            or not np.array_equal(neural.reverse.frozen_p2_rank(frame)[positions], features.candidate_rank.to_numpy())):
        raise ValueError("Frozen probability anchor, first-stage score or candidate rank changed")
    changed = frame.copy()
    changed["probability_original"] = frame.probability
    changed["fine_adapter_correction"] = 0.
    changed.iloc[positions, changed.columns.get_loc("fine_adapter_correction")] = correction
    changed.iloc[positions, changed.columns.get_loc("probability")] = neural.reverse.corrected_probability(base, correction)
    if not np.isfinite(changed.probability.to_numpy()).all():
        raise ValueError("Nonfinite corrected probability")
    mask = np.ones(len(frame), dtype=bool)
    mask[positions] = False
    if not np.array_equal(changed.probability.to_numpy()[mask], frame.probability.to_numpy()[mask]):
        raise ValueError("Outside-route score changed")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("pairs", "pair_manifest", "features", "fine_scores", "fine_input_manifest",
                  "fine_head_manifest", "old_head_manifest", "adapter", "output"):
        parser.add_argument("--" + field.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Require unused output directory")
    import lightgbm as lgb

    frame, pair_manifest = neural.sibling.load_pairs(args.pairs, args.pair_manifest)
    features, feature_manifest = neural.load_features(args.features)
    model_report = json.loads((args.adapter / "report.json").read_text())
    old_head_hash = neural.sibling.digest(args.old_head_manifest)
    fine_head_hash = neural.sibling.digest(args.fine_head_manifest)
    adapter_path = args.adapter / "adapter.txt"
    if (model_report.get("features") != FEATURES
            or model_report.get("source_sha256") != peer.sha(peer.__file__)
            or model_report.get("model_sha256") != peer.sha(adapter_path)
            or model_report.get("original_neural_head_sha256") != old_head_hash
            or model_report.get("fine_head_sha256") != fine_head_hash
            or feature_manifest.get("neural_head_manifest_sha256") != old_head_hash
            or feature_manifest.get("input_pairs_sha256") != pair_manifest["pairs_sha256"]):
        raise ValueError("Adapter, heads or frozen feature lineage differs")
    heldout = set(pair_manifest["reference_ids"])
    head = json.loads(args.fine_head_manifest.read_text())
    old_head = json.loads(args.old_head_manifest.read_text())
    if heldout & (set(model_report["fit_owners"]) | set(head["training_owners"]) | set(old_head["training_owners"])):
        raise ValueError("Adapter or neural head fitted a heldout owner")
    combined = peer.add_fine(features, feature_manifest, args.fine_scores,
                             args.fine_input_manifest, args.fine_head_manifest)
    model = lgb.Booster(model_file=str(adapter_path))
    if model.feature_name() != FEATURES or model.num_feature() != len(FEATURES):
        raise ValueError("Native adapter feature names/order differs")
    values = combined[FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite 38-feature input")
    correction = model.predict(values, raw_score=True, num_threads=8)
    changed = scatter_corrections(frame, combined, correction)
    if not frame[[*KEYS, "candidate_order"]].equals(changed[[*KEYS, "candidate_order"]]):
        raise ValueError("Candidate keys or order changed")
    args.output.mkdir(parents=True)
    target = args.output / "pairs.parquet"
    changed.to_parquet(target, index=False)
    marker = {
        "status": "scored_requires_complete_graph_decision_replay",
        "labels_read": False,
        "rows": len(changed), "routed_rows": len(combined),
        "pair_manifest_sha256": peer.sha(args.pair_manifest),
        "original_pairs_sha256": peer.sha(args.pairs),
        "features_manifest_sha256": peer.sha(args.features / "manifest.json"),
        "fine_scores_manifest_sha256": peer.sha(args.fine_scores / "manifest.json"),
        "fine_head_manifest_sha256": fine_head_hash,
        "adapter_sha256": peer.sha(adapter_path),
        "scored_pairs_sha256": peer.sha(target),
        "source_sha256": peer.sha(__file__),
    }
    (args.output / "manifest.json").write_text(json.dumps(marker, indent=2) + "\n")
    print(json.dumps(marker), flush=True)


if __name__ == "__main__":
    main()
