"""Seal label-free predictions after replaying all heldout claimants together."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_sprint import decide_control, predictions_for
from export_sprint_workbench import reference_rows, sha
from evaluate_sprint import canonical_sha
from build_neural_last4_submission import verify_freeze, ADAPTER_SHA, FINE_HEAD_SHA

KEYS = ["source1_entity_id", "candidate_entity_id", "candidate_order"]


def replay_and_slice(baseline, candidate, config, ids):
    """All owners compete; only then restrict results to reserved owners."""
    ids = list(ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate reserved owner")
    if not set(ids) <= set(baseline.source1_entity_id):
        raise ValueError("Reserved owner outside complete claim universe")
    selected = []
    for frame in (baseline, candidate):
        chosen, _ = decide_control(frame, frame, config)
        mask = frame.source1_entity_id.isin(ids).to_numpy()
        selected.append(predictions_for(frame.loc[mask], chosen[mask], ids))
    return tuple(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("original", "original_manifest", "baseline_scored", "candidate_scored",
                  "reserved_references", "decision_config", "freeze", "universe", "output"):
        parser.add_argument("--" + field.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Require unused output directory")
    frozen = verify_freeze(args.freeze)
    own = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if frozen['code_sha256'].get(own) != sha(__file__):
        raise ValueError('Prediction sealing implementation is not frozen')
    source = json.loads(args.original_manifest.read_text())
    if sha(args.original) != source["pairs_sha256"] or len(source["reference_ids"]) != 331012:
        raise ValueError("Heldout original graph seal differs")
    candidate_manifest = json.loads((args.candidate_scored.parent / "manifest.json").read_text())
    if (candidate_manifest.get("status") != "scored_requires_complete_graph_decision_replay"
            or candidate_manifest.get("labels_read") is not False
            or candidate_manifest.get("original_pairs_sha256") != sha(args.original)
            or candidate_manifest.get("scored_pairs_sha256") != sha(args.candidate_scored)
            or candidate_manifest.get('adapter_sha256') != ADAPTER_SHA
            or candidate_manifest.get('fine_head_manifest_sha256') != FINE_HEAD_SHA):
        raise ValueError("Candidate score graph seal differs")
    original = pd.read_parquet(args.original)
    baseline = pd.read_parquet(args.baseline_scored)
    candidate = pd.read_parquet(args.candidate_scored)
    if (len(original) != len(baseline) or len(original) != len(candidate)
            or candidate_manifest["rows"] != len(original) or source["pairs"] != len(original)
            or any(not original[KEYS].equals(frame[KEYS]) for frame in (baseline, candidate))
            or not np.array_equal(original.probability.to_numpy(), baseline.probability.to_numpy())
            or not np.array_equal(original.probability.to_numpy(), candidate.probability_original.to_numpy())):
        raise ValueError("Complete heldout pair keys, order or frozen p2 differs")
    if original.duplicated(KEYS[:2]).any():
        raise ValueError("Duplicate candidate pairs")
    cfg = json.loads(args.decision_config.read_text())
    config = {k: cfg[k] for k in ("threshold", "t_first", "t_rest")}
    if config != frozen['decision_config'] or sha(args.decision_config) != frozen['parent_frozen_sha256']:
        raise ValueError('Control is not original04 with the exact frozen decision rules')
    if source.get('frozen_sha256') != frozen['parent_frozen_sha256']:
        raise ValueError('Heldout original graph is from another frozen baseline')
    rows = reference_rows(args.reserved_references)
    ids = [row["entity_id"] for row in rows]
    if len(ids) != 10000 or len(set(ids)) != len(ids):
        raise ValueError("Not the reserved 10k")
    bp, cp = replay_and_slice(baseline, candidate, config, ids)
    args.output.mkdir(parents=True)
    for name, predictions in (("baseline", bp), ("candidate", cp)):
        path = args.output / (name + "_predictions.json")
        path.write_text(json.dumps({e: sorted(predictions[e]) for e in ids}) + "\n")
    report = {
        "status": "complete_graph_predictions_frozen_no_labels_read",
        "labels_read": False, "predicted_at": datetime.now(timezone.utc).isoformat(),
        "full_claim_graph": {"references": len(source["reference_ids"]), "pairs": len(original)},
        "baseline_sha256": sha(args.output / "baseline_predictions.json"),
        "candidate_sha256": sha(args.output / "candidate_predictions.json"),
        "references_sha256": sha(args.reserved_references),
        "universe_sha256": sha(args.universe),
        "candidate_frozen_sha256": sha(args.freeze),
        "baseline_frozen_sha256": frozen['parent_frozen_sha256'],
        "decision_config_sha256": canonical_sha(config),
        "decision_config": config, "decision_config_source_sha256": sha(args.decision_config),
        "original_manifest_sha256": sha(args.original_manifest),
        "baseline_scored_sha256": sha(args.baseline_scored),
        "candidate_scored_sha256": sha(args.candidate_scored),
        "candidate_scored_manifest_sha256": sha(args.candidate_scored.parent / "manifest.json"),
        "source_sha256": sha(__file__),
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
