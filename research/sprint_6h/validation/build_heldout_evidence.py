"""Certify a sealed complete heldout score graph without opening owner labels."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from export_sprint_workbench import reference_rows, sha


def build(workbench, reservation, frozen):
    plan = json.loads(reservation.read_text())
    manifest_path = workbench / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("verified_current_pipeline") is not True:
        raise ValueError("Complete heldout workbench must be sealed and verified")
    if manifest.get("split") not in ("holdout", "cohort") or manifest.get("audit_labels_read") is not False:
        raise ValueError("Require unlabeled outer-heldout score graph")
    if plan["claimant_universe"].get("kind") != "complete_heldout" or plan["claimant_universe"].get("labels_read") is not False:
        raise ValueError("Reservation does not define an unlabeled complete heldout population")
    if manifest.get("second_stage_owner_excluded") is not True or manifest.get("frozen_sha256") != sha(frozen):
        raise ValueError("Current control owner exclusion/freeze mismatch")
    for name in ("parity", "asset"):
        proof = Path(manifest[name + "_proof"])
        if sha(proof) != manifest[name + "_proof_sha256"]:
            raise ValueError("Heldout verification proof changed")
    raw = reservation.parent / "heldout_universe.json"
    if sha(raw) != plan["claimant_universe"]["records_sha256"] or manifest["references_sha256"] != sha(raw):
        raise ValueError("Score universe differs from the reserved complete heldout records")
    rows = reference_rows(workbench / "references.json")
    raw_rows = reference_rows(raw)
    ids = [r["entity_id"] for r in rows]
    if rows != raw_rows or ids != manifest["reference_ids"] or len(ids) != manifest["entities"]:
        raise ValueError("Scored reference records/coverage differ")
    if len(ids) != plan["claimant_universe"]["entities"]:
        raise ValueError("Declared heldout coverage differs from reservation")
    offset = 0
    for marker in sorted(manifest["chunks"], key=lambda v: v["chunk"]):
        expected = ids[offset:offset + marker["entities"]]
        if len(expected) != marker["entities"] or hashlib.sha256("\n".join(expected).encode()).hexdigest() != marker["input_ids_sha256"]:
            raise ValueError("Processed chunk reference coverage/order mismatch")
        offset += marker["entities"]
    if offset != len(ids) or sum(c["pairs"] for c in manifest["chunks"]) != manifest["pairs"]:
        raise ValueError("Chunk coverage does not include every processed reference")
    pairs = workbench / "pairs.parquet"
    if sha(pairs) != manifest["pairs_sha256"]:
        raise ValueError("Heldout pair artifact changed after seal")
    encountered = set(); pair_hash = hashlib.sha256(); pair_count = 0
    for batch in pq.ParquetFile(pairs).iter_batches(batch_size=100000, columns=["source1_entity_id", "candidate_entity_id"], use_threads=False):
        sources, targets = batch.column(0).to_pylist(), batch.column(1).to_pylist()
        encountered.update(sources)
        for s, t in zip(sources, targets): pair_hash.update(f"{s}\t{t}\n".encode())
        pair_count += len(sources)
    if pair_count != manifest["pairs"] or pair_hash.hexdigest() != manifest["pair_order_sha256"] or not encountered <= set(ids):
        raise ValueError("Projected pair-key identity/coverage mismatch")
    empty = [e for e in ids if e not in encountered]
    cohorts = {}
    for role in ("confirmation", "audit"):
        path = reservation.parent / role / "references.json"
        reserved = reference_rows(path); reserved_ids = [r["entity_id"] for r in reserved]
        if not set(reserved_ids) <= set(ids) or sha(path) != plan["cohort_fingerprints"][role]["references_sha256"]:
            raise ValueError("Fresh evaluation cohort absent from full scored graph")
        cohorts[role] = {"references": len(reserved_ids), "references_sha256": sha(path), "complete_claimant_scores_available": True}
    return {"kind": "complete_heldout", "complete": True, "scoring_complete": True,
            "supervised_training_excluded": True, "declared_references": len(ids), "scored_references": offset,
            "pair_bearing_references": len(encountered), "zero_candidate_references": len(empty),
            "zero_candidate_reference_ids_sha256": hashlib.sha256("\n".join(empty).encode()).hexdigest(),
            "pairs": pair_count, "processed_chunks": len(manifest["chunks"]),
            "score_manifest_sha256": sha(manifest_path), "pairs_sha256": manifest["pairs_sha256"],
            "pair_order_sha256": manifest["pair_order_sha256"], "raw_reference_records_sha256": sha(raw),
            "frozen_control_sha256": sha(frozen), "reservation_plan_sha256": sha(reservation),
            "prepared_at": datetime.now(timezone.utc).isoformat(), "fresh_cohorts": cohorts,
            "labels_read": False, "exposed_holdout_claimants_retained": True,
            "fit_owner_exclusion_evidence": {"incumbent_fitted_owner_ids_sha256": manifest["incumbent_fitted_owner_ids_sha256"],
                                             "capture_script_sha256": manifest["capture_script_sha256"],
                                             "note": "Capture verifies the shared current incumbent fit-owner ledger disjoint from every heldout reference."},
            "limitation": "Complete declared outer-heldout claim universe; official-test claimant density and country mix differ. Only reserved previously unexposed cohorts may supply fresh quality evidence."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workbench", type=Path, required=True); p.add_argument("--reservation-plan", type=Path, required=True)
    p.add_argument("--frozen", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = build(args.workbench, args.reservation_plan, args.frozen)
    with args.output.open("x") as stream: json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
