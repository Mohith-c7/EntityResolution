"""Attach completed parity and asset evidence to a finished keyed workbench."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from run_frozen_pipeline import digest, load_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbench", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    args = parser.parse_args()
    load_frozen(args.frozen)
    path = args.workbench / "manifest.json"
    result = json.loads(path.read_text())
    proof = json.loads(args.parity.read_text())
    assets = json.loads(args.assets.read_text())
    if result["status"] != "complete" or proof["status"] != "parity_verified" or assets["status"] != "verified":
        raise ValueError("Incomplete score, asset or parity proof")
    capture = digest(Path(__file__).with_name("score_workbench.py"))
    if not result["capture_script_sha256"] == proof["capture_script_sha256"] == capture:
        raise ValueError("Captured code differs from verified replay")
    if not result["frozen_sha256"] == assets["frozen_sha256"] == digest(args.frozen):
        raise ValueError("Frozen pipeline differs from verified assets")
    if not proof["pair_ids_and_order_exact"] or max(proof["max_score_difference"].values()) > 1e-6:
        raise ValueError("Replay did not preserve keys and scores")
    pairs = args.workbench / "pairs.parquet"
    if digest(pairs) != result["pairs_sha256"]:
        raise ValueError("Finished workbench checksum mismatch")
    frame = pd.read_parquet(pairs, columns=["source1_entity_id", "candidate_entity_id"])
    h = hashlib.sha256()
    for s, c in frame.itertuples(index=False, name=None):
        h.update(f"{s}\t{c}\n".encode())
    if h.hexdigest() != result["pair_order_sha256"] or len(frame) != result["pairs"]:
        raise ValueError("Finished workbench pair identity mismatch")
    references = json.loads((args.workbench / "references.json").read_text())
    if [r["entity_id"] for r in references] != result["reference_ids"]:
        raise ValueError("Reference coverage differs from finished manifest")
    if sum(c["entities"] for c in result["chunks"]) != len(references):
        raise ValueError("Missing processed references, including empty candidates")
    result.update(verified_current_pipeline=True, verification_scope="Exact frozen-runner replay; keyed capture; model, alias, index and source asset verification",
                  parity_proof_sha256=digest(args.parity), asset_proof_sha256=digest(args.assets),
                  parity_proof=str(args.parity), asset_proof=str(args.assets))
    temporary = path.with_suffix(".sealed.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": "workbench_verified", "entities": result["entities"],
                      "pairs": result["pairs"], "workbench": str(args.workbench)}), flush=True)


if __name__ == "__main__":
    main()
