"""Seal the existing complete official-test score graph without rescoring it."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_gated_submission as reuse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["reuse_manifest", "test_dir", "output"]:
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Require unused output")
    started = time.monotonic()
    m = json.loads(a.reuse_manifest.read_text())
    rows, frame = reuse.load_verified_chunks(m, a.test_dir)
    ids = [row["entity_id"] for row in rows]
    if len(ids) != 1732544 or len(frame) != 69301760:
        raise ValueError("Official complete test population differs")
    # The verified baseline has exactly forty ordered candidates per reference.
    grouped = frame.source1_entity_id.to_numpy().reshape(len(ids), 40)
    if not (grouped == np.asarray(ids, dtype=object)[:, None]).all():
        raise ValueError("Original complete owner groups are not contiguous")
    frame["candidate_order"] = np.tile(np.arange(40, dtype=np.int16), len(ids))
    a.output.mkdir(parents=True)
    path = a.output / "pairs.parquet"
    frame.to_parquet(path, index=False)
    marker = {
        "status": "complete", "verified_current_pipeline": True,
        "split": "test", "role": "full_official_test", "reference_ids": ids,
        "pairs": len(frame), "pairs_sha256": reuse.digest(path),
        "pair_order_sha256": m["pair_order_sha256"],
        "frozen_sha256": m["baseline_frozen"]["sha256"],
        "parent_reuse_manifest_sha256": reuse.digest(a.reuse_manifest),
        "labels_read": False, "audit_labels_read": False,
        "source_sha256": reuse.digest(__file__),
        "seconds": time.monotonic() - started,
    }
    reuse.write_json(a.output / "manifest.json", marker)
    print(json.dumps({k: marker[k] for k in ["status", "pairs", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
