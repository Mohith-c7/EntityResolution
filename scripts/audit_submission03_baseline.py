"""Score audit references with the frozen submission_03 package on the training indexes.

Runs submission_03's own inference code, model, aliases and threshold, so the
audit compares the new pipeline with exactly what was last submitted. No labels
are read here.
"""
import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SUBMISSION = Path("output/submission_03")
sys.path.insert(0, str(SUBMISSION))
import create_submission as frozen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, default=Path("models/audit_v2"))
    p.add_argument("--output", type=Path, default=Path("models/audit_v2/submission03_predictions.json"))
    p.add_argument("--workers", type=int, default=60)
    args = p.parse_args()
    rows = json.loads((args.audit / "sampled_references.json").read_text())["holdout"]
    rows = [{k: r[k] for k in ("entity_id", "business_name", "business_address", "country")} for r in rows]
    model = SUBMISSION / "model"
    report = json.loads((model / "report.json").read_text())
    initargs = (str((SUBMISSION / "code/business_entity_resolution").resolve()), str(model / "model.txt"),
                [str(Path(f"models/index_train_{s}.sqlite").resolve()) for s in ("S2", "S3")],
                report["search_config"], report["threshold"], report["feature_version"], str(model / "name_aliases.json"),
                0, report.get("retrieval"), None)
    batches = [rows[i:i + 100] for i in range(0, len(rows), 100)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=frozen.initialize, initargs=initargs) as pool:
        results = [r for part in pool.map(frozen.score, batches) for r in part]
    output = {eid: {"candidates": candidates, "predictions": predictions} for eid, candidates, predictions in results}
    if set(output) != {r["entity_id"] for r in rows}: raise ValueError("Missing audit references")
    args.output.write_text(json.dumps(output))
    print(json.dumps({"references": len(output), "predicted_links": sum(len(v["predictions"]) for v in output.values())}))


if __name__ == "__main__":
    main()
