"""Verify copied scoring assets and record provenance before sprint scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_frozen_pipeline as frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    args = parser.parse_args()
    frozen.load_frozen(args.frozen)
    evidence = {"status": "verified", "frozen_sha256": frozen.digest(args.frozen),
                "files": {}, "indexes": {}, "oof": []}
    for source in ("S2", "S3"):
        index = Path(f"models/index_train_{source}.sqlite")
        raw = Path(f"dataset/train/train_source{source[-1]}.tsv")
        with sqlite3.connect(index.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            meta = json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
        if not meta["complete"] or meta["fingerprint"]["source"] != source:
            raise ValueError("Incomplete or incorrect target index")
        if meta["fingerprint"]["size"] != raw.stat().st_size:
            raise ValueError("Target raw file and index metadata do not agree")
        evidence["indexes"][source] = meta
        for path in (raw, index, Path(str(index) + ".frequencies.sqlite")):
            evidence["files"][str(path)] = {"size": path.stat().st_size, "sha256": frozen.digest(path)}
    raw = Path("dataset/train/train_source1.tsv")
    raw_hash = frozen.digest(raw)
    counts = Path("models/audit_v2/run/universe_counts_train.sqlite")
    with sqlite3.connect(counts.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        counted = json.loads(conn.execute("SELECT value FROM meta WHERE key='universe'").fetchone()[0])
    if raw_hash != counted["sha256"]:
        raise ValueError("Competition counts belong to another Source 1 universe")
    evidence["counted_universe"] = counted
    manifest = json.loads(Path("models/bridge_v1/oof/manifest.json").read_text())
    if manifest["status"] != "complete" or {f["fold"] for f in manifest["folds"]} != set(range(5)):
        raise ValueError("Missing owner-excluded first-stage fold")
    for fold in manifest["folds"]:
        number = fold["fold"]
        model = Path(f"models/bridge_v1/oof/model_fold_{number}.txt")
        aliases = Path(f"models/scale_v1_plan/aliases/exclude_{number}.json")
        if frozen.digest(model) != fold["model_sha256"]:
            raise ValueError("OOF first-stage model checksum mismatch")
        alias_meta = json.loads(aliases.read_text())["metadata"]
        if alias_meta["excluded_inner_fold"] != number or alias_meta["fitted_fold"] != "train":
            raise ValueError("Alias fold is not excluded")
        evidence["oof"].append({"fold": number, "model_sha256": fold["model_sha256"],
                                "aliases_sha256": frozen.digest(aliases), "alias_metadata": alias_meta})
    for path in (raw, counts, frozen.NATIVE, Path(__file__),
                 Path("research/sprint_6h/coordination/score_workbench.py")):
        evidence["files"][str(path)] = {"size": path.stat().st_size, "sha256": frozen.digest(path)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"status": "verified", "output": str(args.output),
                      "train_source1_records": counted["records"], "oof_folds": len(evidence["oof"])}), flush=True)


if __name__ == "__main__":
    main()
