"""Serialize sealed routed pair keys from supplied records without labels."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            result.update(part)
    return result.hexdigest()


def serialize(record):
    return "name: " + record["business_name"] + " country: " + record["country"] + " address: " + record["business_address"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    marker = json.loads((a.features / "manifest.json").read_text())
    feature_file = a.features / "features.parquet"
    if marker.get("status") != "complete" or sha(feature_file) != marker["features_sha256"]:
        raise ValueError("Routed feature input is not sealed")
    keys = ["source1_entity_id", "candidate_entity_id"]
    frame = pd.read_parquet(feature_file, columns=keys)
    if frame.duplicated(keys).any() or frame[keys].isna().any().any():
        raise ValueError("Invalid routed pair keys")
    owners = set(frame.source1_entity_id); targets = set(frame.candidate_entity_id)
    records = {}; raw_hashes = {}
    for source in [1, 2, 3]:
        path = a.data_dir / f"train_source{source}.tsv"
        wanted = owners if source == 1 else targets
        raw_hashes[f"S{source}"] = sha(path)
        for batch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=100000):
            for row in batch.loc[batch.entity_id.isin(wanted)].to_dict("records"):
                if row["entity_id"] in records:
                    raise ValueError("Duplicate supplied record ID")
                records[row["entity_id"]] = serialize(row)
    if set(records) != owners | targets:
        raise ValueError("Missing supplied record")
    a.output.mkdir(parents=True, exist_ok=False)
    path = a.output / "pairs.jsonl"
    with path.open("x") as stream:
        for owner, target in frame.itertuples(index=False, name=None):
            stream.write(json.dumps({keys[0]: owner, keys[1]: target,
                "text_left": records[owner], "text_right": records[target]}, ensure_ascii=False) + "\n")
    manifest = {"status": "complete", "rows": len(frame), "labels_read": False,
        "input_sha256": sha(path), "feature_manifest_sha256": sha(a.features / "manifest.json"),
        "feature_file_sha256": sha(feature_file), "source_tsv_sha256": raw_hashes,
        "code_sha256": sha(__file__), "role": marker.get("role"), "source_split": marker.get("split")}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
