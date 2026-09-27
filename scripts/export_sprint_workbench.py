"""Export checkpoint scores with immutable ordered references; never read truth.

Examples: --chunks run/chunks --references ordered_references.json
--lineage lineage.json --output workbench. Reference ordering must match the
original scoring task; zero-candidate references cannot be inferred from pairs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from decision_policy_v2 import PAIR_KEYS, validate_claims


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def order_hash(frame):
    h = hashlib.sha256()
    for row in frame[PAIR_KEYS].itertuples(index=False, name=None):
        h.update((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    return h.hexdigest()


def reference_rows(path):
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        if "ordered_references" in raw: raw = raw["ordered_references"]
        elif len(raw) == 1 and isinstance(next(iter(raw.values())), list): raw = next(iter(raw.values()))
        else: raise ValueError("Supply a single ordered list, not an ambiguous multi-split manifest")
    if not isinstance(raw, list): raise ValueError("References must be an ordered JSON list")
    rows = [{"entity_id": v} if isinstance(v, str) else v for v in raw]
    ids = [r["entity_id"] for r in rows]
    if len(set(ids)) != len(ids): raise ValueError("Duplicate references")
    return rows


def export_workbench(chunks, references, output, lineage):
    rows = reference_rows(references); ids = [r["entity_id"] for r in rows]
    refs = {r["entity_id"]: r for r in rows}; frames, sources, offset = [], [], 0
    paths = sorted(Path(chunks).glob("*.parquet"))
    if not paths: raise ValueError("No checkpoint score chunks")
    for path in paths:
        marker_path = path.with_suffix(".json")
        if not marker_path.exists(): raise ValueError(f"Missing marker: {marker_path}")
        marker = json.loads(marker_path.read_text())
        if sha(path) != marker["parquet_sha256"]: raise ValueError(f"Parquet hash mismatch: {path}")
        expected_ids = ids[offset:offset + marker["entities"]]; offset += marker["entities"]
        expected_hash = hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()
        if len(expected_ids) != marker["entities"] or expected_hash != marker["input_ids_sha256"]:
            raise ValueError(f"Immutable reference order does not match scoring task: {path}")
        frame = pd.read_parquet(path)
        required = PAIR_KEYS + ["first_stage", "second_stage", "probability"]
        if not set(required) <= set(frame): raise ValueError("Chunk lacks complete stored stage scores")
        if len(frame) != marker["pairs"]: raise ValueError("Chunk pair count mismatch")
        validate_claims(frame)
        if not set(frame.source1_entity_id) <= set(expected_ids): raise ValueError("Chunk has foreign references")
        # Stored rows must follow reference task order, with each reference contiguous.
        encountered = list(dict.fromkeys(frame.source1_entity_id))
        if encountered != [e for e in expected_ids if e in set(encountered)]: raise ValueError("Chunk reference order mismatch")
        for stage in ("first_stage", "second_stage"):
            if not np.isfinite(frame[stage]).all() or not frame[stage].between(0, 1).all(): raise ValueError("Invalid stage probabilities")
        frame = frame[required].copy()
        frame["candidate_order"] = frame.groupby("source1_entity_id", sort=False).cumcount()
        frame["candidate_source"] = frame.candidate_entity_id.str.split("-", n=1).str[0]
        frame["country"] = frame.source1_entity_id.map(lambda e: refs[e].get("country", "unknown"))
        frame["retrieval_provenance"] = "frozen_checkpoint_order"
        frame["proposed"] = False; frame["accepted"] = False
        frame["decision_state"] = "not_evaluated"
        frame["chunk"] = marker["chunk"]
        frames.append(frame)
        sources.append({"path": str(path), "sha256": marker["parquet_sha256"], "marker_sha256": sha(marker_path),
                        "input_ids_sha256": expected_hash, "entities": marker["entities"], "pairs": marker["pairs"]})
    if offset != len(ids): raise ValueError("Reference manifest has unscored or extra references")
    result = pd.concat(frames, ignore_index=True); validate_claims(result)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    result.to_parquet(output / "pairs.parquet", index=False)
    (output / "references.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    metadata = json.loads(Path(lineage).read_text())
    expected = {"input", "normalization", "aliases", "indexes", "native_library", "feature_schema", "models"}
    missing = sorted(k for k in expected if not metadata.get(k))
    report = {"status": "pair_keyed_export_complete" if not missing else "export_lineage_incomplete", "pairs": len(result),
              "references": len(rows), "zero_candidate_references": len(set(ids) - set(result.source1_entity_id)),
              "ordered_pair_keys_sha256": order_hash(result), "ordered_references_sha256": sha(references),
              "output_sha256": {"pairs.parquet": sha(output / "pairs.parquet"), "references.json": sha(output / "references.json")},
              "chunks": sources, "lineage": metadata, "missing_lineage_components": missing,
              "feature_availability": "stored stage scores only; pair/context feature matrices absent",
              "truth": "not read; provide separately at authorized evaluation"}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return result, report


def compare_replay(current, saved, tolerance=1e-6):
    """Require exact pairs/order, then compare stage scores. No equal-length joins."""
    validate_claims(current); validate_claims(saved)
    if not current[PAIR_KEYS].equals(saved[PAIR_KEYS]): raise ValueError("Replay pair IDs/order differ")
    if "candidate_order" in current and "candidate_order" in saved and not np.array_equal(current.candidate_order, saved.candidate_order):
        raise ValueError("Replay candidate order differs")
    differences = {c: float(np.max(np.abs(current[c].to_numpy() - saved[c].to_numpy()), initial=0.))
                   for c in ("first_stage", "second_stage", "probability")}
    return {"pairs": len(current), "exact_pair_order": True, "max_score_differences": differences,
            "score_parity": all(v <= tolerance for v in differences.values()), "tolerance": tolerance}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", type=Path, required=True); p.add_argument("--references", type=Path, required=True)
    p.add_argument("--lineage", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); _, report = export_workbench(args.chunks, args.references, args.output, args.lineage)
    print(json.dumps({k: v for k, v in report.items() if k not in ("chunks", "lineage")}))

if __name__ == "__main__": main()
