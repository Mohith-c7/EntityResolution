"""Split the selection macro-F0.5 loss of a stacked model into error types."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("models/next_round/extended_context_cache"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--probabilities", type=Path, help="Final selection scores instead of the model blend")
    p.add_argument("--threshold", type=float, help="Threshold for --probabilities; defaults to the model's")
    args = p.parse_args()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    source = Path(manifest["source_cache"])
    refs, frames = {}, []
    for n in range(manifest["chunks"]):
        for row in json.loads((source / f"{n:06d}.json").read_text())["entity_rows"]: refs[row["entity_id"]] = row
        frames.append(pd.read_parquet(args.cache / f"{n:06d}.parquet", columns=["source1_entity_id", "label", "ctx_probability"]))
    frame = pd.concat(frames, ignore_index=True)
    ids = sorted(refs, key=lambda e: hashlib.sha256(("context-split-v1:" + e).encode()).digest())[30000:]
    frame = frame[frame.source1_entity_id.isin(set(ids))].reset_index(drop=True)
    chosen = json.loads((args.model / "report.json").read_text())["protocol_selection"]
    probability = (1 - chosen["context_weight"]) * frame.ctx_probability.to_numpy() + chosen["context_weight"] * np.load(args.model / "selection_probabilities.npy")
    threshold = chosen["threshold"]
    if args.probabilities: probability, threshold = np.load(args.probabilities), args.threshold or threshold
    frame["chosen"] = probability >= threshold
    n = len(ids)
    stats = frame.groupby("source1_entity_id").agg(tp=("chosen", lambda s: 0), n=("label", "size"))
    tp = frame[frame.chosen & (frame.label == 1)].groupby("source1_entity_id").size()
    fp = frame[frame.chosen & (frame.label == 0)].groupby("source1_entity_id").size()
    retrieved = frame[frame.label == 1].groupby("source1_entity_id").size()
    losses = {"singleton_false_positive": 0., "empty_with_true_candidate": 0., "empty_without_true_candidate": 0.,
              "false_links_in_matched": 0., "missed_retrieved_links": 0., "missed_unretrieved_links": 0.}
    for e in ids:
        t = len(refs[e]["true_ids"]); a, b, r = int(tp.get(e, 0)), int(fp.get(e, 0)), int(retrieved.get(e, 0))
        if t == 0:
            losses["singleton_false_positive"] += float(b > 0); continue
        if a + b == 0:
            losses["empty_with_true_candidate" if r else "empty_without_true_candidate"] += 1.; continue
        f = 1.25 * a / (1.25 * a + b + .25 * (t - a))
        f_clean = 1.25 * a / (1.25 * a + .25 * (t - a))
        f_retrievable = 1.25 * a / (1.25 * a + .25 * (r - a)) if r else 1.
        losses["false_links_in_matched"] += f_clean - f
        losses["missed_retrieved_links"] += 1. - f_retrievable
        losses["missed_unretrieved_links"] += f_retrievable - f_clean
    total = sum(losses.values())
    print(json.dumps({"macro_f05": 1 - total / n, **{k: round(v / n, 5) for k, v in losses.items()}}, indent=1))


if __name__ == "__main__":
    main()
