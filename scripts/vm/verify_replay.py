"""Check that this machine reproduces saved development candidates, features and scores."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_exclusive_assignment as eea
from build_context_features import CONTEXT_NAMES
from train_scale_model import FEATURE_NAMES_V3

cache = Path("models/next_round/extended_context_cache")
manifest = json.loads((cache / "manifest.json").read_text())
source = Path(manifest["source_cache"])
refs = {}
for n in range(manifest["chunks"]):
    for row in json.loads((source / f"{n:06d}.json").read_text())["entity_rows"]: refs[row["entity_id"]] = n
ids = sorted(refs, key=lambda e: hashlib.sha256(("context-split-v1:" + e).encode()).digest())[30000:]
replay = sorted(ids, key=lambda e: hashlib.sha256(("replay:" + e).encode()).digest())[:int(sys.argv[1]) if len(sys.argv) > 1 else 250]
raw = {r["entity_id"]: r for r in json.loads(Path("models/scale_v1_plan/sampled_references.json").read_text())["tune"]}
eea.init(str(Path("models/scale_v1_plan/aliases/full.json").resolve()), "models/next_round/extended_model/model.txt")
fresh = eea.score_chunk([raw[e] for e in replay])
columns = ["source1_entity_id", "candidate_entity_id", *FEATURE_NAMES_V3, *CONTEXT_NAMES]
saved = pd.concat([pd.read_parquet(cache / f"{n:06d}.parquet", columns=columns) for n in sorted({refs[e] for e in replay})])
saved = saved[saved.source1_entity_id.isin(set(replay))]
merged = fresh.merge(saved, on=["source1_entity_id", "candidate_entity_id"], how="outer", suffixes=("_vm", "_saved"), indicator=True)
both = merged[merged._merge == "both"]
worst = {n: float((both[n + "_vm"] - both[n + "_saved"]).abs().max()) for n in [*FEATURE_NAMES_V3, *CONTEXT_NAMES]}
print(json.dumps({"references": len(replay), "pairs_both": len(both), "only_vm": int((merged._merge == "left_only").sum()),
                  "only_saved": int((merged._merge == "right_only").sum()),
                  "max_probability_difference": worst["ctx_probability"],
                  "features_with_difference": {k: v for k, v in worst.items() if v > 1e-6}}, indent=1))
