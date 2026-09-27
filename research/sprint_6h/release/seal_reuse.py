#!/usr/bin/env python3
"""Read-only complete-baseline seal; run only after baseline job releases VM1."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("release_builder", ROOT / "scripts/build_gated_submission.py")
builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline",type=Path,required=True)
parser.add_argument("--test-dir",type=Path,required=True)
parser.add_argument("--frozen",type=Path,required=True)
parser.add_argument("--asset-hashes",type=Path,required=True,help="Immutable native/index path -> SHA-256 evidence JSON")
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
manifest=builder.seal_reuse_manifest(args.baseline,args.test_dir,args.frozen,json.loads(args.asset_hashes.read_text()),args.output)
print(json.dumps({"manifest":str(args.output),"sha256":builder.digest(args.output),"entities":manifest["entities"],"pairs":manifest["pairs"],"chunks":len(manifest["chunks"])},indent=2))
