"""Write tuning-only retrieval/matcher error budgets from a completed run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code/business_entity_resolution"))
from src.evaluation.error_analysis import analyze_tuning


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir",type=Path,required=True)
    parser.add_argument("--summary-output",type=Path)
    args=parser.parse_args()
    result=analyze_tuning(args.artifact_dir)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True,exist_ok=True)
        args.summary_output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
