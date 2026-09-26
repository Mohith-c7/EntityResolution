"""Test feature reliance on saved training and tuning candidate pairs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code/business_entity_resolution"))
from src.model.experiments import run_experiments


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    run_experiments(args.artifact_dir,args.output_dir,threads=args.threads)
