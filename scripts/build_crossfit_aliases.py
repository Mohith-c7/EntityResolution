"""Build outer-training alias counts and five cross-fitted alias models."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.model.crossfit_aliases import build_counts, export_aliases

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("models/scale_v1_plan/aliases"))
    p.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    p.add_argument("--index-dir", type=Path, default=Path("models"))
    args = p.parse_args(); os.nice(10)
    database = (args.output_dir / "counts.sqlite").resolve()
    build_counts(args.train_dir, args.index_dir, database)
    export_aliases(database, args.output_dir / "full.json")
    for fold in range(5): export_aliases(database, args.output_dir / f"exclude_{fold}.json", fold)

if __name__ == "__main__": main()
