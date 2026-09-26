"""Train and evaluate an offline baseline against complete target indexes."""

import argparse
from pathlib import Path

from src.blocking.disk_index import DiskSearchConfig
from src.pipeline.training import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--index-dir", type=Path, default=Path("models"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--train-entities", type=int, default=2000)
    parser.add_argument("--tune-entities", type=int, default=500)
    parser.add_argument("--holdout-entities", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--character-mode", choices=("always", "fallback", "off"), default="always")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reranker-version", choices=("v1", "v2", "v3", "v4"), default="v1")
    parser.add_argument("--numeric-conjunctions", action="store_true")
    parser.add_argument("--max-address-df", type=int, default=10000)
    parser.add_argument("--numeric-location-queries", action="store_true")
    parser.add_argument("--use-name-aliases", action="store_true")
    parser.add_argument("--retrieval-mode", choices=("wide","anchored"), default="wide")
    args = parser.parse_args()
    run_training(args.train_dir, args.index_dir, args.artifact_dir,
        train_entities=args.train_entities, tune_entities=args.tune_entities,
        holdout_entities=args.holdout_entities,
        config=DiskSearchConfig(top_k=args.top_k, character_mode=args.character_mode,
            reranker_version=args.reranker_version, numeric_conjunctions=args.numeric_conjunctions,
            max_address_df=args.max_address_df, numeric_location_queries=args.numeric_location_queries,
            use_name_aliases=args.use_name_aliases, retrieval_mode=args.retrieval_mode), threads=args.threads,workers=args.workers)


if __name__ == "__main__":
    main()
