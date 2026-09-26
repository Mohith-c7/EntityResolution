"""
End-to-end entity resolution pipeline entry point.

Wires together:
  - Loader (Mohit)            : load_source(), load_ground_truth()
  - Preprocessing (Sanhita)   : normalization functions (consumed inside feature builder)
  - Blocking (Harsha)         : generate_candidates(source1, source2, source3, config)
  - Feature builder           : build_feature_matrix()
  - Training pipeline         : entity_level_split(), build_training_dataset(), train()
  - Threshold sweep           : sweep_threshold()
  - Decision layer            : select_matches(), write_outputs()
  - Output validation         : validate_outputs()

Usage:
    python run_pipeline.py --output_dir outputs/
"""
from __future__ import annotations

import argparse
import logging
import os
import random

import numpy as np
import pandas as pd

# Fix all random seeds at pipeline entry point (Requirements: all)
random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(output_dir: str) -> None:
    """
    Execute the full entity resolution pipeline.

    Steps
    -----
    1.  Load source records (S1, S2, S3) via Mohit's loader.
    2.  Generate blocking candidates via Harsha's blocking module.
    3.  Adapt column names to internal conventions.
    4.  Build feature matrix for all candidate pairs.
    5.  Split into entity-level train / validation partitions.
    6.  Build training dataset (positives + hard negatives) for train split.
    7.  Build feature matrix for validation split (all pairs, for threshold sweep).
    8.  Train LightGBM model.
    9.  Sweep threshold to maximise validation macro F0.5.
    10. Score all candidate pairs.
    11. Select final matches (zero or many per S1, threshold-gated).
    12. Write output TSV files (release format).
    13. Validate output consistency.
    14. Persist trained model and feature schema.
    """
    from src.integration.adapters import (
        adapt_candidates,
        adapt_ground_truth,
        adapt_source_records,
    )

    # ------------------------------------------------------------------
    # Step 1: Load source records and ground truth
    # ------------------------------------------------------------------
    # TODO: uncomment when Mohit's loader is available
    # from src.data.loader import (
    #     load_train_source1, load_train_source2, load_train_source3, load_ground_truth
    # )
    # s1_raw = load_train_source1()
    # s2_raw = load_train_source2()
    # s3_raw = load_train_source3()
    # gt_raw = load_ground_truth()
    logger.info("Step 1: Loading source records — awaiting Mohit's loader integration.")
    s1_raw = s2_raw = s3_raw = gt_raw = None

    if s1_raw is None:
        logger.warning(
            "Pipeline halted: Mohit's loader is not yet integrated. "
            "Uncomment the loader calls in Step 1 above."
        )
        return

    # Adapt to internal format
    source_records = adapt_source_records(s1_raw, s2_raw, s3_raw)
    ground_truth = adapt_ground_truth(gt_raw)  # columns: s1_id, match_id
    logger.info(
        "Loaded %d S1 / %d S2 / %d S3 records; %d ground-truth pairs.",
        len(source_records["S1"]),
        len(source_records["S2"]),
        len(source_records["S3"]),
        len(ground_truth),
    )

    # ------------------------------------------------------------------
    # Step 1b: Preprocess source records (Sanhita's preprocessing)
    # Harsha's generate_candidates requires norm_name, name_core,
    # norm_address, digit_tokens, norm_country on all DataFrames.
    # ------------------------------------------------------------------
    # TODO: uncomment when integrated
    # from src.preprocessing.preprocess import preprocess_dataframe
    # s1_prep = preprocess_dataframe(s1_raw)
    # s2_prep = preprocess_dataframe(s2_raw)
    # s3_prep = preprocess_dataframe(s3_raw)
    # s1_prep = preprocess_source(s1_raw)
    # s2_prep = preprocess_source(s2_raw)
    # s3_prep = preprocess_source(s3_raw)
    logger.info("Step 1b: Preprocessing source records — awaiting Sanhita's preprocess_source.")
    s1_prep = s2_prep = s3_prep = None

    if s1_prep is None:
        logger.warning(
            "Pipeline halted: Sanhita's preprocess_source is not yet integrated. "
            "Uncomment the preprocess_source calls in Step 1b above."
        )
        return

    # ------------------------------------------------------------------
    # Step 2: Generate candidates via blocking
    # ------------------------------------------------------------------
    # generate_candidates expects preprocessed DataFrames with columns:
    #   entity_id, norm_name, name_core, norm_address, digit_tokens, norm_country
    # Signature: generate_candidates(source1, source2, source3, *, top_k=20)
    # TODO: uncomment when Harsha's blocking module is available
    # from src.blocking.blocking import generate_candidates
    # harsha_candidates = generate_candidates(s1_prep, s2_prep, s3_prep, top_k=20)
    logger.info("Step 2: Generating candidates — awaiting Harsha's blocking integration.")
    harsha_candidates = None

    if harsha_candidates is None:
        logger.warning(
            "Pipeline halted: Harsha's blocking module is not yet integrated. "
            "Uncomment the generate_candidates call in Step 2 above."
        )
        return

    # Adapt Harsha's column names to internal convention
    candidate_pairs = adapt_candidates(harsha_candidates)
    logger.info("Step 2: %d candidate pairs generated.", len(candidate_pairs))

    # ------------------------------------------------------------------
    # Step 3: Build feature matrix for all candidate pairs
    # ------------------------------------------------------------------
    from src.features.pairwise_features import FEATURE_NAMES, _DEFAULTS, build_feature_matrix

    logger.info("Step 3: Building feature matrix for all %d candidate pairs.", len(candidate_pairs))
    feature_matrix, _ = build_feature_matrix(candidate_pairs, source_records)

    # ------------------------------------------------------------------
    # Step 4: Entity-level split
    # ------------------------------------------------------------------
    from src.model.train import entity_level_split

    logger.info("Step 4: Entity-level train/val split.")
    train_pairs, val_pairs = entity_level_split(
        candidate_pairs,
        seed=42,
        split_manifest_path="reports/splits/split_manifest.json",
    )

    # ------------------------------------------------------------------
    # Step 5: Build training dataset (positives + hard negatives)
    # ------------------------------------------------------------------
    from src.model.train import build_training_dataset

    logger.info("Step 5: Building training dataset (positives + hard negatives).")
    train_X, train_y = build_training_dataset(train_pairs, ground_truth, source_records)

    # ------------------------------------------------------------------
    # Step 6: Build validation feature matrix (all val pairs)
    # ------------------------------------------------------------------
    logger.info("Step 6: Building validation feature matrix (%d pairs).", len(val_pairs))
    val_X, _ = build_feature_matrix(val_pairs, source_records)

    # Val labels for LightGBM monitoring
    gt_keys: set[tuple] = set(zip(ground_truth["s1_id"], ground_truth["match_id"]))

    def _is_positive(row: pd.Series) -> bool:
        match_id = row.get("s2_id") or row.get("s3_id")
        return (row["s1_id"], match_id) in gt_keys

    val_labels = val_pairs.apply(_is_positive, axis=1).astype(float).values

    # ------------------------------------------------------------------
    # Step 7: Train LightGBM
    # ------------------------------------------------------------------
    from src.model.train import train

    logger.info("Step 7: Training LightGBM.")
    model = train(train_X, train_y, val_X, val_labels)

    # ------------------------------------------------------------------
    # Step 8: Sweep threshold on validation set
    # ------------------------------------------------------------------
    from src.model.predict import predict
    from src.model.threshold import sweep_threshold

    logger.info("Step 8: Sweeping threshold on validation set.")
    val_probs = predict(model, val_X)
    best_threshold, best_score = sweep_threshold(val_pairs, val_probs, ground_truth)
    logger.info("Best threshold: %.4f, macro F0.5: %.4f", best_threshold, best_score)

    # ------------------------------------------------------------------
    # Step 9: Score all candidate pairs
    # ------------------------------------------------------------------
    logger.info("Step 9: Scoring all candidate pairs.")
    all_probs = predict(model, feature_matrix)

    candidate_pairs = candidate_pairs.copy()
    candidate_pairs["match_probability"] = all_probs

    # ------------------------------------------------------------------
    # Step 10: Select final matches
    # ------------------------------------------------------------------
    from src.model.predict import select_matches

    # Pass all S1 IDs so singletons with no candidates still get output rows
    all_s1_ids = source_records["S1"].index.tolist()
    logger.info("Step 10: Selecting final matches (threshold=%.4f).", best_threshold)
    matches = select_matches(candidate_pairs, all_probs, best_threshold, all_s1_ids=all_s1_ids)

    # ------------------------------------------------------------------
    # Step 11: Write output files (release format)
    # ------------------------------------------------------------------
    from src.model.predict import write_outputs

    logger.info("Step 11: Writing output files to %s.", output_dir)
    write_outputs(matches, candidate_pairs, output_dir)

    # ------------------------------------------------------------------
    # Step 12: Validate outputs
    # ------------------------------------------------------------------
    from src.evaluation.validation import validate_outputs

    logger.info("Step 12: Validating outputs.")
    validate_outputs(matches, candidate_pairs)

    # ------------------------------------------------------------------
    # Step 13: Persist model and feature schema
    # ------------------------------------------------------------------
    from src.model.train import save_feature_schema, save_model

    model_path = os.path.join(output_dir, "model.txt")
    schema_path = os.path.join(output_dir, "feature_schema.json")

    defaults = dict(zip(FEATURE_NAMES, _DEFAULTS))
    save_model(model, best_threshold, model_path)
    save_feature_schema(FEATURE_NAMES, defaults, schema_path)

    logger.info("Pipeline complete. Outputs written to %s.", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run entity resolution pipeline.")
    parser.add_argument("--output_dir", default="outputs", help="Output directory for TSV files.")
    args = parser.parse_args()
    run(args.output_dir)
