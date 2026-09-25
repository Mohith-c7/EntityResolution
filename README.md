# Amazon Business Entity Resolution 2026

## Overview
**Amazon ML Challenge 2026**

This project tackles business entity resolution at scale: identifying and matching business records referring to the same real-world entity across diverse and noisy data sources. The solution accurately resolves entities by linking variations in names, addresses, numeric identifiers, and localized formats without relying on external lookup APIs or oracle models.

## Locked Architecture

The finalized, locked system architecture comprises eight linear stages:

```
EDA
  ↓
Normalization
  ↓
Multi-Strategy Blocking
  ↓
Pairwise Features (36)
  ↓
LightGBM
  ↓
F0.5 Threshold Calibration
  ↓
Output Generation
  ↓
Submission Validation
```

### Key Architectural Constraints & Decisions
- **Multi-strategy blocking**: Union of diverse indices (rare-token name, address, digits, exact keys) achieving $\ge 95\%$ candidate recall with default `TOP_K = 20` per source.
- **36 Pairwise Features**: Comparison metrics spanning token overlaps, edit distances, phonetic matches, digit similarities, and null indicators.
- **LightGBM Classifier**: Trained with hard negatives mined during blocking, evaluated using entity-level disjoint validation splits (`RANDOM_SEED = 42`).
- **Metric & Optimization**: Precision-weighted Macro $F_{0.5}$ metric with threshold calibration across candidates.
- **No external APIs**: No external geocoding, business registries, or LLMs as oracles.
- **Open-set country handling**: Non-hardcoded country logic to generalize cleanly across geographies.

## High-Level Pipeline

1. **Ingestion & Validation**: Loading raw source data into validated schemas.
2. **Preprocessing & Normalization**: Standardizing case, punctuation, entity suffixes, and addresses.
3. **Candidate Blocking**: Generating a high-recall candidate pool using inverted indices and reranking.
4. **Feature Extraction**: Computing 36 pairwise attribute similarity features.
5. **Model Scoring & Calibration**: Predicting match probabilities and applying calibrated decision thresholds.
6. **Output Generation & Validation**: Emitting `matching_results.tsv` and `candidate_pairs.tsv` and validating format compliance.

## Project Structure

```text
amazon-entity-resolution-2026/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── output/
│
├── code/
│   └── business_entity_resolution/
│       ├── __init__.py
│       ├── run_pipeline.py
│       │
│       └── src/
│           ├── __init__.py
│           ├── config.py
│           │
│           ├── data/
│           │   ├── __init__.py
│           │   ├── loader.py
│           │   └── schema.py
│           │
│           ├── preprocessing/
│           │   ├── __init__.py
│           │   ├── normalize.py
│           │   └── preprocess.py
│           │
│           ├── blocking/
│           │   ├── __init__.py
│           │   ├── name_index.py
│           │   ├── address_index.py
│           │   ├── digit_index.py
│           │   ├── exact_blocks.py
│           │   ├── candidate_generator.py
│           │   └── reranker.py
│           │
│           ├── features/
│           │   ├── __init__.py
│           │   └── pairwise_features.py
│           │
│           ├── model/
│           │   ├── __init__.py
│           │   ├── train.py
│           │   ├── predict.py
│           │   └── threshold.py
│           │
│           ├── evaluation/
│           │   ├── __init__.py
│           │   ├── metrics.py
│           │   ├── blocking_recall.py
│           │   └── validation.py
│           │
│           ├── pipeline/
│           │   ├── __init__.py
│           │   └── run.py
│           │
│           └── utils/
│               ├── __init__.py
│               └── logging.py
│
├── experiments/
│   ├── E0_rule_baseline/
│   ├── E1_name_features/
│   ├── E2_full_features/
│   ├── E3_hard_negatives/
│   ├── E4_threshold/
│   ├── E5_source_threshold/
│   ├── E6_country_gate/
│   ├── E7_dense_blocking/
│   └── E8_transformer/
│
├── notebooks/
│
├── models/
│
├── reports/
│   ├── eda/
│   ├── blocking/
│   ├── experiments/
│   └── error_analysis/
│
├── utils/
│   └── validate_submission.py
│
├── README.md
├── requirements.txt
├── methodology.md
└── .gitignore
```

## Environment Setup

The repository uses Python 3.11.

1. **Create and activate the virtual environment:**
   ```powershell
   # Windows PowerShell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install core dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

*Note: Optional transformer/deep-learning libraries are deferred to future experiments.*

## How to Run the Pipeline

Execute the smoke-test pipeline entry point:

```powershell
python code/business_entity_resolution/run_pipeline.py
```

## Current Status

**Stage 0 — Repository Setup**

The repository structure, virtual environment, and configuration scaffolding are initialized.
Actual EDA on the challenge dataset is the next implementation step.
