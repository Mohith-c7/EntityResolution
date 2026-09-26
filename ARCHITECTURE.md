# Business Entity Resolution — Amazon ML Challenge 2026
## Final Locked Architecture & System Design

> **Version:** 1.0 — Final  
> **Status:** Locked for Implementation  
> **Metric:** Macro F₀.₅ (precision-weighted, per Source 1 entity, averaged)

---

## Table of Contents

1. [Problem Restatement & Constraints](#1-problem-restatement--constraints)
2. [Core Design Principles](#2-core-design-principles)
3. [Architecture Overview](#3-architecture-overview)
4. [Stage 0 — Data Loading & EDA](#stage-0--data-loading--eda)
5. [Stage 1 — Preprocessing & Normalization](#stage-1--preprocessing--normalization)
6. [Stage 2 — Blocking (Candidate Generation)](#stage-2--blocking-candidate-generation)
7. [Stage 3 — Pairwise Feature Engineering](#stage-3--pairwise-feature-engineering)
8. [Stage 4 — LightGBM Pair Scorer](#stage-4--lightgbm-pair-scorer)
9. [Stage 5 — Decision Layer & Threshold Calibration](#stage-5--decision-layer--threshold-calibration)
10. [Stage 6 — Output & Validation](#stage-6--output--validation)
11. [Validation Strategy](#validation-strategy)
12. [Experiment Ladder](#experiment-ladder)
13. [Project Directory Structure](#project-directory-structure)
14. [Toolchain & Dependencies](#toolchain--dependencies)
15. [72-Hour Timeline](#72-hour-timeline)
16. [What We Are NOT Doing (and Why)](#what-we-are-not-doing-and-why)
17. [Architecture Validation Against Problem Statement](#architecture-validation-against-problem-statement)

---

## 1. Problem Restatement & Constraints

### What must be produced

| Output File | Description | Scored? |
|---|---|---|
| `output/matching_results.tsv` | One row per S1 entity. `matched_entity_ids` = comma-separated S2/S3 IDs that match. Empty if singleton. | ✅ Yes — this drives the leaderboard |
| `output/candidate_pairs.tsv` | One row per S1 entity. All S2/S3 IDs your model *considered* before making the final decision. | ❌ Not scored — used by Amazon to audit blocking quality |

### Hard rules from the problem statement

- Every S1 entity must have exactly one row in both output files.
- `matched_entity_ids` contains only S2/S3 IDs — never S1 IDs.
- No duplicate IDs within a single row's list.
- All IDs in `matching_results.tsv` must also appear in `candidate_pairs.tsv` (subset rule).
- A match that was never a candidate is a pipeline bug — the validator will warn.
- Country is **open-set**: France appears only in test, never in train. Do not hard-code or one-hot country.
- No external APIs, geocoding, or business registry lookups. Code must be fully reproducible from provided data.
- Final model must be ≤ 8B parameters, MIT or Apache 2.0 license.
- Run `utils/validate_submission.py` locally before every leaderboard upload.
- Tab-separated files only. Never read `.tsv` without `sep="\t"`.

### Metric

```
F₀.₅ = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- Computed per S1 entity, then **macro-averaged** across all S1 entities.
- Precision is weighted **2× over recall**.
- Singletons score **1.0** when predicted empty, **0.0** when any match is predicted.
- One false merge (wrong match) damages score more than one missed match.
- **Strategic implication: default to NOT matching when uncertain.**

---

## 2. Core Design Principles

### Principle 1 — Blocking recall is the hard ceiling
If a true match is not in the candidate set, no downstream component can recover it. No model, no matter how sophisticated, can match what it never sees.
> **Target: ≥ 95% blocking recall before training any matcher.**

### Principle 2 — Precision over recall at every decision point
The F₀.₅ metric explicitly weights precision 2× over recall. Every architectural decision must default to precision.
> **When uncertain between two designs: choose the one that generates fewer false positives.**

### Principle 3 — Singletons are free points
A correct empty prediction scores 1.0 for that entity. False merges on singletons score 0.0.
> **Never force a match when the model is not confident.**

### Principle 4 — Country is an open-set string
France is in the test set but not the training set. No country-specific code paths.
> **Use country as a plain string feature. Never hard-code country values.**

### Principle 5 — No over-engineering
LightGBM + rich features beats a complex Transformer pipeline that fails to run in 72 hours.
> **Build the simple thing well. Add complexity only when validation proves it helps.**

---

## 3. Architecture Overview

```
RAW TSV FILES (train_source1, train_source2, train_source3, train_ground_truth)
(test_source1, test_source2, test_source3)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 0 — EDA & Data Profiling                             │
│  Understand: singleton rate, noise patterns, country dist.  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1 — Preprocessing & Normalization                    │
│  lowercase · abbreviation expansion · legal suffix strip    │
│  retain raw fields alongside normalized fields              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2 — BLOCKING (Multi-Strategy, Union)                 │
│                                                             │
│  Path A: Rare-token inverted index on name tokens           │
│  Path B: Rare-token inverted index on address tokens        │
│  Path C: Rare-token inverted index on digit/number tokens   │
│  Path D: Exact normalized name match                        │
│  Path E: Exact name-core match (suffix-stripped)            │
│  Path F: Name-core prefix block (first 8 chars)             │
│  Path G: Postcode + name token block                        │
│                                                             │
│  → Union all 7 paths                                        │
│  → Fused symmetric rerank with exactness bonuses            │
│  → Per-source top-k cap (configurable, default 20)          │
│  → Measure blocking recall on val holdout BEFORE training   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼  (candidate_pairs.tsv written here)
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3 — PAIRWISE FEATURE ENGINEERING (36 features)       │
│                                                             │
│  Name features (8):                                         │
│    jaro_winkler · levenshtein_ratio · fuzz_ratio            │
│    token_set_ratio · token_sort_ratio · partial_ratio       │
│    token_jaccard_exact · length_ratio                       │
│                                                             │
│  Address features (8):                                      │
│    jaro_winkler · levenshtein_ratio · fuzz_ratio            │
│    token_set_ratio · token_sort_ratio · partial_ratio       │
│    token_jaccard_exact · length_ratio                       │
│                                                             │
│  Digit/number features (4):                                 │
│    digit_token_jaccard · digit_exact_match                  │
│    numeric_overlap_ratio · postcode_match                   │
│                                                             │
│  Agreement flags (6):                                       │
│    country_exact_match · name_exact_match                   │
│    core_name_exact_match · any_token_exact_match            │
│    name_contains_address_token · address_contains_name_token│
│                                                             │
│  Length covariates (4):                                     │
│    s1_name_len · cand_name_len                              │
│    s1_address_len · cand_address_len                        │
│                                                             │
│  Blocking score (1): fused blocking similarity score        │
│  Source indicator (3): source_is_s2 / source_is_s3          │
│  Country agreement (2): open-set string match flags         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4 — LightGBM PAIR SCORER                             │
│                                                             │
│  Training:                                                  │
│    Positives: pairs from ground truth                       │
│    Hard negatives: top-k non-matching blocked candidates    │
│    Ratio: 1:2 positives to hard negatives (tunable)         │
│    Validation: entity-level split (seed=42, 15% holdout)    │
│                                                             │
│  Model: LightGBM binary classifier                          │
│    objective: binary                                        │
│    metric: binary_logloss                                   │
│    num_leaves: 63                                           │
│    n_estimators: 500 with early stopping                    │
│    class_weight: balanced                                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5 — DECISION LAYER & THRESHOLD CALIBRATION          │
│                                                             │
│  Threshold search:                                          │
│    Sweep 0.30 → 0.95 in steps of 0.01                      │
│    Evaluate macro F₀.₅ per S1 entity on holdout            │
│    Lock threshold that maximises F₀.₅ (expect 0.6–0.75)    │
│                                                             │
│  Optional: 2-D threshold search (separate threshold         │
│    for S2 candidates vs S3 candidates)                      │
│                                                             │
│  Singleton rule: if no candidate survives threshold → empty │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 6 — OUTPUT & FORMAT VALIDATION                       │
│                                                             │
│  Write matching_results.tsv                                 │
│  Write candidate_pairs.tsv                                  │
│  Run: python3 utils/validate_submission.py                  │
│    --matching output/matching_results.tsv                   │
│    --candidate output/candidate_pairs.tsv                   │
│    --test-dir dataset/test                                  │
│  Must print PASS before any leaderboard upload              │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage 0 — Data Loading & EDA

**Goal:** Understand the data before writing any model code.

### Tasks

```python
# Load with explicit tab separator — mandatory
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

### Key questions to answer before Stage 1

| Question | Why it matters |
|---|---|
| What fraction of S1 entities are singletons? | If >30%, singleton detection is a priority |
| Country distribution in train? | Understand India vs US split |
| Average matches per S1 entity? | Determines how aggressive blocking needs to be |
| What abbreviation patterns exist? | Drives the normalization dictionary |
| Average name length / address length? | Affects feature scaling |
| Are there null names or null addresses? | Need null-safe feature computation |
| What noise patterns appear most frequently? | Prioritize which similarity features to add |

### Outputs

- `notebooks/01_eda.ipynb` with all statistics
- `src/data/profiling.py` that produces a printed summary
- Ground truth analysis: distribution of match counts per S1 entity

---

## Stage 1 — Preprocessing & Normalization

**Goal:** Create clean, comparable representations of every record. Retain raw fields.

### Name normalization

```python
ABBREV_MAP = {
    "pvt": "private", "ltd": "limited", "llc": "llc",
    "corp": "corporation", "inc": "incorporated",
    "co": "company", "&": "and", "intl": "international",
    "tech": "technology", "svc": "services", "svcs": "services",
    "dept": "department", "mfg": "manufacturing",
    # Extend from EDA findings
}

def normalize_name(raw: str) -> str:
    s = raw.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)          # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()      # collapse whitespace
    tokens = s.split()
    tokens = [ABBREV_MAP.get(t, t) for t in tokens]
    return " ".join(tokens)

def name_core(normalized: str) -> str:
    """Strip common legal suffixes to get the business name core."""
    LEGAL_SUFFIXES = {"private", "limited", "llc", "corporation",
                      "incorporated", "company", "pvt", "ltd"}
    tokens = [t for t in normalized.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)
```

### Address normalization

```python
ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "apt": "apartment", "ste": "suite", "no": "number",
    # India-specific
    "nagar": "nagar", "marg": "marg",
}

def normalize_address(raw: str) -> str:
    if pd.isna(raw): return ""
    s = raw.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [ADDR_ABBREV.get(t, t) for t in s.split()]
    return " ".join(tokens)
```

### Record serialization (for use as text in optional transformer experiments)

```
COL business_name VAL {normalized_name} COL business_address VAL {normalized_address} COL country VAL {country}
```

### Fields retained per record

| Field | Description |
|---|---|
| `raw_name` | Original name as loaded |
| `norm_name` | Lowercased, punctuation removed, abbreviations expanded |
| `name_core` | Legal suffixes stripped |
| `raw_address` | Original address as loaded |
| `norm_address` | Lowercased, punctuation removed, abbreviations expanded |
| `digit_tokens` | Extracted numeric tokens (house numbers, postal codes) |
| `country` | Original country string — kept as-is (open set) |

### Current preprocessing contract

- Primary normalized text preserves Unicode letters, numbers, and combining marks.
  `accent_folded_name` and `accent_folded_address` provide Latin-accent-folded
  alternate views without removing Indic marks.
- `norm_address` expands the standard address abbreviations. The separate
  `norm_address_unexpanded` retains the punctuation-cleaned, unexpanded address
  view for blocking.
- `digit_tokens`, `name_tokens`, `address_tokens`, `name_numeric_tokens`,
  `address_numeric_tokens`, and `postcode_candidates` are `tuple[str, ...]`.
  Missing or empty inputs produce `()`. Postcode candidates are generic address
  numeric tokens with four or more digits, retaining leading zeros.
- `name_missing`, `address_missing`, and `country_missing` are boolean flags.
  Normalized string fields use `""` for missing input. Raw source columns are
  retained unchanged.

---

## Stage 2 — Blocking (Candidate Generation)

**Goal:** For every S1 entity, produce a small set of S2/S3 candidates that contains ≥ 95% of true matches.

### Why 7 paths?

Different noise types are caught by different blocking strategies. Unioning all paths maximizes recall without requiring any single path to be perfect.

| Path | Catches |
|---|---|
| A — Rare name tokens | Token overlap despite word-order changes |
| B — Rare address tokens | Address token overlap |
| C — Digit tokens | Numeric identifiers, postal codes |
| D — Exact norm name | Fast exact-match retrieval |
| E — Exact name core | Names that differ only in legal suffix |
| F — Name core prefix (8 chars) | Prefix-level name matches |
| G — Postcode + name token | Address-anchored blocking |

### Rare-token inverted index (Paths A, B, C)

Use token document frequency to identify **rare** tokens. High-frequency tokens (like "pvt", "road", "limited") are poor blocking keys because they match everything. Use only tokens appearing in fewer than `max_token_df` records (configurable, default 10% of corpus).

```python
# Build inverted index: token → list of record IDs
# For each S1 record, retrieve candidates sharing ≥ 1 rare token
# Compute dot-product similarity: query @ postings.T
# Use sorted token sets for determinism
```

### Fused symmetric rerank

After collecting raw candidates from all 7 paths:
1. Sum the path-specific scores for each (S1, candidate) pair
2. Add exactness bonuses: exact name match +2.0, exact core match +1.5
3. Apply per-source top-k cap (default k=20)
4. Keep the fused score as the `blocking_score` feature for Stage 3

### Diagnostic gate (run before training)

```bash
python run_pipeline.py diag
```

**Must report** before proceeding:
- Blocking recall on validation holdout ≥ 90% (target ≥ 95%)
- Average candidates per S1 entity (target: 10–50)
- Candidate reduction ratio

**If blocking recall < 90%: tune blocking before writing the matcher.**

### candidate_pairs.tsv is written here

Every S2/S3 record in the union becomes a candidate. This is the file Amazon uses to audit blocking quality. Write it here, before any filtering.

---

## Stage 3 — Pairwise Feature Engineering

**Goal:** For every (S1, candidate) pair, compute a 36-dimensional feature vector.

### Full feature list

```python
from rapidfuzz import fuzz, distance

def compute_pair_features(s1: dict, cand: dict) -> dict:
    n1, n2 = s1["norm_name"], cand["norm_name"]
    a1, a2 = s1["norm_address"], cand["norm_address"]

    return {
        # --- Name features (8) ---
        "name_jaro_winkler":   jaro_winkler(n1, n2),
        "name_levenshtein":    levenshtein_ratio(n1, n2),
        "name_fuzz_ratio":     fuzz.ratio(n1, n2) / 100,
        "name_token_set":      fuzz.token_set_ratio(n1, n2) / 100,
        "name_token_sort":     fuzz.token_sort_ratio(n1, n2) / 100,
        "name_partial_ratio":  fuzz.partial_ratio(n1, n2) / 100,
        "name_token_jaccard":  token_jaccard(n1, n2),
        "name_len_ratio":      safe_len_ratio(n1, n2),

        # --- Address features (8) ---
        "addr_jaro_winkler":   jaro_winkler(a1, a2),
        "addr_levenshtein":    levenshtein_ratio(a1, a2),
        "addr_fuzz_ratio":     fuzz.ratio(a1, a2) / 100,
        "addr_token_set":      fuzz.token_set_ratio(a1, a2) / 100,
        "addr_token_sort":     fuzz.token_sort_ratio(a1, a2) / 100,
        "addr_partial_ratio":  fuzz.partial_ratio(a1, a2) / 100,
        "addr_token_jaccard":  token_jaccard(a1, a2),
        "addr_len_ratio":      safe_len_ratio(a1, a2),

        # --- Digit/number features (4) ---
        "digit_jaccard":       token_jaccard(s1["digit_tokens"], cand["digit_tokens"]),
        "digit_exact":         int(s1["digit_tokens"] == cand["digit_tokens"]),
        "numeric_overlap":     numeric_overlap_ratio(s1, cand),
        "postcode_match":      int(postcode(s1) == postcode(cand)) if postcode(s1) else 0,

        # --- Agreement flags (6) ---
        "country_exact":       int(s1["country"] == cand["country"]),
        "name_exact":          int(n1 == n2),
        "core_name_exact":     int(s1["name_core"] == cand["name_core"]),
        "any_token_exact":     int(bool(set(n1.split()) & set(n2.split()))),
        "name_in_addr":        int(any(t in a2.split() for t in n1.split())),
        "addr_in_name":        int(any(t in n2.split() for t in a1.split())),

        # --- Length covariates (4) ---
        "s1_name_len":         len(n1),
        "cand_name_len":       len(n2),
        "s1_addr_len":         len(a1),
        "cand_addr_len":       len(a2),

        # --- Blocking score (1) ---
        "blocking_score":      cand["blocking_score"],

        # --- Source indicators (3) ---
        "source_is_s2":        int(cand["entity_id"].startswith("S2-")),
        "source_is_s3":        int(cand["entity_id"].startswith("S3-")),

        # --- Country open-set agreement (2) ---
        "country_nonempty":    int(bool(s1["country"]) and bool(cand["country"])),
        "country_mismatch":    int(bool(s1["country"]) and bool(cand["country"])
                                   and s1["country"] != cand["country"]),
    }
```

**Total: 36 features.** No embeddings at this stage. No transformer calls. Just fast Python.

### Hard negative mining

For each S1 entity during training:
- All ground-truth matches → **positives**
- Top-k blocked candidates that are NOT ground-truth matches → **hard negatives**
- Target ratio: 1:2 (positives:hard negatives). Adjust based on validation F₀.₅.

This teaches the model the actual decision boundary — "ABC Bank Ltd" vs "ABC Banking Services" — rather than easy cases.

---

## Stage 4 — LightGBM Pair Scorer

**Goal:** Learn a match probability for every candidate pair.

### Why LightGBM, not a Transformer?

| Factor | LightGBM | Transformer |
|---|---|---|
| Training time (72h budget) | Minutes | Hours |
| Feature interpretability | Full SHAP support | Black box |
| Handles missing values | Native | Requires preprocessing |
| Precision on tabular similarity features | Excellent | Not always better |
| Risk of overfitting | Low (tree regularization) | Higher (needs more data) |
| License | MIT | Varies |
| Parameter count | Thousands | 100M–8B |

LightGBM is the right primary model for this task. A Transformer cross-encoder is an optional upgrade for Stage 2 experiments only after the LightGBM baseline is validated.

### Configuration

```python
import lightgbm as lgb

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "class_weight": "balanced",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
```

### Training split

- **Entity-level split**, not row-level.
- `seed=42`, `validation_fraction=0.15` (fixed, deterministic).
- All candidate pairs for a given S1 entity go to either train or validation — never split across both.
- This prevents data leakage where the same S1 entity's easy pairs train the model and hard pairs validate it.

---

## Stage 5 — Decision Layer & Threshold Calibration

**Goal:** Convert pair probabilities into final entity-level match predictions, optimized for F₀.₅.

### Why threshold matters critically here

The default `probability > 0.5` threshold is calibrated for F₁, not F₀.₅. Because F₀.₅ weights precision 2× over recall, the optimal threshold for this metric is typically higher — often 0.60–0.75.

### Threshold search procedure

```python
def find_optimal_threshold(model, X_val, val_pairs, ground_truth):
    probs = model.predict_proba(X_val)[:, 1]

    best_threshold, best_f05 = 0.5, 0.0
    for threshold in np.arange(0.30, 0.96, 0.01):
        predictions = build_entity_predictions(val_pairs, probs, threshold)
        f05 = compute_macro_f05(predictions, ground_truth)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold

    return best_threshold, best_f05

def compute_macro_f05(predictions, ground_truth):
    """Exact reproduction of the challenge metric."""
    scores = []
    for s1_id in ground_truth:
        pred_set = set(predictions.get(s1_id, []))
        true_set = set(ground_truth[s1_id])

        if not pred_set and not true_set:
            scores.append(1.0)  # correct singleton
        elif not pred_set or not true_set:
            tp = len(pred_set & true_set)
            p = tp / len(pred_set) if pred_set else 0
            r = tp / len(true_set) if true_set else 0
            if p + r == 0:
                scores.append(0.0)
            else:
                scores.append(1.25 * p * r / (0.25 * p + r))
        else:
            tp = len(pred_set & true_set)
            p = tp / len(pred_set)
            r = tp / len(true_set)
            if p + r == 0:
                scores.append(0.0)
            else:
                scores.append(1.25 * p * r / (0.25 * p + r))

    return np.mean(scores)
```

### Optional: 2-D threshold search

Search separate thresholds for S2 and S3 candidates if diagnostics show their score distributions differ.

### Singleton rule

Any S1 entity where zero candidates survive the threshold → empty `matched_entity_ids`. This is the correct prediction for singleton entities and scores 1.0 for them.

---

## Stage 6 — Output & Validation

### Write matching_results.tsv

```python
with open("output/matching_results.tsv", "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1_id in all_test_s1_ids:  # every S1 entity must appear
        matches = final_predictions.get(s1_id, [])
        f.write(f"{s1_id}\t{','.join(matches)}\n")
```

### Write candidate_pairs.tsv

```python
with open("output/candidate_pairs.tsv", "w") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id in all_test_s1_ids:
        candidates = all_candidates.get(s1_id, [])
        f.write(f"{s1_id}\t{','.join(candidates)}\n")
```

### Validation (mandatory before every upload)

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Must print `PASS` (exit 0). Never upload if it prints issues.**

---

## Validation Strategy

### Entity-level split (not row-level)

```python
from sklearn.model_selection import train_test_split

all_s1_ids = list(train_ground_truth["source1_entity_id"].unique())
train_s1, val_s1 = train_test_split(
    all_s1_ids, test_size=0.15, random_state=42
)
```

All pairs for train S1 entities → training set.
All pairs for val S1 entities → validation set.

**Never mix pairs from the same S1 entity across train/val.**

### Metrics to track at every experiment

| Metric | Where measured | Acceptable range |
|---|---|---|
| Blocking recall | Val holdout before training | ≥ 90% (target: ≥ 95%) |
| Avg candidates per S1 | After blocking | 5–50 |
| Pair precision (val) | After LightGBM at threshold | ≥ 80% |
| Pair recall (val) | After LightGBM at threshold | ≥ 70% |
| **Entity macro F₀.₅ (val)** | **Primary metric** | **Maximize this** |
| Singleton correct rate (val) | Singletons predicted empty | ≥ 90% |

---

## Experiment Ladder

Work through experiments in order. Only advance when current stage is solid.

| Exp | Change | Expected F₀.₅ delta | Complexity |
|---|---|---|---|
| E0 | Rule baseline: blocking score threshold directly | — (baseline) | Trivial |
| E1 | + 8 name similarity features + LightGBM | +0.05–0.10 | Low |
| E2 | + full 36 feature set | +0.03–0.07 | Low |
| E3 | + hard negative mining (1:2 ratio) | +0.02–0.05 | Low |
| E4 | + F₀.₅-optimized threshold sweep | +0.01–0.04 | Trivial |
| E5 | + 2-D per-source threshold | +0.01–0.02 | Low |
| E6 | + country gate (data-driven, not hard-coded) | +0.00–0.03 | Low |
| E7 | + multilingual bi-encoder blocking path | +0.01–0.03 | Medium |
| E8 | + XLM-RoBERTa cross-encoder reranker | +0.02–0.05 | High |

**Stop at E5 or E6 if running low on time. The first 5 experiments are where most of the score comes from.**

---

## Project Directory Structure

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv       ← uploaded to leaderboard
│   └── candidate_pairs.tsv        ← submitted in final zip
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── data/
│       │   │   ├── loader.py          # TSV loading with tab separator
│       │   │   ├── schema.py          # Field definitions
│       │   │   └── profiling.py       # EDA statistics
│       │   │
│       │   ├── normalization/
│       │   │   ├── names.py           # Name normalization + core extraction
│       │   │   ├── addresses.py       # Address normalization
│       │   │   └── text.py            # Shared text utilities
│       │   │
│       │   ├── blocking/
│       │   │   ├── inverted_index.py  # Rare-token sparse index
│       │   │   ├── exact.py           # Exact match paths (D, E, F)
│       │   │   ├── postcode.py        # Path G
│       │   │   └── pipeline.py        # Union + rerank + topk cap
│       │   │
│       │   ├── features/
│       │   │   ├── name_features.py   # 8 name similarity features
│       │   │   ├── addr_features.py   # 8 address similarity features
│       │   │   ├── digit_features.py  # 4 digit/number features
│       │   │   ├── flags.py           # 6 agreement flags
│       │   │   └── pipeline.py        # Assemble full 36-feature vector
│       │   │
│       │   ├── models/
│       │   │   ├── lightgbm_scorer.py # Train, predict, feature importance
│       │   │   └── rule_scorer.py     # Baseline: threshold blocking score
│       │   │
│       │   ├── evaluation/
│       │   │   ├── metrics.py         # compute_macro_f05() — exact reproduction
│       │   │   ├── threshold.py       # Threshold sweep + 2-D search
│       │   │   └── diagnostics.py     # Blocking recall, candidate volume
│       │   │
│       │   └── output/
│       │       └── writer.py          # Write both TSV files
│       │
│       ├── run_pipeline.py            # CLI: all/prepare/block/diag/train/predict/validate/score
│       ├── README.md                  # Exact run instructions
│       └── requirements.txt           # Pinned versions
│
└── Documentation_template.md         # Filled methodology document
```

---

## Toolchain & Dependencies

```
# requirements.txt (pinned)
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
lightgbm==4.3.0
rapidfuzz==3.9.0       # Jaro-Winkler, Levenshtein, fuzz ratios
scipy==1.13.0          # Sparse matrix ops for blocking
pyarrow==16.0.0        # Parquet caching for normalized records
tqdm==4.66.4           # Progress bars
```

**No transformer libraries in the core pipeline.** If you add the optional E7/E8 experiments:

```
# Optional — only for experiments E7, E8
sentence-transformers==3.0.1   # MIT license
faiss-cpu==1.8.0               # MIT license
transformers==4.41.0           # Apache 2.0
torch==2.3.0                   # BSD license
```

---

## 72-Hour Timeline

| Hours | Milestone | Deliverable |
|---|---|---|
| 0–3 | Setup + EDA | `notebooks/01_eda.ipynb` complete, singleton rate known |
| 3–6 | Normalization layer | `src/normalization/` complete, all fields computed |
| 6–10 | Blocking implementation | 7 paths built, `diag` command running |
| 10–12 | **Blocking recall check** | ≥ 90% recall confirmed on val holdout |
| 12–16 | Feature engineering | All 36 features computing correctly |
| 16–20 | LightGBM training (E1–E2) | First model trained, val F₀.₅ measured |
| 20–22 | Hard negatives + retrain (E3) | Improved val F₀.₅ |
| 22–24 | Threshold calibration (E4–E5) | F₀.₅-optimal threshold locked |
| 24–25 | **First leaderboard submission** | Validate → upload |
| 25–35 | Error analysis + iteration | Analyze false merges and missed matches |
| 35–45 | Optional: E6 country gate | Data-driven, only if beneficial on val |
| 45–60 | Optional: E7 dense ANN blocking | Only if blocking recall < 93% |
| 60–65 | Final threshold re-tune | On full training data |
| 65–68 | Final submission + zip packaging | `scripts/make_submission_zip.sh` |
| 68–72 | Documentation template | Fill `Documentation_template.md` |

---

## What We Are NOT Doing (and Why)

| Excluded | Reason |
|---|---|
| External geocoding APIs | Explicitly prohibited — instant disqualification |
| Business registry lookups | Explicitly prohibited |
| Country-specific code paths | France unseen in train; generalize or lose |
| One-hot encoding of country | Open-set requirement; use string features |
| LLM as an oracle judge | Rule ambiguity re: "external lookup"; unnecessary given feature approach |
| Aggressive early transformer use | 72h budget; LightGBM reaches similar precision in minutes |
| Row-level train/val split | Data leakage — must use entity-level split |
| Force-matching singletons | Destroys precision; singletons are free 1.0s when left empty |
| Random pair negatives | Easy negatives teach nothing — hard negatives only |
| Threshold = 0.5 by default | F₀.₅ optimal threshold is 0.60–0.75, not 0.5 |

---

## Architecture Validation Against Problem Statement

| PS Requirement | Our Architecture | Status |
|---|---|---|
| Every S1 entity has exactly one row | Stage 6 iterates over all S1 IDs explicitly | ✅ |
| `matched_entity_ids` empty for singletons | Threshold default when no candidate passes | ✅ |
| No duplicate IDs in lists | Set-based deduplication before output | ✅ |
| All matched IDs must be in candidate set | Candidates written before filtering | ✅ |
| Only S2/S3 IDs in output | Source prefix checked at output stage | ✅ |
| Country is open-set (France in test) | String equality feature, no closed vocabulary | ✅ |
| F₀.₅ optimized (not F₁) | `compute_macro_f05()` exactly reproduces metric; threshold swept on this | ✅ |
| Singletons included in F₀.₅ average | `compute_macro_f05()` handles empty-vs-empty as 1.0 | ✅ |
| No external data lookup | Only provided TSV files used | ✅ |
| Model ≤ 8B params, MIT/Apache 2.0 | LightGBM (thousands of params, MIT) | ✅ |
| Tab-separated files | `sep="\t"` explicit everywhere | ✅ |
| Both TSV files in output/ | Stage 6 writes both | ✅ |
| Validate before submitting | `validate_submission.py` in run pipeline | ✅ |
| Methodology document | `Documentation_template.md` filled in final step | ✅ |

---

*This document is the single source of truth for architecture decisions in this project. Any deviation from this design must be logged in the experiment table with a measured F₀.₅ delta justification.*
