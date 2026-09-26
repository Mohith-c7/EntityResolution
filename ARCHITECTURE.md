# Business Entity Resolution — Amazon ML Challenge 2026
## Team Execution Architecture & System Specification

> **Version:** 3.0
> **Status:** Team Execution Architecture / Locked
> **Target Metric:** Macro F₀.₅ (precision-weighted, per Source 1 entity, macro-averaged)
> **Team:** Mohit · Sanhita · Harsha · Sahasra

> **Implementation checkpoint (26 September 2026):** the 36-feature baseline and versioned 50-feature extension are runnable. The improved full-target pilot achieved 0.9511 local macro F₀.₅ on 2,000 held-out S1 entities. Complete official-test inference and release are pending. See the package README and `reports/experiments/PILOT_RESULTS.md` for the measured configuration; the original baseline diagram below is not a claim that every optional stage is selected.

---

## What Changed in v2.0

Version 2.0 reconciles the original design with the team-approved execution plan:
1. **Explicit Pipeline Separation**: Cleanly distinguishes between the runtime inference pipeline and the offline EDA/profiling/experimental components.
2. **Contract-First Engineering**: Defines formal data contracts (Raw Records, Normalized Records, Candidate Table) and component interfaces with early synthetic fixture integration.
3. **Eight-Path Baseline Blocking**: Expands candidate retrieval from 7 to 8 baseline paths by adding Path 8 (Name character n-gram TF-IDF retrieval) and explicit name/address provenance for numeric tokens.
4. **Authoritative Feature Registry**: Establishes that the 36-feature baseline specification is maintained in a versioned feature registry published by Mohith, rather than relying on an informal dictionary.
5. **Rigorous Grouped Validation**: Formally defines connected-component grouping for overlapping labels, distractor partitioning, US↔India transfer checks, and bootstrap resampling.
6. **Configurable vs Fixed Decisions**: Replaces rigid heuristics with clear boundaries: the architecture fixes contracts, interfaces, and evaluation protocols, while candidate depth (top-K), thresholds, and retrieval weights are empirically calibrated.
7. **Team Roles & 5-Phase Lifecycle**: Maps explicit deliverables, review pairings, feature branches, and phase exit criteria across all four team members.

---

## Table of Contents

1. [Problem Definition & Hard Rules](#1-problem-definition--hard-rules)
2. [Operating Principles](#2-operating-principles)
3. [Canonical Runtime Pipeline](#3-canonical-runtime-pipeline)
4. [Data & Raw Record Contract](#4-data--raw-record-contract)
5. [Normalized Record Contract](#5-normalized-record-contract)
6. [Internal Candidate Table Contract](#6-internal-candidate-table-contract)
7. [Component Interfaces](#7-component-interfaces)
8. [Candidate Blocking Architecture (8 Baseline Paths)](#8-candidate-blocking-architecture-8-baseline-paths)
9. [Blocking Evaluation Protocol](#9-blocking-evaluation-protocol)
10. [Pairwise Feature Architecture](#10-pairwise-feature-architecture)
11. [LightGBM Scoring & Negative Sampling](#11-lightgbm-scoring--negative-sampling)
12. [Shared Grouped Validation Protocol](#12-shared-grouped-validation-protocol)
13. [Decision Layer & Threshold Calibration](#13-decision-layer--threshold-calibration)
14. [Team Ownership & Responsibilities](#14-team-ownership--responsibilities)
15. [Parallel Work Phases (A–E)](#15-parallel-work-phases-a-e)
16. [Collaboration & Experimentation Rules](#16-collaboration--experimentation-rules)
17. [Final Integration & Release Architecture](#17-final-integration--release-architecture)
18. [Non-Negotiable Constraints](#18-non-negotiable-constraints)
19. [Configurable Choices vs Fixed Requirements](#19-configurable-choices-vs-fixed-requirements)

---

## 1. Problem Definition & Hard Rules

### Deliverable Output Files

| Output File | Description | Scored? |
| :--- | :--- | :---: |
| `output/matching_results.tsv` | Exactly one row per test S1 entity. `matched_entity_ids` = comma-separated S2/S3 IDs that match. Empty string if singleton. | ✅ **Yes** — Primary Leaderboard Driver |
| `output/candidate_pairs.tsv` | Exactly one row per test S1 entity. All S2/S3 IDs considered by the model prior to thresholding. | **Not leaderboard-scored; reviewed for final ranking, including candidate size** |

### Non-Negotiable Challenge Rules
- **Reference Entity Constraint**: Source 1 is the reference; each S1 entity may match zero, one, or many Source 2/3 records.
- **Row Completeness**: Every test Source 1 entity must appear exactly once in both output files.
- **Source Separation**: `matched_entity_ids` and `candidate_entity_ids` contain only S2 and S3 IDs — never S1 IDs.
- **Deduplication**: No duplicate IDs within any row's comma-separated list.
- **Subset Rule**: All IDs in `matching_results.tsv` must be a subset of the corresponding row in `candidate_pairs.tsv`.
- **Open-Set Country**: France appears in ~14.5%–15.0% of the test set but 0% in training. The system must never hard-code country lists, one-hot encode country, or exclude unseen countries.
- **Data Hermeticity**: No external geocoding, business registry lookups, web scraping, or LLMs as oracles.
- **Model Constraints**: Final model must be $\le 8\text{B}$ parameters and licensed under MIT or Apache 2.0.
- **Format Integrity**: Tab-separated files only (`sep="\t"`). Never read or write `.tsv` files with commas or default delimiters.

### Evaluation Metric: Macro F₀.₅

$$\text{Macro } F_{0.5} = \frac{1}{|S_1|} \sum_{s \in S_1} F_{0.5}(s)$$

Where for each entity $s$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- The score penalizes false positives more heavily: in the count formula, an FP contributes 1 and an FN contributes 0.25 to the denominator.
- Singletons score **1.0** when predicted empty, and **0.0** when any false match is predicted.
- **Strategic rule: Default to NOT matching when uncertain.**

---

## 2. Operating Principles

1. **Contract-First Development**: Components are developed independently against shared, documented contracts.
2. **Early Integration**: Integrate early using a small end-to-end synthetic fixture bundle before connecting full pipelines.
3. **Shared Contracts Before Code**: Shared contracts and schemas are committed before implementation integration.
4. **Configurable Choices**: Retrieval depth, feature additions, model hyperparameters, and thresholds remain configurable rather than fixed prematurely.
5. **Empirical Evidence Over Intuition**: Every change requires measured validation evidence (Macro $F_{0.5}$, link recall) rather than speculative redesign.
6. **No Postponed Integration**: Do not wait for all experiments to finish before integrating. A working baseline must exist before advanced features are added.
7. **Blocking Recall is the Ceiling**: A true match not retrieved during blocking can never be recovered by the model.

---

## 3. Canonical Runtime Pipeline

```
Raw TSV Files (Source 1, Source 2, Source 3)
     │
     ▼
[Load & Validate Schema]
     │
     ▼
[Normalize & Preprocess] ──→ Retains raw fields, generates multi-view clean representations
     │
     ▼
[Candidate Blocking]     ──→ 8 baseline retrieval paths run independently on S2 and S3
     │
     ▼
[Union & Deduplicate]    ──→ Retains retrieval path provenance
     │
     ▼
[Rerank & Top-K Cap]     ──→ Symmetric scoring + exactness bonuses; per-source selection
     │
     ▼
[Export candidate_pairs.tsv] ──→ Freezes final candidate pool for scoring and audit
     │
     ▼
[Pairwise Feature Construction] ──→ Computes 36-feature vector from frozen candidate pairs
     │
     ▼
[LightGBM Scoring]       ──→ Evaluates pair match probabilities
     │
     ▼
[Threshold Calibration]  ──→ Empirically swept across 0.30–0.95 optimizing Macro F0.5
     │
     ▼
[Export matching_results.tsv] ──→ Emits final match lists (subsets of candidates)
     │
     ▼
[Submission Validation]  ──→ utils/validate_submission.py must report PASS
```

> **Note**: Stage 0 EDA, profiling, and error analysis inform design decisions and experiments; they are development activities, not mandatory runtime components.

---

## 4. Data & Raw Record Contract

Source loaders return raw DataFrames adhering to the following schema without mutating business text:

### Source Record Schema (`train_source*.tsv`, `test_source*.tsv`)

| Column | Type | Requirement |
| :--- | :--- | :--- |
| `entity_id` | `String` | Non-null, unique within source, correct prefix (`S1-`, `S2-`, `S3-`). |
| `business_name` | `Nullable string` | Original text preserved; never coerced to numeric. |
| `business_address`| `Nullable string` | Original text preserved; permitted to be null (~3.3% missing in S2/S3). |
| `country` | `Nullable string` | Open-set string label; no country whitelist. |

> **Country Contract & Challenge Dataset Invariant**:
> `country` is an open-set string label. The schema remains null-safe, but the supplied challenge datasets currently contain no missing country values. Data validation must preserve and report the actual dataset invariant rather than introducing a country whitelist. Challenge datasets must not be restricted to a closed vocabulary (e.g. US, India, France); any valid string label must be supported without filtering or rejection.

### Ground Truth Record Schema (`train_ground_truth.tsv`)

| Column | Type | Requirement |
| :--- | :--- | :--- |
| `source1_entity_id` | `String` | Non-null, unique, 100% bijective correspondence with `train_source1.tsv`. |
| `matched_entity_ids`| `Nullable string` | Comma-separated list of true S2/S3 IDs. Empty string for singletons (5.58%). |

The ground truth loader parses match lists into sets for internal evaluation while preserving the raw table format.

---

## 5. Normalized Record Contract

The preprocessing module (`src/preprocessing/`) transforms raw records into enriched records. All original columns are preserved intact, accompanied by deterministic derived views:

| Field | Type | Semantics & Meaning |
| :--- | :--- | :--- |
| `name_norm` | `String` | Unicode NFKC, casefolded, punctuation stripped, whitespace collapsed, abbreviations expanded. |
| `name_folded` | `String` | Accent-folded alternate view (removes diacritics while preserving Indic combining characters). |
| `name_core` | `String` | Legal-suffix-stripped representation (removes `pvt`, `ltd`, `llc`, `corp`, etc.). |
| `name_tokens` | `Tuple[str, ...]` | Ordered sequence of normalized name tokens. |
| `address_norm` | `String` | Normalized address text (abbreviations expanded, lowercase, cleaned). |
| `address_folded`| `String` | Accent-folded address view. |
| `address_tokens`| `Tuple[str, ...]` | Ordered sequence of normalized address tokens. |
| `name_digits` | `Tuple[str, ...]` | Ordered numeric tokens extracted from `business_name` (strings preserving leading zeros). |
| `address_digits`| `Tuple[str, ...]` | Ordered numeric tokens extracted from `business_address` (strings preserving leading zeros).|
| `postcode_candidates` | `Tuple[str, ...]` | Conservative format-based postal code candidates; empty tuple if none. |
| `country_norm` | `String` | Normalized open-set country string label (whitespace cleaned, casefolded). |
| `name_missing` | `Boolean` | Flag indicating if original `business_name` was null/empty. |
| `address_missing`| `Boolean` | Flag indicating if original `business_address` was null/empty. |
| `country_missing`| `Boolean` | Flag indicating if original `country` was null/empty. |

### Contract Guarantees:
1. Missing text becomes `""` (empty string), missing token collections become `()` (empty tuple).
2. Numeric tokens remain strings to preserve leading zeros (`"0123"`, `"07"`).
3. Ordered tokens retain original sequence; retrieval components derive sets when needed.
4. Postcode candidates represent uncertain format evidence only — never verified geographical attributes. External postal database lookups are strictly prohibited.

---

## 6. Internal Candidate Table Contract

The candidate blocking module emits an internal table linking reference S1 entities to retrieved candidates:

| Column | Type | Invariant / Requirement |
| :--- | :--- | :--- |
| `source1_entity_id` | `String` | Reference S1 entity ID. |
| `candidate_entity_id`| `String` | Target candidate entity ID (prefixed `S2-` or `S3-`). |
| `candidate_source` | `String` | Literal `"S2"` or `"S3"`. |
| `blocking_score` | `Float` | Fused similarity score from blocking reranker. |
| `rank_within_source` | `Integer` | Candidate rank per S1 entity within source ($1, 2, \dots, K$). |
| `blocking_paths` | `Tuple[str, ...]` | Sorted collection of path names that retrieved this candidate. |

### Invariants & Freezing Contract:
1. **Exact Candidate Pool Representation**: `candidate_pairs.tsv` MUST represent the exact frozen candidate dataframe passed to feature construction and matcher scoring. Once scoring begins, candidates cannot be added or removed, and the file must not be independently regenerated from another source.
2. **Scoring Invariant**: $\text{final\_matches}(S_1) \subseteq \text{candidate\_pairs}(S_1)$, and `candidate_pairs.tsv` equals the actual scored candidate set.
3. **Uniqueness**: Exactly one row per unique $(S_1, \text{Target})$ candidate pair.
4. **Universe Completeness**: Reference S1 entities with zero candidates remain represented in the master S1 universe so they receive empty outputs.

---

## 7. Component Interfaces

All components adhere to explicit, functional interfaces:

```python
# Ingestion
load_source(path: Path | str) -> pd.DataFrame
load_ground_truth(path: Path | str) -> pd.DataFrame

# Preprocessing & Normalization
normalize_name(value: Any) -> str
normalize_address(value: Any) -> str
normalize_country(value: Any) -> str
extract_digits(value: Any) -> tuple[str, ...]
name_core(value: Any) -> str
preprocess_source(records: pd.DataFrame) -> pd.DataFrame

# Candidate Blocking
generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: dict[str, Any]
) -> pd.DataFrame

# Feature Engineering
build_pair_features(
    s1_record: dict[str, Any],
    candidate_record: dict[str, Any],
    candidate_metadata: dict[str, Any]
) -> dict[str, Any]

# Model Training & Inference
train(
    features: pd.DataFrame,
    labels: pd.Series,
    config: dict[str, Any],
    sample_weight: pd.Series | None = None
) -> Any

predict(model: Any, features: pd.DataFrame) -> np.ndarray

# Decision & Export
select_matches(
    candidate_pairs: pd.DataFrame,
    probabilities: np.ndarray,
    all_source1_ids: list[str],
    config: dict[str, Any]
) -> dict[str, list[str]]
```

> **State Isolation Rule**: Any learned retrieval state (TF-IDF vectorizers, document frequencies) must expose an explicit `fit()` step restricted strictly to the permitted training split.

---

## 8. Candidate Blocking Architecture (8 Baseline Paths)

The blocking engine executes eight complementary retrieval paths independently against Source 2 and Source 3:

| Path | Name | Target Mechanism | Why It Is Essential |
| :---: | :--- | :--- | :--- |
| **1** | Rare Name Tokens | Inverted index on `name_tokens` with configurable DF cutoff (initial baseline $\le 10\%$). | Catches word-order changes and token overlap. |
| **2** | Rare Address Tokens | Inverted index on `address_tokens` with configurable DF cutoff (initial baseline $\le 10\%$). | Catches address matches when names differ significantly. |
| **3** | Rare Numeric Tokens | Inverted index on `name_digits` and `address_digits`. | Preserves provenance; matches house numbers and tax IDs. |
| **4** | Exact Normalized Name | Exact lookup on `name_norm`. | High-precision baseline for unmodified names. |
| **5** | Exact Name Core | Exact lookup on `name_core`. | Matches entities differing only in legal forms (`Pvt Ltd`). |
| **6** | Name Core Prefix | Prefix lookup on first 8 characters of `name_core`. | Bridges stem variations and minor suffix modifications. |
| **7** | Postcode + Name Token | Joint key: `postcode_candidate` + highest-IDF name token. | Anchor blocking in dense metropolitan areas. |
| **8** | Name Character n-gram | TF-IDF retrieval on character 3-grams/4-grams of `name_norm`. | Bridges typographical errors, OCR noise, and leetspeak. |

### Candidate Selection Rules:
1. **Independent Retrieval**: Query Source 2 and Source 3 independently.
2. **Union & Deduplication**: Combine candidate pools while tracking path provenance.
3. **Fused Symmetric Reranker**:
   $$\text{Score} = \sum_{\text{paths}} w_p \cdot s_p + \text{Bonuses}$$
   Exact name bonus: $+2.0$, Exact core bonus: $+1.5$.
4. **Deterministic Tie-Breaking**: Broken deterministically by `(score DESC, candidate_entity_id ASC)`.
5. **Configurable Top-K**: Default $K=20$ per source (yielding up to 40 candidates per S1). Evaluate $K \in \{5, 10, 20, 40\}$ on development folds.
6. **Configurable Rarity Cutoffs**: Rarity cutoffs are configurable experimental parameters. The initial baseline uses the currently agreed cutoff; alternative cutoffs may be evaluated empirically.
7. **Open-Set Country Gate**: Country agreement is configurable evidence; records with missing or unseen country labels must never be discarded.
8. **No Truth Rescue**: No validation-label candidate rescue during validation or inference.

---

## 9. Blocking Evaluation Protocol

Every blocking configuration must be evaluated against the shared validation split across 14 standardized metrics:

1. **Link recall before reranking** (raw union recall).
2. **Link recall after final top-K** (effective ceiling for model).
3. **Fraction of non-singletons with $\ge 1$ true candidate**.
4. **Fraction of non-singletons with all true matches retrieved**.
5. **Oracle Macro $F_{0.5}$** (macro score if an oracle perfectly selected true matches from candidates).
6. **Recall broken down by source** (S2 vs S3).
7. **Recall broken down by country** (US vs India vs unseen).
8. **Recall broken down by address missingness**.
9. **Recall broken down by match-count bucket** (1, 2, 3–4, 5+).
10. **Incremental recall contribution of each of the 8 retrieval paths**.
11. **Candidate count statistics**: Mean, Median, P95, Maximum.
12. **Peak memory usage during indexing and query execution**.
13. **Runtime throughput** (records processed per second).
14. **Error analysis**: Reviewed qualitative audit of false negatives (misses).

> **Milestone vs Final Target**: $\ge 95\%$ link recall is an initial milestone. Final release configuration balances recall against downstream feature extraction cost and Macro $F_{0.5}$.

---

## 10. Pairwise Feature Architecture

The first model baseline operates on the agreed **36-feature schema**.

### Feature Registry Authority & Contract:
- **Role of ARCHITECTURE.md**: ARCHITECTURE.md defines the required feature categories, structural contracts, and broad functional roles.
- **Authoritative Feature Registry**: The versioned feature registry maintained by Mohith (`code/business_entity_resolution/src/features/registry.py`) is authoritative for exact feature names, ordering, formulas, types, missing-value imputation behavior, and feature version.
- **Prerequisite for Dependent Work**: The number "36" alone is NOT the specification; the versioned feature registry must be formally published before model training and integration work depend on it.

### Feature Categories:
1. **Name Similarity Features (8)**: Jaro-Winkler, Levenshtein ratio, token sort ratio, token set ratio, partial ratio, token Jaccard, length ratio, character n-gram cosine similarity.
2. **Address Similarity Features (8)**: Jaro-Winkler, Levenshtein ratio, token sort ratio, token set ratio, partial ratio, token Jaccard, length ratio, address numeric overlap.
3. **Numeric & Identifier Evidence (4)**: Digit token Jaccard, exact numeric match flag, postcode candidate agreement, street number match flag.
4. **Agreement & Contradiction Flags (6)**: Exact name match, exact name core match, any token exact match, name token in address, address token in name, country exact match.
5. **Missingness Indicators (3)**: S2/S3 address missing flag, postcode missing flag, country missing flag.
6. **Length Covariates (4)**: S1 name length, candidate name length, S1 address length, candidate address length.
7. **Blocking Metadata (1)**: Fused blocking score.
8. **Source Indicators (2)**: `source_is_s2`, `source_is_s3`.

### Priority Feature Experiments (Post-Baseline):
- Distinctive-token agreements and rare token contradictions.
- Numeric conflict features with ambiguity handling.
- Local name/address corpus frequency.
- Reverse-reference competition (multiple S1 entities competing for the same target).

---

## 11. LightGBM Scoring & Negative Sampling

- **Primary Scorer**: LightGBM binary classifier (`objective: binary`, `metric: binary_logloss`).
- **Positives**: Ground-truth pairs present in the candidate set.
- **Hard Negative Mining**: Non-matching candidates retrieved by blocking for that entity. Easy random pairs are strictly avoided.
- **Ratio & Weighting**: Default $1:2$ or $1:3$ positive-to-negative ratio. Sampling parameters and sample weights are recorded in the experiment log.
- **Validation Fidelity**: Validation evaluation must score the full candidate pool generated by blocking (never a subsampled pool) to mirror inference.

---

## 12. Shared Grouped Validation Protocol

All teammates evaluate against a single, shared, entity-disjoint validation split created before label-informed tuning:

1. **Entity-Level Grouping & Connected Components**:
   - An S1 reference entity and all its labeled S2 and S3 counterparts must always remain in the same fold. Row-level random splitting is strictly prohibited.
   - If match relationships create multi-reference or overlapping clusters, entire connected components remain together in the same fold.
   - Any unusually large or unexpectedly overlapping connected components must be audited and investigated during split generation.
   - Unmatched target records (distractors) are partitioned consistently across folds to maintain realistic candidate density and retrieval competition.
2. **Stratification**: Stratified by country, singleton status, and match-count bucket where feasible (`seed=42`, default 15% holdout).
3. **Strict Inference Simulation**: Validation retrieval indexes contain held-out target records; validation labels never influence candidate generation.
4. **Fitted State Isolation**: Supervised transforms, learned retrieval, and model calibration use training/development partitions only. Label-free frequencies used to construct each target search index are corpus-local and documented; validation labels never influence retrieval.
5. **Geographic Generalization Checks (Transfer Diagnostics)**:
   - Where feasible, cross-partition transfer checks must be run:
     - **US $\to$ India transfer check**: Train on US entities, evaluate on India holdout.
     - **India $\to$ US transfer check**: Train on India entities, evaluate on US holdout.
   - **Diagnostic Role**: These checks evaluate the pipeline's sensitivity to country distribution shift. They do NOT establish French test accuracy (since France is entirely unseen in training).
   - **Non-Target Principle**: These checks are validation diagnostics to detect brittle country-specific features, not new optimization targets.
6. **Bootstrap Significance Testing**:
   - For close final configurations, compare score differences using bootstrap resampling of Source 1 entities.

### Validation Scorecard Metrics:
- Macro $F_{0.5}$ (Primary selection metric)
- Micro precision and recall
- Singleton false-positive rate
- Non-singleton empty-prediction rate
- Blocking recall and oracle Macro $F_{0.5}$ ceiling
- Breakdown by country and candidate source
- Throughput, RAM, and candidate volumes

---

## 13. Decision Layer & Threshold Calibration

1. **Empirical Sweep**: Decision threshold $\tau$ is swept from $0.30 \to 0.95$ in steps of $0.01$ on validation predictions.
2. **Metric Optimization**: Select $\tau^*$ that strictly maximizes official entity Macro $F_{0.5}$.
3. **Range Clarification**: The initial sweep is refined and extended where validation warrants it. No expected threshold interval is treated as an optimum.
4. **Per-Source Calibration (Optional)**: If score distributions diverge between S2 and S3, calibrate independent thresholds $(\tau_{S2}, \tau_{S3})$.
5. **Cardinality Rules**:
   - Singletons: If no candidate exceeds $\tau^*$, output empty string (scores 1.0).
   - Multi-match: All candidates exceeding $\tau^*$ are emitted.
   - Predictions must never be forced.
   - Predictions are strictly constrained to the frozen candidate pool.

---

## 14. Team Ownership & Responsibilities

The current execution and submission plan is [docs/TEAM_PLAN.md](docs/TEAM_PLAN.md).

| Team Member | Primary Domain | Deliverables & Responsibilities | Primary Reviewer |
| :--- | :--- | :--- | :--- |
| **Mohith** | Data, Features, Validation Infrastructure, Integration | Loaders, EDA, integrity checks, feature registry and extraction, grouped split manifests, pipeline orchestration, release packaging. | Harsha |
| **Sanhitha** | Normalization & Preprocessing | Unicode-safe text views, legal suffixes, tokens, numeric evidence, postcode candidates, regression tests. | Mohith |
| **Harsha** | Retrieval, Model Training & Decisions | Eight retrieval paths, reranking, candidate budgets, hard negatives, LightGBM training, model experiments, calibration, final inference. | Mohith |
| **Sahasra** | Evaluation Verification & Submission QA | Independent metric tests, scorecards, output checks, error summaries, methodology contributions. | Mohith |

Mohith coordinates shared configuration, dependencies, integration, and packaging. Harsha selects the retrieval/model configuration using measured validation. Each owner maintains their component's tests and documentation.

---

## 15. Parallel Work Phases (A–E)

```
Phase A: Contracts & Fixtures     ──→ Shared contracts committed; synthetic fixtures available
     │
     ▼
Phase B: Independent Baselines    ──→ Each component passes unit tests on fixtures
     │
     ▼
Phase C: Early Integration        ──→ End-to-end pipeline smoke test passes on synthetic data
     │
     ▼
Phase D: Measured Improvement     ──→ Iterative ablation experiments against shared validation split
     │
     ▼
Phase E: Release Packaging        ──→ Final clean rerun; validator PASS; submission archive produced
```

- **Phase A (Contracts)**: Shared interfaces and synthetic test fixtures committed. *Exit: Contracts committed.*
- **Phase B (Baselines)**: Modules implemented independently against fixtures. *Exit: Component unit tests pass.*
- **Phase C (Early Integration)**: Minimal end-to-end pipeline wired and executed on synthetic data. *Exit: Pipeline smoke test passes.*
- **Phase D (Improvement)**: Controlled experiments logged with ablation reports. *Exit: Measurable Macro $F_{0.5}$ gain.*
- **Phase E (Release)**: Full dataset execution, validator check, package bundling. *Exit: `validate_submission.py` PASS.*

---

## 16. Collaboration & Experimentation Rules

### Branching Model
- `main`: Protected branch; direct pushes prohibited. All changes arrive via reviewed Pull Requests.
- `Mohith`: data, features, splits, and integration.
- `Sanhitha`: normalization and preprocessing.
- `Harsha`: blocking, training, and decisions.
- Evaluation and QA use a separate development branch.

### Experimentation Hygiene
- **One Change at a Time**: Change one variable per experiment (retrieval path, feature set, sampling ratio).
- **Mandatory Logging**: Every experiment logs: Hypothesis, Configuration, Split version, Seed (42), Validation Macro $F_{0.5}$, Precision, Recall, Runtime, Decision (Keep/Reject).
- **No Leaderboard Overfitting**: Decisions are guided by the shared local validation holdout, not public leaderboard noise.

---

## 17. Final Integration & Release Architecture

### Output File Specifications
1. `matching_results.tsv`:
   ```text
   source1_entity_id<TAB>matched_entity_ids
   S1-1001<TAB>S2-2001,S3-3001
   S1-1002<TAB>
   ```
2. `candidate_pairs.tsv`:
   ```text
   source1_entity_id<TAB>candidate_entity_ids
   S1-1001<TAB>S2-2001,S2-2005,S3-3001
   S1-1002<TAB>S2-2040
   ```
*(Actual tab delimiter; comma-separated IDs without spaces; empty strings for singletons).*

### Candidate Export & Scoring Invariant
- `candidate_pairs.tsv` MUST represent the exact frozen candidate dataframe passed to feature construction and matcher scoring.
- Once scoring begins, candidates cannot be added or removed, and the file must not be independently regenerated from another source.
- **Invariant**: $\text{final\_matches}(S_1) \subseteq \text{candidate\_pairs}(S_1)$, and `candidate_pairs.tsv` equals the actual scored candidate set.

### Release Checklist
- [ ] Exactly one row per test S1 ID in both TSV files.
- [ ] No duplicate IDs in lists; no duplicate S1 rows.
- [ ] Every target ID exists in test Source 2 or Source 3.
- [ ] All matched IDs are subsets of candidate IDs; equality is allowed.
- [ ] Candidate export matches the exact inputs scored by LightGBM.
- [ ] All countries represented without filtering.
- [ ] Full pipeline runs end-to-end via one documented command.
- [ ] `python3 utils/validate_submission.py` passes with exit code 0.

### Mandatory Submission Validator Command
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

### Final Submission Archive Structure
```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

## 18. Non-Negotiable Constraints

1. **No External Lookups**: No external geocoding, business registries, phone directories, or web APIs.
2. **No LLM Oracle**: No LLM used as an identity judge.
3. **No ID or Order Leakage**: No row index, file position, or ID digit values as predictive features.
4. **No Row-Level Random Split**: Never split pairs from the same reference entity across train and validation.
5. **No Validation Rescue**: No adding ground-truth matches into candidate sets during validation.
6. **No Country Whitelist**: No filtering or discarding records from unseen countries.
7. **No Forced Matching**: Singletons must remain empty if no candidate exceeds the calibrated threshold.
8. **No 1-to-1 Matching Assumption**: S1 entities can match 0, 1, or multiple S2/S3 entities.
9. **No Uncalibrated Connected Components**: No transitive closure merging from uncertain pairwise predictions.
10. **No Fixed Top-K Assumption**: Candidate depth must be empirically justified against recall ceilings.
11. **No Untested Semantic Models**: Heavyweight transformer models are deferred until baseline LightGBM is validated.

---

## 19. Configurable Choices vs Fixed Requirements

The architecture locks interfaces, data contracts, evaluation protocol, output invariants, and challenge constraints; it does not claim experimentally optimal hyperparameters prematurely. Hyperparameters and experimental choices must be determined and tuned empirically on the shared validation split.

### Explicitly Configurable Parameters:
The following remain configurable experimental parameters rather than hard architectural constraints:
1. **Rarity Cutoffs**: Rarity cutoffs are configurable experimental parameters. The initial baseline uses the currently agreed cutoff; alternative cutoffs may be evaluated empirically.
2. **Candidate Depth ($K$)**: Number of candidates retrieved per source ($K \in \{5, 10, 20, 40\}$).
3. **Retrieval Weights ($w_p$)**: Path-specific weights used in candidate fusion.
4. **Reranking Bonuses**: Additive exact-match or core-match bonuses.
5. **Feature Additions**: Post-baseline interaction, frequency, or conflict features registered in the feature registry.
6. **Negative Sampling Configuration**: Sampling ratio ($1:2$, $1:3$, $1:5$), hard-negative selection strategy, and sample weights.
7. **LightGBM Parameters**: Tree depth, number of leaves, learning rate, feature fraction, min child samples.
8. **Decision Thresholds**: Calibrated decision threshold $\tau^*$ (swept 0.30–0.95), and whether per-source thresholds $(\tau_{S2}, \tau_{S3})$ are applied.

### Architectural Matrix:

| Dimension | Fixed Architectural Requirement | Configurable Choice |
| :--- | :--- | :--- |
| **Data Contracts** | Exact column names, string types, source prefixes (`S1-`, `S2-`, `S3-`). | Internal batching/chunk sizes for processing. |
| **Country Handling** | Open-set string comparison; no closed whitelist. | Country agreement feature weighting and interaction rules. |
| **Candidate Retrieval**| 8 baseline paths; union & deduplication; provenance tracking. | Rarity DF cutoffs; per-source candidate depth $K$; retrieval weights; reranking bonuses. |
| **Pair Features** | Required feature categories; feature registry contract. | Exact registered feature set (baseline 36 + post-baseline additions). |
| **Scoring Model** | LightGBM binary classifier baseline. | Hyperparameters (num_leaves, learning_rate, depth, subsample). |
| **Negative Sampling** | Hard negatives mined from blocked non-matches. | Negative ratio ($1:2$ vs $1:3$ vs $1:5$); weighting schemes. |
| **Decision Layer** | Empirical sweep across 0.30–0.95 optimizing Macro $F_{0.5}$. | Calibrated threshold value $\tau^*$; per-source thresholds $(\tau_{S2}, \tau_{S3})$. |
| **Output Invariants** | Exact TSV tab-separation; candidate subset constraint; singletons empty. | Output buffering and chunk export sizes. |
