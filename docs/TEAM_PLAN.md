# Amazon ML Challenge 2026

## Business Entity Resolution — Team Execution Plan

**Team:** Mohith · Sanhitha · Harsha · Sahasra  
**Goal:** Produce a strong, reproducible first submission, then improve toward the leading leaderboard scores.  
**Primary metric:** Macro F₀.₅ across Source 1 entities, including singletons.

## 1. Submission and Ranking Requirements

The leaderboard scores `matching_results.tsv`. The organizer's candidate-generation update also states that `candidate_pairs.tsv` and its producing code are considered in the final ranking review, with smaller candidate sets favored. The update does not specify a numerical formula for combining those factors.

Optimize three things together:

1. Correct final matches under macro F₀.₅.
2. High candidate recall and oracle macro F₀.₅ ceiling.
3. Small candidate sets, bounded memory, and practical runtime.

Do not reduce candidates so aggressively that the matching score deteriorates. Compare candidate budgets on the same validation split and prefer the smallest configuration that preserves the measured score and retrieval ceiling.

Required final archive:

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

Every test S1 entity appears exactly once in both TSVs. Lists contain unique, existing test S2/S3 IDs. Empty predictions are valid. Matches are a subset of candidates; equality is allowed.

`candidate_pairs.tsv` must describe the final set actually passed into the matching model. Do not trim the export after model scoring to make the reported candidate count smaller. Persist the candidate configuration and verify export fidelity.

## 2. Ownership

| Member | Primary ownership | Deliverables | Primary reviewer |
|---|---|---|---|
| Harsha | Retrieval, training, and decision system | Blocking paths, indexes, reranking, candidate budgets, hard negatives, LightGBM training, model experiments, calibration, inference | Mohith |
| Mohith | Data, features, validation infrastructure, integration | Loaders, EDA, data checks, normalized-data handoff, feature registry and extraction, entity split, pipeline wiring, release package | Harsha |
| Sanhitha | Normalization and preprocessing | Unicode-safe text views, name cores, tokens, numeric evidence, postcode candidates, deterministic preprocessing, regression tests | Mohith |
| Sahasra | Evaluation verification, experiment records, submission QA | Independent metric tests, output checks, validation scorecards, error-analysis summaries, methodology sections | Mohith |

Each owner publishes a working interface with small fixtures. Development proceeds independently; integration starts as soon as a baseline is available.

Mohith coordinates shared configuration, dependencies, README, and release packaging. Harsha approves the retrieval/model configuration selected for a submission. Sahasra checks the exported outputs independently.

## 3. Runtime Architecture

```text
Raw TSV files
    → Schema and ID validation
    → Multi-view normalization
    → Bounded candidate retrieval
    → Union, reranking, and candidate-budget selection
    → Freeze the candidate set
    → Pair features
    → Trained model scores
    → Validated thresholds and decision rules
    → Export matching_results.tsv and candidate_pairs.tsv
    → Organizer validator
```

EDA and model selection are offline activities. All runtime assets must be generated from the supplied data or included in the package. No external business lookup is permitted.

## 4. Shared Contracts

### Raw sources

`entity_id`, `business_name`, `business_address`, and `country` are preserved as strings or explicit nulls. IDs must be unique and have the expected source prefix. Load with an explicit tab separator.

Ground truth contains `source1_entity_id` and `matched_entity_ids`. Parse labels into sets for evaluation, while validating duplicate IDs before deduplication.

### Normalized records

Keep the existing integration fields stable:

```text
entity_id
norm_name
name_core
norm_address
norm_country
digit_tokens
```

Add documented fields for name/address tokens, accent-folded alternatives, original missingness, numeric tokens with field provenance, and uncertain postcode candidates. Numeric tokens remain strings so leading zeros survive. Empty text and empty token collections never create exact-match keys.

Preserve Unicode combining marks in the primary view. Keep an address representation without ambiguous abbreviation expansion. Accent folding and abbreviation expansion are alternate views, not destructive replacements.

### Candidate table

Use one row per unique S1–target pair:

```text
source1_entity_id
candidate_entity_id
candidate_source
blocking_score
rank_within_source
blocking_paths
```

Keep the complete S1 ID list separately so entities without candidates receive empty output rows. Submission list-format TSVs are exports, not the internal training table.

### Feature registry

Mohith publishes `src/features/registry.py` with exact names, order, formulas, types, missing-value rules, and a version. Harsha consumes this registry for training and inference and validates feature-version compatibility.

The baseline contains the agreed 36 feature categories from `ARCHITECTURE.md`. Extensions require a registry version and an ablation result. Blocking metadata is passed explicitly to feature extraction; it is not silently embedded in business records.

## 5. Harsha — Retrieval, Model Training, and Decisions

**Own:** `src/blocking/`, `src/model/`, retrieval/model configurations, and training/inference experiments.

### Retrieval baseline

Implement these paths against S2 and S3 independently:

1. Rare name tokens.
2. Rare address tokens.
3. Rare numeric tokens with field provenance.
4. Exact normalized name.
5. Exact name core.
6. Name-core prefix.
7. Postcode candidate plus name token.
8. Name character n-grams for typos and missing spaces.

Use bounded postings and query depth. Compare path contribution and recall losses from each cap. Do not build a full S1 × target similarity matrix.

Start with top-K=20 per source, then compare 5, 10, 20, and 40. Use larger development budgets only to measure missed-match ranks and the cost of pruning. Adaptive depth is an experiment driven by retrieval ambiguity, not by validation labels.

Process large target pools in shards and reference queries in batches. Use consistent corpus statistics across shards, deterministic ordering, and a disk-backed merge. Sharded path limits can change retrieval behavior, so benchmark shard size as well as K.

### Model baseline

- Train LightGBM on retrieved positives and hard nonmatching candidates.
- Preserve singleton entities in training and evaluation.
- Record negative sampling and any correction weights.
- Use full validation candidate pools for threshold selection and evaluation.
- Select model settings using grouped validation; keep an untouched audit split where practical.
- Sweep the decision threshold broadly, then refine near useful score breakpoints.
- Accept zero or many target records; never force a match.

### Priority improvements

1. Distinctive-token agreement and contradictions.
2. Address/numeric disagreement with missingness handling.
3. Reverse competition between S1 references for the same target.
4. Hard-negative mining from held-out predictions.
5. A CatBoost challenger or small ensemble if errors are complementary.
6. An entity-level singleton gate only if independent validation improves.

Learned context models and calibrators require held-out training scores. No graph component is merged automatically from uncertain links. Pretrained or semantic stages require a clear measured need and eligible licenses.

### Deliverables

Runnable candidate generation, measured blocking reports, trained model artifacts, model/feature version checks, threshold configuration, batched prediction, and an experiment log with reproducible seeds.

## 6. Mohith — Data, Features, Splits, and Integration

**Own:** `src/data/`, `src/features/`, split infrastructure, pipeline orchestration, release documentation, and packaging.

- Obtain the organizer's dataset/resource archive and make the supplied data available through the documented local folder structure. Keep raw data out of Git.
- Verify file counts, schemas, missingness, IDs, ground-truth references, and coverage.
- Make EDA tables reproducible from measured artifacts rather than hard-coded notebook values.
- Preserve raw strings in generic and source-specific loaders.
- Publish the feature registry and implement pair features against synthetic fixtures.
- Build the entity-level split manifest with Harsha before label-informed tuning begins.
- Keep an S1 record and all labeled S2/S3 variants together; group overlapping labels together.
- Retain realistic target distractors in validation and document partitioning.
- Wire preprocessing, candidates, features, training, decisions, and exports into documented commands.
- Add the actual organizer-provided validator and verify the final archive layout.

**Done when:** contracts pass, feature extraction is deterministic, split IDs reproduce, the pipeline runs from a clean environment, and the final package regenerates both TSVs.

## 7. Sanhitha — Normalization and Preprocessing

**Own:** `src/preprocessing/` and normalization tests.

- Fix null handling for Python, pandas, and NumPy scalar values.
- Preserve combining marks and non-Latin scripts.
- Handle accented text through primary and alternate views.
- Preserve meaningful numbers, leading zeros, and raw fields.
- Expose ordered tokens and clearly typed token collections.
- Keep legal-suffix-free names as an additional view.
- Provide cautious postcode candidates with explicit uncertainty.
- Avoid destructive substitutions for ambiguous address tokens.
- Test the DataFrame entry point for schema, nonmutation, missingness, and determinism.

**Done when:** the agreed schema is available, Unicode/null regressions pass, original data is preserved, and unseen country labels work without special code paths.

## 8. Sahasra — Evaluation and Submission QA

**Own:** independent evaluation checks, experiment summaries, methodology contributions, and submission QA fixtures.

- Verify per-entity F₀.₅, empty/empty handling, false singleton merges, and multi-match behavior.
- Compare the official metric implementation against a small independent fixture calculation.
- Maintain scorecards by country, source, singleton status, and match count.
- Review sampled false positives and missed matches and record recurring categories.
- Check exact output headers, row completeness, valid target IDs, duplicates, and candidate-subset consistency.
- Verify that exported candidates are the actual scored set.
- Fill methodology sections from the final measured pipeline and experiment log.

**Done when:** independent checks agree, outputs pass the organizer validator, and the methodology accurately describes the submitted implementation.

## 9. Evaluation and Model Selection

For a nonsingleton entity:

```text
F0.5 = 1.25 × TP / (1.25 × TP + FP + 0.25 × FN)
```

True and predicted sets both empty score 1. A true singleton with any prediction scores 0. Average over every validation S1 entity.

Record for each run:

| Measurement | Purpose |
|---|---|
| Macro F₀.₅ | Main matching objective |
| Micro precision and recall | Link-level behavior |
| Singleton false-positive rate | Costly false merges |
| Nonsingleton empty-prediction rate | Entire businesses missed |
| Retrieval recall before/after top-K | Blocking losses |
| Oracle macro F₀.₅ | Maximum score available to the matcher |
| Mean/p95/max candidates per S1 | Ranking review and workload |
| Reduction ratio | Search-space reduction |
| Runtime and peak memory | Scalability |
| Country/source breakdowns | Generalization and uneven errors |

Aim for a retrieval ceiling comfortably above the desired matching score. A 95% link-recall milestone is not enough evidence for a near-0.99 submission. Compare actual ceilings and score curves rather than assuming a target recall guarantees a score.

Use grouped development predictions for tuning. Do not report the highest tuned score as an independent estimate. Run US→India and India→US transfer checks where practical; these cannot establish French accuracy.

Corpus-local retrieval frequencies may be used to construct the target search index without labels and must be documented. Supervised transforms, model fitting, score calibration, and label-informed tuning use training/development partitions only.

## 10. First Submission Schedule

The end-of-day goal is a real, validated submission supported by local results. Execution begins as soon as the official train/test files are available.

| Checkpoint | Harsha | Mohith | Sanhitha | Sahasra | Exit condition |
|---|---|---|---|---|---|
| Start | Fixture-tested retrieval and training interfaces | Data access, schemas, feature registry | Normalization fixes | Metric and output fixtures | Data and contracts usable |
| Baseline | Candidate generation and LightGBM run | Split and pair-feature generation | Preprocessing integration | Scorecard verification | Genuine grouped validation score |
| Improvement | Recall/candidate-budget tuning, hard negatives, thresholds | Feature fixes and pipeline integration | Systematic text-error fixes | Error categories and QA | Selected configuration improves held-out results |
| Release | Refit and full-test prediction | Package and clean reproduction | Preprocessing sign-off | Validator and methodology | Both TSVs valid; portal upload ready |

If time is limited, prioritize a complete tested baseline, retrieval misses, false merges, and threshold selection. Do not spend the submission window adding a complex stage without a measured benefit.

## 11. Git and Release Rules

- Keep `main` protected; implementation arrives through reviewed pull requests.
- Use the existing `Mohith`, `Sanhitha`, and `Harsha` branches. Evaluation work uses its own branch.
- Small commits; explicit contract changes; no simultaneous uncoordinated edits to shared files.
- Pin the tested dependencies and record random seeds, split IDs, configurations, and model versions.
- Document model licenses and comply with the parameter limit.
- Never upload synthetic fixture outputs as competition predictions.
- Run the official validator before every portal upload.
- The archive's matching file must be identical to the leaderboard upload.
- Preserve the selected candidate file and configuration alongside the matching result.

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The methodology template must cover preprocessing, blocking, features, model architecture, training, validation, decisions, scalability, licenses, measured results, and reproduction steps. It should describe only the pipeline actually submitted.
