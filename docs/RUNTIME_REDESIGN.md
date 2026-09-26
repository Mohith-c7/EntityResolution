# Retrieval redesign and score improvement

## Current run

`output/submission_03/` contains the completed, frozen first submission. Both validators passed with ID checking enabled. The team reports a portal score of **0.9399**. It uses the saved v3 LightGBM matcher, bounded-postings retrieval, and threshold 0.75. Submission 01 and 02 remain stopped with their checkpoints preserved.

The complete test set contains 1,732,544 Source 1 references. Every reference, including France and empty predictions, must appear once in each output file. The matching file is published to `output/matching_results.tsv` only after strict and official validation pass with ID checking enabled.

The inherited 0.960616 holdout score belongs to an earlier retrieval pipeline. The submitted pipeline's **0.9594108 tuning score** and reported **0.9399 portal score** measure different datasets. The subsequent 300k matcher passed a new audit at **0.965888** and has not been submitted. New submission generation is paused by request; current model work is documented in [the round-two plan](ROUND_2_MODEL_PLAN.md).

## Runtime evidence

All isolated timings below used the existing Mac, full target indexes, and 10,000 query references.

| Configuration | References/second | Tuning macro F0.5 | Notes |
|---|---:|---:|---|
| Earlier full pipeline | Approximately 20 | 0.9602384 | Historical timing; same tuning set for score comparison |
| Narrow postings | 354.4 | 0.9567508 | Rejected quality tradeoff |
| Wider postings, 12 candidates/source | 272.6 | 0.9587388 | Lower candidate recall |
| Wider postings, 20 candidates/source, compact accumulator | 257.0 | 0.9592621 | Score at the old 0.76 threshold |
| Same wider engine on official test sample | 252.0 | Not available | Includes 1,481 French references; includes feature-cache writes |

At 252 references/second, inference projects to approximately 115 minutes. Final assembly and validators require additional time. This is a projection, not a completed full run. The first sustained full-run checkpoints reached approximately 265 references/second; live progress is the authoritative estimate.

## What changed

1. Retrieve complete, bounded token postings without sorting broad matches by BM25 for each reference.
2. Cache postings with a fixed byte budget. Oversized token blocks are skipped rather than taking arbitrary first rows.
3. Accumulate name, address, numeric and training-derived name-alias evidence using compact arrays and a small native hash accumulator.
4. Union bounded field-specific shortlists and rerank them. The current frozen configuration retains 20 candidates per source.
5. Compute the existing 65 matching features and score every retained candidate.
6. Save candidate IDs, provenance, ranks, all model features and probabilities in checkpointed Parquet files. Future compatible scoring experiments can reuse these features.
7. Export exactly those scored candidates. Final matches are a subset. Both validators run after complete assembly.

The native extension is project code under the MIT license. It transfers integer postings and accumulates weights; it does not contain identity data or call external services. It has a pure NumPy fallback. Build it with `scripts/build_postings_native.py` using a C compiler and loadable-extension SQLite headers.

Compatibility checks found **227,683 overlapping pairs with all 64 non-blocking features identical** to the previous pipeline. Blocking score differs because retrieval provenance changed. Compact and reference implementations produced identical candidate lists and predictions on the complete tuning comparison. The test suite passed 116 tests before launch, including finalization and checkpoint resumption.

## Score target

The target remains 0.99 macro F0.5. It is not guaranteed by the current model or by the redesign.

Current tuning error counts:

- 1,212 true links missing from the retained candidates.
- 1,705 retrieved true links rejected by the matcher.
- 427 false links accepted, including errors on 33 singleton references.

An ideal matcher restricted to the current candidates reaches only **0.9885139**. Reaching 0.99 therefore requires improving candidate retention as well as matching.

Ordered attribution of the total macro loss:

| Error source | Macro F0.5 lost |
|---|---:|
| True links absent from candidates | 0.0114861 |
| Retrieved true links rejected | 0.0178557 |
| False links on references with true matches | 0.0079474 |
| False links on singletons | 0.0033000 |

These counterfactual contributions are order-dependent, not independent achievable gains.

## Active experiments

### 1. Learned candidate ranking

Train a small LightGBM ranking classifier using 15 inexpensive pair features and the existing training-fold hard negatives. Use its scores to choose the final candidates from a broader cheap shortlist. Preserve the main matcher's feature definitions and keep the same final candidate budget for the first comparison.

The initial ranker was fitted on 20,000 training entities / 582,358 pairs. Tuning data is used for early stopping only; no audit or test labels fit the model. This first ranker has not yet learned from all raw retrieval negatives, which is a known limitation.

Measure retained-link recall, candidate oracle macro F0.5, actual matcher macro F0.5, precision, singleton errors, country slices and candidate counts. Promote only after a larger tuning comparison and a fresh entity-level audit.

### 2. Matcher evidence and training coverage

Use cached error pairs to prioritize changes. Of the rejected true links, 431 have missing candidate addresses, 355 have strong address similarity but weak name similarity, and 309 have a numeric contradiction. These slices overlap.

Test improvements on training data, including stronger name evidence for missing addresses, handling number edits without weakening distinct-business checks, and more training entities with retrieval-generated hard negatives. Use a controlled model that reduces dependence on blocking score to test whether the matcher is over-relying on the earlier ranking heuristic.

### 3. Country generalization and ownership

France has no labels. Its probability and match-count distributions are diagnostics, not measured accuracy. Do not force its match-count distribution to resemble the US or India.

A target record has one owner in the provided training labels. Before adding global ownership decisions, evaluate on a complete labelled slice containing competing Source 1 records; per-reference samples are insufficient.

## Experiment discipline

- Keep the background submission's code, model, retrieval configuration and threshold frozen.
- Run new experiments in separate model directories. Limit CPU usage while inference runs.
- Select changes using tuning data. Never retune on the already inspected v3 audit set.
- Use a fresh entity-level audit for the chosen next pipeline and report paired uncertainty.
- Distinguish tuning, holdout and leaderboard scores in every report.
- Use only the provided business data. No registries, geocoding or external identity lookups.

## Files

- `reports/experiments/redesign/current_error_budget.json`: current error attribution.
- `reports/experiments/redesign/postings_test_10k.json`: test throughput and country counts.
- `reports/experiments/redesign/wide_threshold_selection.json`: tuning-only threshold sweep.
- `reports/experiments/redesign/feature_compatibility.json`: feature equivalence checks.
- `output/submission_03/progress.json`: live generation status.
- `output/submission_03/generation.log`: runner log.
- `output/submission_03/feature_cache/`: scored candidate evidence.

## Follow-up experiments and review findings

The first learned-ranker probe increased candidate oracle F0.5 to 0.992341 on 2,000 tuning references (the original pipeline: 0.990537 on those references). Actual matcher F0.5 was 0.957858 versus 0.959658 for the original pipeline. This experiment changed both shortlist width and ranking, so the ranking-only effect is not isolated. It is **not promoted**: better candidate recall has not yet produced a better final score.

The full-ground-truth ownership audit found:

| Pipeline | False links | Target has another labelled owner | That owner is outside tuning sample |
|---|---:|---:|---:|
| Original v3 | 424 | 223 | 219 |
| Postings v1 | 427 | 218 | 214 |

Ownership is therefore worth testing, but its benefit is not established by these counts. The correct competing owner must also be retrieved and scored. The inspected owner IDs have been recorded as exclusions for future fresh audits.

Of 618 empty predictions in the current tuning run, **537 are correct singletons and 81 are missed non-singletons**. The top candidate is true for only 45 of those references. A grid of simple top-candidate rescue rules produced a best gain of 0.0001569 while adding three singleton errors; none of the tested rules avoided additional singleton errors. No rescue rule has been adopted.

Larger experiment splits are now reserved under `models/scale_v1_plan/`:

- 300,000 training references, retaining the earlier 20,000 for controlled comparisons.
- 50,000 tuning references, retaining the current 10,000.
- 10,000 previously uninspected audit references. Their labels have not been used for model selection or error analysis.

The larger training run has completed. A single cache made with one supervised alias model cannot safely serve every fold. Full-universe out-of-fold ownership experiments require consistent fold-specific supervision throughout retrieval, aliases and matching; shared label-free indexes can still be reused.

A separate complementary-matcher control is testing removal of blocking score and eight alias-derived features. It uses cached training pairs and two CPU threads. Any resulting score is exploratory tuning evidence until confirmed on an untouched audit. None of these experiments changes the running submission.

The complementary-matcher experiment completed: a 25% blend of the 56-feature control with the existing matcher selected threshold 0.74 and reached 0.9595487 on tuning. The gain over 0.9594108 is only 0.0001379, with the same 33 singleton errors. This is too small to justify promotion without independent evidence; it does not establish progress toward 0.99. Larger leak-safe training and complete competing-owner evaluation remain the priority.

## Larger matcher experiment

The completed experiment under `models/scale_v1_run/` ran independently of submission 03. It passed the fresh 10k audit at 0.965888. Its source snapshot was frozen and checked before each stage. It implements:

1. Five inner entity folds within the outer training fold. Each training entity uses an alias model fitted without its entire inner fold, for both candidate retrieval and matching features.
2. Full outer-training aliases for tuning, audit and test inference. Their 653,487 confident mappings and 46,580 ambiguous mappings exactly match the existing inference aliases.
3. Checkpointed candidate and feature generation for 300,000 training entities and 50,000 tuning entities. Training excludes targets belonging to outer tuning/holdout identities. Unmatched targets receive deterministic fold assignments.
4. A 65-feature LightGBM matcher with inverse-candidate-count weights per reference, up to 1,200 trees and early stopping on tuning log loss. Float32 memory maps avoid loading all string metadata into model-training memory.
5. Threshold selection using actual macro F0.5, including every tuning reference and true links absent from retrieval. Compare with the existing matcher on exactly the same tuning candidates. Require precision within 0.002 of baseline and no increase in singleton errors.
6. If tuning improves, freeze model and threshold before generating the reserved 10,000-entity audit. Accept only when the paired 95% gain interval is above zero, precision drops by at most 0.002, and singleton errors do not increase. An unsuccessful tuning model leaves the audit unused.

The cache builder uses two workers while submission inference runs and can increase to eight once it completes. Model fitting waits for the submission to finish to avoid competing for memory. These are resource controls, not a promised completion time.

Validation before launch: 112 tests passed. A real-data pilot generated 14,514 retained training pairs for 500 training entities and 20,000 tuning pairs for 500 tuning entities. A 25-tree smoke fit exercised data loading, training, baseline comparison and threshold rejection; its numbers are not reported as an improved model score.

Monitor:

```sh
cat models/scale_v1_run/progress.json
cat models/scale_v1_run/cache_train/progress.json
tail -n 20 models/scale_v1_run/run.log
```

The full-run model report will be `models/scale_v1_run/model_300k/report.json`. A fresh audit, if reached, will be saved to `models/scale_v1_run/audit/report.json`. Neither is a leaderboard score. No automatic replacement of the running submission occurs.

Compatible new matchers can rescore submission 03's saved candidate features. Changed normalization, retrieval or feature definitions require a new versioned cache. One-owner decisions, sibling expansion and improved candidate retention remain separate experiments; they are not silently included in this training run.

To reproduce with the prepared split reservation and existing training indexes, run from the repository root in the pinned project environment:

```sh
python scripts/build_crossfit_aliases.py
python scripts/freeze_scale_experiment.py --output models/scale_reproduction --launch
```

Use a fresh output directory. `build_crossfit_aliases.py` reuses completed artifacts only when input fingerprints and settings match. The frozen runner resumes completed cache chunks after validating their hashes. It refuses to overwrite a partial model fit or reuse an already started audit for a different selection.

## Targeted retrieval diagnosis

`scripts/profile_missing_candidates.py` examined all 1,212 known retrieval misses from the existing 10,000-entity tuning comparison, using one low-priority worker. It did not inspect the reserved audit or modify either running job.

| Location of the missing link | Links |
|---|---:|
| Present in the original shortlist, then removed by the top-20 cut | 354 |
| Absent from the original shortlist | 858 |
| Missing candidate address, across both groups | 570 |

Increasing the fused shortlist budget from 64 to 128 made 502 of the known misses available before truncation. Keeping the original reranker recovered 88 into the final top 20; the experimental learned ranker recovered 490, including 345 links with missing candidate addresses.

This is an error-only diagnostic. It does not count previously retrieved true links lost by the new ranking, false matches, or the full runtime cost. It is not a macro F0.5 improvement and does not override the earlier full-pipeline probe that failed to improve matching. The next controlled comparison should measure both gains and losses on the full tuning set, using the larger matcher if it passes its own evaluation. Missing-address ranking and the 858 pre-shortlist misses deserve separate attention.

Aggregate results: `reports/experiments/redesign/missing_candidate_profile.json`. Detailed pairs remain in the ignored model-artifact directory.
