# Entity Resolution: Review Implementation Plan

26 September 2026

## Current direction

Improve matching evidence and reduce retrieval cost before another full submission run. The original baseline remains frozen under `output/submission_01/`, stopped at 37,000 completed references. It has no upload-ready output and no leaderboard score. Resume it only on an explicit decision to do so; its checkpoints remain usable.

The review's 0.957153 result is an exploratory score on the previously inspected 2,000-reference holdout. It is evidence for an experiment, not a confirmed replacement for the 0.951057 baseline. The completed fresh comparison below determines its quality acceptance.

## Completed fresh audit

Saved model: `models/features_v3_audit01`; feature version `pairwise-v3-65`; threshold `0.76`.

| Metric on the same 5,000 unused references | Frozen v2 baseline | Version 3 |
|---|---:|---:|
| Macro F0.5 | 0.948512 | **0.960616** |
| Link precision | 0.981779 | 0.988304 |
| Link recall | 0.901517 | 0.916518 |
| Singleton false positives / 258 | 27 | 19 |
| US macro F0.5 | 0.959463 | 0.968440 |
| India macro F0.5 | 0.931063 | 0.948151 |

Paired macro gain: **0.012104**, with 95% entity-bootstrap interval **[0.009178, 0.015012]**. All three predefined audit gates passed. Version 3's own macro interval is [0.957222, 0.963927].

The baseline's 0.951057 result used a different, older 2,000-reference cohort. Compare the two columns above for the valid paired gain. Neither score is a leaderboard result. Retuning the baseline on the same 10,000 tuning entities retained threshold 0.71; that control was frozen before audit scoring.

The fresh audit contains 200,000 scored pairs, 40 per reference. Blocking recall is 0.968038; the retained-candidate oracle is **0.989322**. This ceiling is still below 0.99 on this cohort, so reaching 0.99 requires better retained candidates as well as matching improvements. The audit is now consumed; further model selection requires another unused audit.

Compatibility checks reproduced all 65 features on 120 actual indexed pairs and reloaded probabilities on 120 saved tuning pairs with zero difference. The test suite passes 112 tests. The model, provenance, source snapshot and aggregate audit report are saved.

## Completed runtime experiment

Six runs used the same 600 previously inspected tuning references, eight workers, and the v2 model. The order was baseline → mmap → mmap without numeric/location → reverse order. Rates below divide total references by total time across both repetitions; they include worker startup, retrieval, features and predictions. Short timings on a desktop are indicative, not full-test time guarantees.

| Variant | References/second | Macro F0.5 on this probe | Decision |
|---|---:|---:|---|
| Baseline | 17.74 | 0.958426 | Quality/control configuration |
| Memory mapping | 19.99 | 0.958426 | About 13% more throughput; exact candidates and predictions preserved |
| Memory mapping, numeric/location path removed | 21.92 | 0.956390 | Not adopted: matcher quality fell despite slightly higher oracle |

Memory mapping was tested in the benchmark without changing the production default or frozen artifacts. These rates remain far below the 120-reference/second target. Batch retrieval and learned reranking are the next engineering priority. Full submission generation remains stopped.

## 1. Feature version 3

Implemented schema: `pairwise-v3-65`. The existing 50 columns retain their names, order and meaning. Fifteen additional columns describe:

- Numeric containment, one-digit substitutions, disjoint evidence, and unmatched numbers on each side.
- Approximate phonetic agreement from Unicode character names, character trigrams and collapsed name strings.
- Minimum name-token document frequency on both sides, using the relevant target source without labels.
- Unmatched address-token weights and distinctive token disagreement.

The word-final inherent-vowel treatment is consistent at both spaces and the end of a string. This is an approximate comparison view, not a complete transliterator. Primary Unicode text remains intact. Leetspeak folding applies to alternate name comparisons. Standalone address numbers have their own evidence view; legitimate alphanumeric identifiers remain available in the original features. No global replacement of `1` with `l`, or `0` with `o`, is applied to addresses.

This experiment retains the original normalization, aliases, indexes, retrieval paths, top-20-per-source budget and classifier capacity. French legal forms and learned candidate ranking are separate experiments because they change candidate generation.

## 2. Selection and audit protocol

Training reuses the 582,358 eligible pairs from the baseline's 20,000 training references. New tuning and audit references are sampled from the existing entity-disjoint folds, excluding IDs recorded in previous experiment reference manifests. Sampling uses ID hashes, without labels.

| Population | Size | Use |
|---|---:|---|
| Training references | 20,000 | Fit the classifier using training-fold pairs only |
| New tuning references | 10,000 | Early stopping, threshold selection and precision/singleton constraints |
| New audit references | 5,000 | One paired comparison after model and threshold freeze |

Choose the threshold with the highest tuning macro F0.5 among thresholds satisfying both conditions on those same tuning entities:

1. Precision is no more than 0.002 below the frozen baseline.
2. Singleton false positives do not exceed the frozen baseline.

The fresh audit compares both models on identical references and candidates. Acceptance requires a paired entity-bootstrap gain interval entirely above zero, precision loss no greater than 0.002, and no increase in singleton false positives. Report a failed gate even if the main score rises. Do not adjust the threshold using audit results.

The audit records hashes for the model, aliases, source files, training/tuning pairs and reference manifest before scoring the held-out cohort. The alias channel is still the original training-fold channel. Out-of-fold aliases remain a separate experiment; this run does not claim to solve training-entity self-contribution.

Run from the repository root:

```bash
python scripts/run_features_v3.py \
  --baseline models/larger_alias_v4 \
  --artifact-dir models/features_v3_audit01 \
  --tune-entities 10000 --audit-entities 5000 \
  --workers 8 --threads 8
```

The script refuses to reevaluate a completed audit. Use the saved artifacts to inspect its result; do not run a new selection against the same audit.

## 3. Retrieval runtime

Measure end-to-end candidate retrieval, features and matching predictions on the same previously inspected tuning entities. Test the baseline, memory mapping, and memory mapping without `numeric_location`, then reverse the order to expose cache/order effects. Include process startup and report the benchmark scope; final TSV writing and validation are outside this short timing test.

```bash
python scripts/benchmark_retrieval_options.py \
  --artifact-dir models/larger_alias_v4 \
  --entities 600 --workers 8
```

Memory mapping must preserve candidate IDs, order and predictions. Removing a retrieval path changes the evidence distribution and cannot be selected from speed alone. Reject an oracle loss greater than 0.002; also check actual precision, singleton errors and macro F0.5. A small aggregate oracle loss can conceal damage to missing-address or rare-name cases, so inspect those slices before release.

The installed SQLite build caps a requested 4 GiB map at 2,147,418,112 bytes per database. The actual setting must be recorded. Mapping is a measurement candidate, not a guaranteed speedup. [SQLite memory-mapping documentation](https://www.sqlite.org/mmap.html).

If quick changes remain far below 120 references/second, move to batch key joins:

1. Emit versioned keys and target-side frequencies from the supplied normalized records: rare name/address tokens, selective token pairs, number-plus-address tokens, name cores and alternate-script names.
2. Join only bounded blocks. Combine common tokens into more selective keys before skipping them. Log which paths lose true links.
3. Aggregate evidence per unique S1/target pair and retain a bounded preliminary pool per source.
4. Apply a cheap learned ranker, then the matching model. Report the preliminary pair volume, ranking volume and final scoring volume separately.
5. Partition by country for execution, while retaining documented routes for missing or inconsistent country labels. Accept arbitrary country labels; do not drop France or turn country equality into an untested mandatory match condition.
6. Stream chunks and partitioned files within a measured memory budget. Benchmark on full target pools, not a reduced database.

A 120-reference/second target implies approximately four hours for the 1,732,544 test references before finalization overhead. It is a target, not a current measurement.

## 4. Ranking and missing-address recovery

Collect pre-truncation training candidates and label them using training-fold truth. Fit a LightGBM ranker with inexpensive similarities, missingness, numeric edit types and retrieval-path indicators. Avoid training on tuning or audit owners, including their target records.

Compare pre-cap recall, retained recall and macro oracle on the same tuning set. Try a small reserved missing-address pool and adaptive budgets of 3–20 per source. A missing-address reservation supplies candidates, not automatic matches. Select budgets jointly with final matching quality and candidate counts.

Target retained-candidate macro oracle: at least 0.995 with at most 40 mean final candidates. Verify both mean and tail counts. A 100-reference probe is insufficient to establish this target.

The final `candidate_pairs.tsv` must contain exactly the pairs fed to the final matching model. If a learned ranking stage scores a wider preliminary pool, disclose its inputs and cost in the methodology and retain stage diagnostics. Do not hide ranking work behind the final candidate count.

## 5. France and country transfer

Treat the partial French prediction tail as a distribution warning. No French labels are available, so it cannot establish French error rates. Do not impose an 8-link cap, use the training maximum of 11 as a test limit, or optimize French predictions to a presumed 1% tail.

Version legal-form handling for `sarl`, `sas`, `sasu`, `eurl`, `sa`, `sci`, `snc`, `cie` and `selarl`, including tests for dotted forms and ambiguous name tokens. Preserve the full primary name as a separate view. Rebuild affected core/prefix indexes into new paths, refit aliases consistently, and retrain before evaluating the complete change.

Evaluate US/India labeled slices with city words in names, names repeated in addresses, shared buildings, missing addresses, absent aliases and unseen name families. Fit supervised preprocessing inside each transfer split. Track country-wise probability histograms, empty predictions, links per reference and name/address overlap on test data as diagnostics only.

## 6. Team deliverables

| Owner | Next deliverable |
|---|---|
| Harsha | Version 3 training and audit; learned reranker; retrieval runtime benchmarks; model selection and final submission integration |
| Mohit | Reproducible evaluation manifests; blocking-loss and memory/runtime reports; partitioned retrieval data preparation; full input and artifact audit |
| Sanhitha | Versioned legal-form normalization; alternate-script and alphanumeric fixtures; index/normalization compatibility checks |
| Sahasra | Submission format checks, methodology tables, reproducibility command checks and aggregate report review |

## 7. Release gates

Select on tuning data, freeze assets, evaluate once on a new audit, and benchmark throughput before starting full test inference. Keep every test Source 1 record, including France and empty predictions. Generate both exact TSVs, run the strict and official validators with ID checks enabled, and preserve versioned outputs and all model/configuration assets.

All identities and training evidence come from the supplied challenge data. No external business lookup, geocoding or hosted inference is used. Local scores, oracle ceilings and leaderboard scores must remain separately labeled.
