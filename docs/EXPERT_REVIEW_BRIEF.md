# Business Entity Resolution — Technical Review Brief

Prepared: 26 September 2026, approximately 3:42 PM IST.

Repository: https://github.com/Mohith-c7/EntityResolution/tree/Harsha

## Message to send with this brief

We are working on the Amazon ML Challenge business entity resolution problem. We need to match each deduplicated Source 1 business to zero or more records from Sources 2 and 3. The official metric is macro F0.5 per Source 1 entity, including singletons.

Our best saved pipeline uses Unicode-preserving normalization, SQLite FTS5 candidate retrieval, heuristic reranking, 50 pair features, a training-fitted name-variant channel, and LightGBM with a tuned threshold of 0.71. It achieves **0.951057 local held-out macro F0.5**, with link precision 0.984827 and recall 0.904598. This is not a leaderboard score.

We have two problems. First, the retained candidates have an oracle macro F0.5 ceiling of **0.989379** on that holdout, so a better classifier alone cannot reach 0.99 on those same entities. Second, full test inference covers **1.73 million references against 9.97 million targets** and currently processes about **19 references/second**, projecting roughly a day of execution.

Profiling points to candidate retrieval and reranking as the main runtime bottleneck. The matcher itself is only about 4.3 MB; it scores 80,000 pairs in about 6.9 seconds on one thread. We need advice on improving retrieval selectivity and recall together, reducing false merges, improving unseen-country generalization, and deciding whether CPU cloud infrastructure is justified.

Please review the implementation and measurements below and propose a ranked experiment plan. We would value concrete algorithms, expected tradeoffs, validation requirements, and a practical inference design rather than a promise of a particular score. All business identity evidence must come from the supplied challenge data.

## 1. The challenge

Each record contains:

```text
entity_id
business_name
business_address
country
```

All input and output files are tab-separated. IDs identify the source through `S1-`, `S2-`, or `S3-`; they are not shared business identifiers.

Source 1 is deduplicated. One Source 1 entity can have zero, one, or multiple matching records in either target source. A one-to-one assignment within each target source would therefore be incorrect.

Noise includes name typos, abbreviations, legal forms, trade names, word reordering, transliteration, partial addresses, missing components, landmark references, numbering differences, and omitted postcodes.

Training covers the US and India. Test data also contains France. Country labels must remain open strings, and every French reference must receive an output row. We have no labeled French validation set.

The organizer also reviews candidate-generation quality and efficiency for final rankings. A smaller scored candidate set is useful only when matching quality is preserved.

### Required outputs

```text
matching_results.tsv
  source1_entity_id<TAB>matched_entity_ids

candidate_pairs.tsv
  source1_entity_id<TAB>candidate_entity_ids
```

Both files require exactly one row for every test Source 1 entity. Empty ID lists remain empty fields. Lists contain existing S2/S3 IDs only, without duplicates. Final matches must be subsets of the exported candidates.

The candidate file must contain **exactly the final candidate set scored by the matching model**. It must not be shortened after scoring to make blocking appear more efficient. Earlier heuristic retrieval/reranking candidates may be larger than this final set.

### Permitted methods

- Use the supplied data for training and identity evidence.
- No external business registries, identity lookup services, geocoding, or internet data augmentation.
- Pretrained open-weight models are allowed only when each model independently meets the MIT/Apache-2.0, at-most-8B-parameter, offline-inference, and supplied-data-only fine-tuning requirements.
- Hosted model APIs cannot be used for matching.
- Our selected pipeline currently uses no pretrained neural model. The tree model and learned name-channel assets carry an MIT license.

## 2. Dataset scale and audit

| Source | Training rows | Test rows |
|---|---:|---:|
| Source 1 | 2,206,821 | 1,732,544 |
| Source 2 | 5,034,616 | 4,887,273 |
| Source 3 | 5,285,603 | 5,082,316 |
| Searchable S2 + S3 pool | 10,320,219 | 9,969,589 |

The training ground-truth file contains 2,206,821 reference rows and 7,638,365 matched target IDs. The audit found valid ID references and no target assigned to multiple training S1 owners.

Training has 123,247 singletons, approximately 5.58% of references. Mean matches per reference are about 3.46, the median is 3, and the observed maximum is 11.

Test Source 1 country counts:

| Country | References |
|---|---:|
| India | 809,986 |
| US | 663,106 |
| France | 259,452 |

The files have been audited for schemas, malformed rows, duplicate IDs, source prefixes, and ground-truth integrity. Training and test IDs are disjoint within each source. That does not establish that every name family is disjoint across splits.

## 3. What the score means

The main score is macro F0.5, not ordinary pair classification accuracy and not micro precision.

For a reference with nonempty truth:

```text
F0.5 = 1.25 × TP / (1.25 × TP + FP + 0.25 × FN)
```

Calculate this separately for each S1 entity, then average over all S1 entities. A singleton scores 1 when its predicted list is empty and 0 when any link is predicted.

This makes false merges, especially false links on singletons, costly. Reporting 98.48% link precision does not mean our official metric is 0.9848.

The desired target is 0.99 or higher. Reported leading leaderboard scores are approximately 0.988–0.99. We have not verified a leaderboard score for our own pipeline or established that this approach can reach that target.

## 4. Current implementation

```text
Official TSVs
    ↓
String-preserving loaders and integrity checks
    ↓
Deterministic normalization
    ↓
Complete target-source indexes
    ↓
Multiple retrieval paths
    ↓
Candidate union and heuristic reranking
    ↓
Up to 20 retained candidates per target source
    ↓
50 pair features
    ↓
LightGBM probability
    ↓
Global threshold 0.71; retain zero or multiple matches
    ↓
Exact scored-candidate export and matching export
    ↓
Strict validator and unmodified organizer validator
```

### Normalization

- Preserve strings and leading zeros during loading.
- Apply Unicode normalization, case folding, whitespace cleanup, and punctuation handling.
- Preserve combining marks in scripts such as Devanagari and Odia.
- Use a separate Latin-accent-folded view.
- Expand a small set of business-name abbreviations and remove legal-form tokens in a separate name-core view.
- Preserve the primary address view without aggressive expansion of ambiguous abbreviations.
- Extract ordered numbers and numeric sets; retain original numeric strings, with a zero-stripped alternate view for selected features.
- Normalize country formatting without a country whitelist or country one-hot features.

Address numeric tokens of at least four digits are treated as uncertain postcode candidates. The first ordered number is a proxy; it is not reliably identified as a street number.

### Retrieval and blocking

Each complete target source has a disk-backed SQLite index containing normalized records, B-tree exact-name/core/prefix lookups, FTS5 word retrieval, and label-free corpus-frequency statistics. Character retrieval exists but is disabled in the selected run.

The original seven paths are:

1. Rare name tokens.
2. Rare address tokens.
3. Rare digit/number tokens.
4. Exact normalized name.
5. Exact name core.
6. Name-core prefix.
7. Postcode candidate plus name token.

The selected configuration also includes numeric conjunctions, numeric-plus-address-tail queries, and learned name-variant queries combined with number/address evidence.

Important settings:

```text
retrieval_mode: wide
top_k: 20 per source
path_top_k: 80
query_tokens: 6
max_word_df: 100000
max_address_df: 100000
character_mode: off
reranker_version: v4
numeric_conjunctions: enabled
numeric_location_queries: enabled
use_name_aliases: enabled
```

The union is deduplicated and heuristically reranked using name, address, numeric, exactness, and path evidence. Reranking includes an address-led score that allows a strong address to retrieve a business with a substantially changed name. This is useful for trade names and transliteration, but could also rank different businesses at a shared address too highly.

The final model receives at most 40 pairs per reference. The selected samples and the current test prefix all average 40 candidates. Earlier retrieval still evaluates hundreds of records per reference; a small exported set does not imply cheap retrieval.

### Learned name-variant channel

This is a fitted string mapping, not a hosted service or an LLM. It learns target-name variants associated with canonical S1 name cores using training-fold ground truth only.

The saved asset was fitted on approximately 1.545 million training references and 5.348 million matched targets. It contains 653,487 confident alias strings and 46,580 ambiguous strings.

Confident mappings require support of at least 2 and dominance of at least 0.98. Ambiguous strings retain up to three supported alternatives. Pair features use reference-specific evidence rather than treating every ambiguous alias as equivalent.

Validation labels are excluded from fitting. However, a classifier-training entity can contribute to the same channel used for its training features. Out-of-fold channel construction is a planned generalization check. This possible training shortcut must be distinguished from validation-label leakage.

### Features and model

The initial design had 36 features; the selected version is `pairwise-v2-50`.

Feature groups include:

- Jaro-Winkler, edit, token-sort, token-set, partial, Jaccard, and length similarities for names and addresses.
- Binary character 3/4-gram cosine; this is not a fitted dense embedding or TF-IDF vectorizer.
- Number overlap, uncertain postcode agreement, first-number agreement, and canonical numeric comparisons.
- Exact/core agreement, cross-field overlaps, missingness, lengths, and source indicators.
- Learned canonical-name similarities, alias confidence/support, and family frequency.
- Corpus-weighted name/address overlap and distinctive address evidence.
- The heuristic blocking score.

The blocking score has substantially more recorded gain importance than the other individual features. It may be a useful summary or an overdependence; gain importance alone does not establish causality.

Selected model:

```text
LightGBM: 4.3.0
Objective: binary classification
Trees: 1200
Maximum leaves per tree: 31
Learning rate: 0.04
Minimum child samples: 30
L2 regularization: 2.0
Feature subsampling: 0.90
Row subsampling: 0.90
Max bin: 127
Deterministic CPU fitting, seed 42
Decision threshold: 0.71
Model artifact size: approximately 4.3 MB
```

Training uses blocked nonmatches as hard negatives rather than primarily random unrelated pairs. Each reference contributes total training weight one. No forced match, learned singleton gate, global target-competition rule, ensemble, or transitive merge is currently applied.

## 5. Validation protocol

Stable S1 hashes create 70% training, 15% tuning, and 15% holdout folds. The split is entity-level and row-order-independent, but not exact stratification.

The selected classifier run uses 20,000 training references, 2,000 tuning references, and 2,000 held-out references. The name channel is fitted separately on the complete training fold.

Candidate retrieval is evaluated against all 10.32 million training targets. Held-out target identities are excluded from supervised classifier-training pairs, including negative pairs. Index frequencies are label-free and use the complete target pool. Tuning and holdout retrieval retain the full distractor population.

The threshold is selected on tuning macro F0.5. Previously examined holdouts are now exploratory evidence. Final changes should be frozen and evaluated on at least 5,000 previously unused holdout references, excluding every already inspected entity.

Name-family holdouts and US/India transfer checks remain to be completed with supervised preprocessing refitted inside each split. None of these checks can directly establish French performance.

## 6. Best saved results

Artifact: `models/larger_alias_v4`.

| Measurement | Tuning: 2,000 S1 | Holdout: 2,000 S1 |
|---|---:|---:|
| Macro F0.5 | 0.954912 | **0.951057** |
| Micro link precision | 0.984279 | 0.984827 |
| Micro link recall | 0.906340 | 0.904598 |
| Blocking link recall | 0.966995 | 0.970115 |
| Oracle macro F0.5 on retained candidates | 0.991005 | 0.989379 |
| True links | 6,908 | 6,960 |
| Correctly predicted links | 6,261 | 6,296 |
| Incorrectly predicted links | 100 | 97 |
| True links lost in blocking | 228 | 208 |
| Retrieved true links rejected | 419 | 456 |
| Singleton entities | 122 | 81 |
| Singletons with false predictions | 8 | 4 |
| Mean candidates per S1 | 40 | 40 |

Holdout country scores are 0.960051 for the US and 0.936099 for India. The entity-bootstrap 95% interval for held-out macro F0.5 is approximately [0.944841, 0.956882].

On these same 2,000 held-out entities, the earlier smaller-training model scores 0.933597. The selected model's paired gain is 0.017460, with bootstrap interval [0.012734, 0.022544]. Comparisons against an earlier 500-entity holdout are not equivalent matched-cohort comparisons.

The oracle scores assume ground-truth knowledge and perfect classification of retained pairs. They are diagnostic ceilings, not deployable scores. The holdout ceiling being below 0.99 means that both retrieval and matching need improvement for that target on this cohort.

## 7. Error evidence

The tuning sample has 228 blocked true links, 419 retrieved-but-rejected true links, and 100 false predictions.

Among the 419 rejected links:

- 368 lack learned alias evidence.
- 165 have high address-token similarity.
- 83 have a missing candidate address.

These slices overlap. They are investigation areas, not proven causes or automatic acceptance rules. Only 62 of the 519 matcher errors lie within 0.05 of the threshold; adjusting the threshold alone is unlikely to resolve the majority.

Important questions include whether we distinguish:

- Number absence from a real numeric contradiction.
- Component reordering from incorrect address components.
- Name abbreviations/typos from different businesses in a common name family.
- Strong address evidence for an unknown trade name from businesses sharing premises.
- Truly uncertain pairs from examples with sufficient evidence missing from current features.

Possible issues to audit include alignment between Python Unicode tokens and FTS5 tokenization, accent handling in frequency lookups, and overdependence on the blocking score or training-fitted aliases. These are hypotheses requiring controlled tests.

## 8. Experiments already attempted

| Experiment | Evidence | Current decision |
|---|---|---|
| Larger classifier-training sample | Paired holdout gain from 0.933597 to 0.951057 | Selected |
| Longer boosting/lower regularization | Tuning 0.955451 versus 0.954912; precision decreased | Not selected |
| More leaves | Tuning 0.955490; precision decreased | Not selected |
| Remove blocking score | Small pilot tuning gain, lower precision | Not selected |
| Remove blocking and direct alias scores | Small pilot tuning gain, larger precision decline | Not selected |
| Fixed K=5 instead of K=20 per source | Pilot candidates fell from 40 to 10; tuning score and oracle ceiling decreased | Needs larger-sample/adaptive-budget evaluation |
| Fast unranked compound retrieval | 100-reference probe about 5.1 seconds, but link recall only 0.8911 | Rejected for production |
| Selective lexical bridge prototype | One 100-reference tuning probe had recall 0.9741 and oracle 0.9950 | Preliminary; no promoted matching score or representative validation |
| Exact BM25 frequency-reuse prototype | Candidate IDs, records, scores, and paths identical on 100 tuning references; 1.19× measured speedup | Not integrated into submission |

Small probes are diagnostic. Their timing, recall, or oracle score must not be presented as a verified full-pipeline result. The exact-ranking benchmark ran under concurrent inference load and includes cache/order effects.

## 9. Why full inference takes so long

The saved pipeline retains up to 40 final candidates per S1:

```text
1,732,544 references × 40 candidates = 69,301,760 matching predictions
```

The more expensive work happens before that stage. Each reference performs multiple ranked FTS queries against both complete source indexes. Broad token unions visit many postings. SQLite BM25 also counts query-phrase frequencies, and the pipeline constructs records and runs string reranking on the union before truncation.

A 50-reference profile reranked approximately 640 retrieved records per reference before the final cap. This is a sample measurement, not an exact test-wide average.

Profiles put FTS searching at approximately 70–80% of candidate-generation time. Feature construction, Python reranking, record materialization, and I/O add work. These percentages do not constitute a complete independently instrumented end-to-end timing breakdown.

The name/core exact indexes and FTS indexes do avoid the full Cartesian comparison. Nevertheless, reducing billions of possible comparisons to 40 exported pairs does not by itself make the preceding search efficient.

### Model-only timing

An 80,000-pair cached feature matrix was scored in:

| Prediction threads | Seconds |
|---|---:|
| 1 | 6.87 |
| 4 | 1.75 |

These exclude retrieval, feature construction, and output validation, and were measured while the full run was active. The actual inference runner uses multiple worker processes and one prediction thread per worker.

Reducing tree count could shorten prediction time but would leave most retrieval work intact and require a new quality comparison. A GPU allocation would not automatically accelerate the current SQLite/Python stages.

### Throughput targets

Ignoring index construction and validation overhead:

| Desired full-inference time | Required S1 throughput |
|---|---:|
| 1 hour | 481 references/second |
| 2 hours | 241 references/second |
| 4 hours | 120 references/second |
| 8 hours | 60 references/second |

Current observed throughput is approximately 19 references/second. A four-hour run therefore needs roughly a sixfold throughput improvement at the current observed rate. This is a target, not a demonstrated capability.

## 10. Current first-submission status

A frozen copy of the selected model, alias asset, feature registry, code, and configuration is under `output/submission_01/`. Nine critical inference modules match recorded training hashes. Replaying 120 saved pairs reproduced their features and probabilities exactly.

Full inference is active, detached from the interactive session, and checkpointed in batches of 100 references using 12 workers. On resumption, asset/input hashes, completed chunk checksums, and Source 1 input order are checked.

Snapshot at approximately 3:42 PM IST on 26 September 2026:

| Measurement | Partial result |
|---|---:|
| Completed references | 28,200 / 1,732,544 |
| Completed France references | 4,243 |
| Predicted links | 93,295 |
| Scored/exported candidate links | 1,128,000 |
| Empty predictions | 1,634 |
| Estimated inference remaining | Approximately 24.8 hours at snapshot throughput |
| Ready for upload | No |

These are partial counts and become stale as execution continues. Empty predictions are predicted no-match entities; true test singleton prevalence is unknown.

After complete coverage, the runner assembles both files, runs the strict validator and the unmodified official validator with ID checks enabled, and publishes complete files at `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The versioned copies remain in `output/submission_01/`.

Both validators have been exercised on small end-to-end fixtures. They have not yet passed on the complete official-test outputs because full inference has not finished. The repository test suite passes 108 tests.

No first submission has been uploaded, and no portal score is known. The current priority decision is whether to keep waiting for the unchanged baseline or redirect effort toward a faster, separately validated retrieval pipeline while preserving the baseline and its checkpoints.

## 11. Cloud and implementation options

More CPU workers and fast storage may reduce wall time. References can currently be sharded independently because the final decision does not use global cross-reference competition. Every shard still needs access to both complete target sources and the same frozen model/configuration. Combined files require exact coverage and duplicate checks.

A CPU instance with 32–64 vCPUs and 64–128 GiB RAM is a candidate for benchmarking. AWS C8g/C8gd instances offer configurations in this range; C8gd includes local NVMe storage. This is a hardware option, not a verified speed or cost estimate. [AWS instance specifications](https://aws.amazon.com/ec2/instance-types/c8g/)

A 10,000-reference benchmark should determine actual throughput, peak memory, storage behavior, scaling efficiency, and projected cost before committing to a full cloud run. Faster storage, larger caches, process count, and per-core speed all affect the result; vCPU count alone does not establish speedup.

Cloud execution would mean running the supplied-data pipeline locally inside a VM. It would not use hosted matching services or external business lookup. No cloud VM has been provisioned and no dataset has been uploaded to one.

The main algorithmic options to assess are selective compound blocking, adaptive fallback paths, better candidate ranking, compact inverted indexes, batched/vectorized operations, and adaptive candidate budgets. Any changed retrieval pipeline needs matching-quality evaluation; preserving the classifier file alone does not preserve the reported score.

## 12. Questions for the reviewer

1. **Architecture:** Is this feature-based matcher a suitable foundation for very high macro F0.5 on these noise patterns? Which additional stage would be justified by our error evidence?
2. **Retrieval design:** What concrete strategy can reach an oracle macro F0.5 of at least 0.995 while materially reducing posting-list work, reranking volume, and final candidate count?
3. **Runtime:** Should we replace broad FTS unions with compound inverted keys, use a different retrieval engine, implement batch similarity joins, or introduce approximate retrieval? Which option is most practical at this scale and why?
4. **Adaptive budgets:** How should easy versus ambiguous references be identified without inference-time labels? What evidence should trigger expansion or fallback?
5. **Features:** Which additions most directly address missing addresses, number-format variations, shared premises, and unseen name variants? How should contradictory evidence affect decisions?
6. **Training:** Would more classifier-training entities, targeted hard-negative mining, or out-of-fold alias features be more valuable than larger trees or an ensemble?
7. **Group evidence:** Could candidate-to-candidate consistency or exclusive target ownership improve precision/recall without unsafe transitive merges? How should this be evaluated without using validation labels to create features?
8. **Generalization:** What realistic name-family and country-transfer tests should we run before using the pipeline on France? Would a permitted multilingual model help enough to justify its inference cost?
9. **Infrastructure:** What CPU/RAM/storage configuration should we benchmark, and how should we shard and merge work reproducibly? What measured throughput would justify cloud spending?
10. **Validation:** Are our split, training-target exclusions, alias fitting, threshold selection, and fresh-audit plan sufficient? What additional leakage or distribution-shift checks are needed?
11. **Priority:** What are the top three changes to implement first, and what quality/runtime acceptance criteria should each satisfy?

Please distinguish recommendations supported by these measurements from hypotheses needing experiments. A useful answer should specify what to implement, what to measure, and when to reject an approach.

## 13. Reviewable files and methodological references

Repository paths:

```text
code/business_entity_resolution/src/blocking/disk_index.py
code/business_entity_resolution/src/blocking/anchor_index.py
code/business_entity_resolution/src/features/registry.py
code/business_entity_resolution/src/features/pairwise_features.py
code/business_entity_resolution/src/model/name_aliases.py
code/business_entity_resolution/src/model/train.py
code/business_entity_resolution/src/model/threshold.py
code/business_entity_resolution/src/evaluation/validation.py
code/business_entity_resolution/src/evaluation/metrics.py
code/business_entity_resolution/src/pipeline/training.py
code/business_entity_resolution/src/pipeline/inference.py
scripts/create_submission.py
docs/SCORE_IMPROVEMENT_PLAN.md
reports/experiments/larger_alias_v4.json
reports/experiments/larger_tuning_error_budget.json
```

Fitted models, raw datasets, and generated outputs are local ignored artifacts; they are not included in the public Git repository. Experimental runtime/retrieval prototypes are not automatically part of the selected submission.

Relevant technical references consulted:

- [SQLite FTS5: BM25, ranking, and extension APIs](https://www.sqlite.org/fts5.html) — explains the rank computation used by our index.
- [SQLite 3.50.4 BM25 implementation](https://github.com/sqlite/sqlite/blob/version-3.50.4/ext/fts5/fts5_aux.c) — source used to investigate exact frequency-reuse acceleration.
- [Generalized Supervised Meta-blocking](https://www.vldb.org/pvldb/vol15/p1902-gagliardelli.pdf) — motivates assessing candidate pruning independently from final matching quality.
- [A Comparison of String Distance Metrics for Name-Matching Tasks](https://www.cs.cmu.edu/~wcohen/postscript/ijcai-ws-2003.pdf) — motivates hybrid token/edit evidence; its benchmark results are not our challenge results.
- [Cross-fitting explanation in scikit-learn](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html) — supports the reasoning behind auditing supervised preprocessing shortcuts; our alias channel is custom.
- [LightGBM 4.3.0 parameters](https://lightgbm.readthedocs.io/en/v4.3.0/Parameters.html) — documents our model's tree/thread parameters.

These references supply technical methods only. No external business identity evidence has been used.
