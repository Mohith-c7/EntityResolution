# Business Entity Resolution — Project Experience and Technical Retrospective

**Amazon ML Challenge 2026 · Development period: 25–27 September 2026**

## Outcome

This project grew from a lexical matching baseline into an offline entity-resolution system that processed millions of business records, generated bounded candidate sets, trained multiple matching stages, and produced validated competition submissions.

The best reported portal score was **0.970**, achieved by `submission_05`. The strongest separately audited local candidate reached **0.9811416633 macro F0.5** on 10,000 held-out reference entities. That later candidate did not become a completed submission before work stopped. The qualification target shown near the end was **0.989279**; the project did not demonstrate a result at that level.

These numbers described different evaluation populations. Local validation, development selection, candidate-retrieval ceilings, and portal scores were kept separate throughout this retrospective.

## 1. Understanding the task

The task involved matching records across three independent business-data sources. Source 1 contained deduplicated reference entities. Each reference could have zero, one, or several matching records in Sources 2 and 3. A useful solution therefore had to support both multi-match businesses and genuine singletons.

The official inputs were tab-separated files with four fields: an entity ID, a business name, an address, and a country. Training contained the US and India; the test set also contained France. Country handling had to remain open to unseen labels.

The initial data audit established the scale:

| Dataset component | Records |
| --- | ---: |
| Training Source 1 | 2,206,821 |
| Training Source 2 | 5,034,616 |
| Training Source 3 | 5,285,603 |
| Test Source 1 | 1,732,544 |

An all-pairs comparison was not practical. The problem required a retrieval system before it required a powerful classifier.

The metric also changed the engineering priorities. Macro F0.5 assigned equal weight to every Source 1 entity and penalized false links more strongly than missed links. A correct empty prediction on a singleton scored 1; any link on that singleton scored 0. Optimizing aggregate pair accuracy would have missed this behavior.

For a nonempty denominator, the per-entity metric was implemented as:

```text
F0.5 = 1.25 × TP / (1.25 × TP + FP + 0.25 × FN)
```

The both-empty case was handled explicitly as 1. The metric was later checked independently against the official formula on 256 cases.

## 2. Establishing reliable data contracts

The first phase focused on loaders, schema inspection, ground-truth parsing, and normalization. Explicit tab separators prevented TSV files from silently loading as a single column. String-preserving ingestion protected numeric-looking fields, and blank-field regressions were corrected.

The audit checked duplicate IDs, invalid prefixes, missing values, target references, train/test ID overlap, and target ownership. The provided training truth assigned no target record to multiple Source 1 entities. This observation later motivated ownership experiments; it was treated as an observed training property rather than proof of every unseen label.

Normalization handled Unicode, punctuation, whitespace, abbreviations, name cores, and numeric tokens. Several details proved important: empty strings could not be treated as ordinary tokens, a legal suffix could create false name similarity, and digits inside corrupted words could be mistaken for street numbers.

Stable interfaces allowed data loading, preprocessing, retrieval, feature generation, and model training to evolve separately. Integration still required explicit checks: changing normalization or a feature schema could invalidate an existing index or saved model.

**Lesson:** data contracts and normalization were part of the model, not preliminary housekeeping.

## 3. Building the first matching system

The first pipeline combined multiple blocking paths with string-similarity features and LightGBM. Candidate generation used names, addresses, number tokens, exact normalized forms, and name-core evidence. The classifier learned from retrieved nonmatches rather than relying mainly on easy random negatives.

A training-only name-alias channel added evidence for noisy business-name variants. The feature schema expanded from the initial 36 features to 50 and later 65. New evidence included numeric edit patterns, collapsed or domain-style names, script-related name views, and rarity information.

The early pilots showed substantial progress:

| Experiment | Local macro F0.5 | Interpretation |
| --- | ---: | --- |
| Initial lexical pilot | 0.8462164 | Small exploratory holdout |
| Improved retrieval, aliases, and 50 features | 0.9267946 | Small exploratory holdout; several components changed together |
| `larger_alias_v4`, 20,000 training references | 0.9510572 | Separate 2,000-reference holdout |

The larger model was also compared with the previous model on the same 2,000 entities. Its paired gain was **0.0174601**, with a 95% interval of **[0.0127344, 0.0225441]**. This was stronger evidence than comparing isolated scores from different samples.

Small candidate-budget experiments explored retaining fewer matches per source. They reduced work, but could also remove true matches. A higher score on one small holdout was not sufficient to select a new budget after inspecting that holdout.

**Lesson:** realistic hard negatives, broader training coverage, and better evidence produced more reliable gains than increasing model capacity alone.

## 4. Discovering the runtime bottleneck

The early submission attempts exposed a major design problem. Repeated SQLite full-text searches and per-reference ranking consumed most of the wall time. Reducing neural parameters would not have fixed this bottleneck because the initial system was primarily a retrieval and feature-computation workload.

Submission attempts 01 and 02 were stopped before completion. Their checkpoints were retained, but they were not successful portal submissions.

The retrieval engine was redesigned around bounded posting lists:

1. Retrieved token postings with explicit size limits.
2. Skipped oversized token blocks instead of accepting arbitrary first rows.
3. Cached reusable postings within a fixed memory budget.
4. Accumulated evidence through compact arrays and a native C component.
5. Formed field-specific shortlists and retained bounded candidates per source.
6. Stored candidate IDs, feature values, ranks, and probabilities in resumable Parquet chunks.

The earlier pipeline ran at roughly **20 references/second** in the recorded comparison. Wider posting-list benchmarks reached approximately **252–257 references/second** on 10,000-reference samples. These were benchmark measurements, not a guarantee for every later pipeline.

Compatibility was checked rather than assumed: 227,683 overlapping pairs retained identical values for all 64 non-blocking features. The blocking score changed with the retrieval method. Candidate and prediction equivalence was also checked between compact and reference implementations.

The first completed submission, `submission_03`, used the faster engine and scored **0.9399 on the portal**.

**Lesson:** redesigning the execution model mattered more than moving an inefficient query pattern to a larger machine.

## 5. Learning from the local-to-portal gap

The first portal result was lower than local estimates around 0.96. France was an obvious distribution-shift risk because it had no training labels, but the observed gap did not establish a measured French score.

Label-free diagnostics found more uncertain links and more contested targets in France. Generic names, locality words repeated in names and addresses, and legal-form differences could make distinct businesses look similar. Identical names did not reliably imply identical businesses.

Street-aware gates, name-ambiguity signals, and token-alignment features were explored. Some proposed gates changed almost nothing on the labeled development population. A more aggressive variant removed true links without removing false ones in its development check, so it was rejected.

The final documentation distinguished three statements:

- France exhibited suspicious prediction patterns.
- Those patterns justified targeted robustness tests.
- They did not reveal French ground truth or prove that a particular gate improved portal accuracy.

The portal gap was not treated as a fixed number that could simply be subtracted from every future local score.

**Lesson:** an unseen-country problem required evidence about generalization; matching another country's output distribution was not an accuracy target.

## 6. Scaling training and adding context

Training grew to **300,000 reference entities**, producing roughly 8.7 million candidate pairs. Five-fold training aliases reduced supervision leakage: a training entity's own fold did not supply its alias evidence. Shared label-free indexes remained reusable.

The larger first-stage matcher achieved **0.965888** on a fresh 10,000-entity audit, compared with **0.957786** for its control on the same entities. The paired gain was **0.008103**, with a 95% interval of **[0.006429, 0.009761]**.

The next stages added information beyond an isolated pair:

- Candidate rank, probability margins, and the strength of alternative candidates.
- Similarity to other high-confidence records associated with a reference.
- Counts and fuzzy evidence about competing Source 1 owners.
- Cross-fitted first-stage predictions for training the second-stage model.
- A rank-one rescue rule for some otherwise empty predictions.
- Bridge retrieval, where confident records supplied another retrieval query.

Bridge retrieval improved the achievable candidate ceiling, but a ceiling was only a counterfactual score with a perfect matcher. New candidates still had to be classified correctly.

The larger two-stage pipeline reached an audited local score of approximately **0.978547**. Its completed `submission_04` received a reported portal score of **0.968**.

**Lesson:** context helped, but list-level features required comparable candidate populations during training and inference. Filtering held-out targets from training lists changed their average size, which remained a potential distribution mismatch rather than a proven explanation of the portal gap.

## 7. Handling ownership correctly

The one-owner-per-target idea sounded simple but introduced several edge cases. A rule applied before low-threshold rescues could leave duplicate owners afterward. Exact probability ties also needed deterministic resolution. An unselected rival could incorrectly suppress a valid selected claim if arbitration happened at the wrong stage.

The later helper first formed the selected links and then resolved ownership among those links, choosing the highest probability and using the Source 1 ID as a stable tie-break. It did not create new matches.

Evaluation also had to include the competitors. Testing ownership on a small sample of reference entities omitted many rival owners and could produce misleading gains. The later audited decision graph contained **331,012 held-out owners and 13,240,480 candidate pairs** before results were sliced to the reserved audit population.

**Lesson:** global decision rules had to be evaluated on a sufficiently complete claim graph, not only on the references whose scores were reported.

## 8. Introducing a small offline neural matcher

When tree-model capacity stopped producing useful gains, a multilingual neural matcher was introduced on a sparse route of uncertain pairs. The model was `microsoft/Multilingual-MiniLM-L12-H384`, with **117,654,530 parameters** and an MIT license. Inference ran offline.

The work progressed from a frozen neural representation to controlled fine-tuning of upper encoder layers. Training used a sealed set of **120,000 pairs from 20,000 training owners**. Street-decoy examples taught the model that a shared business name with contradictory street evidence could indicate a nonmatch.

The neural probabilities were combined with 33 reverse-competition features, original stage-one and stage-two scores, candidate rank, and a second neural probability. A 38-feature adapter applied a correction to the original second-stage logit exactly once. Outside the neural route, original scores stayed unchanged.

The selected four-layer continuation passed a fresh audit:

| Same final-audit cohort | Control | Four-layer candidate |
| --- | ---: | ---: |
| Macro F0.5 | 0.9793161919 | **0.9799622272** |
| Link precision | 0.9952899973 | 0.9958352196 |
| Link recall | 0.9544283274 | 0.9553548163 |
| Singleton false predictions | 12 | 12 |

The gain was **0.0006460352**, with a paired 95% interval of **[0.0002035806, 0.0010892613]**. This pipeline became `submission_05` and reached **0.970 on the portal**.

**Lesson:** a small, selectively applied neural model added useful evidence without requiring neural inference over every candidate pair.

## 9. Generating and validating complete submissions

The completed Submission 05 output included:

| Output property | Count |
| --- | ---: |
| Test Source 1 rows | 1,732,544 |
| Scored candidates | 69,301,760 |
| Candidates per reference | 40 |
| Accepted links | 5,772,254 |
| Empty predictions | 99,399 |
| Empty-prediction rate | 5.73717% |

Every Source 1 record appeared once, including France and singletons. The candidate file represented the actual scored forward candidates; accepted matches were a subset. Both the strict validator and the unmodified organizer validator passed with ID checking enabled.

A reproducible package was also assembled with source code, pinned environments, model licenses, offline weights, documentation, and the two outputs. A Git clone was not equivalent to that package: large data files, checkpoints, and caches were intentionally excluded from Git.

Packaging revealed its own correctness risks. One validator distinction involved **1,473,940 peer triplets versus 801,380 distinct routed pairs**. The export-validation correction was documented separately from model behavior, and the scoring code, thresholds, route, and weights were checked for parity.

## 10. Experiments that did not justify promotion

Many plausible ideas did not improve the final system enough to justify a release:

| Experiment | What happened |
| --- | --- |
| Narrower candidate retrieval | Ran faster but lost too much matching quality. |
| Larger or less-regularized tree models | Often increased recall while lowering precision; capacity alone did not resolve the remaining errors. |
| Broad feature and token-alignment adapters | Several variants failed to improve the selected model on development data. |
| Sibling and peer neural evidence | Produced limited gains or failed to beat the simpler eligible candidate. |
| Missing-address candidate quotas | Recovered some true links but also increased false matches and singleton errors. |
| Standalone eight-layer continuation | Improved the development point estimate, but failed its original minimum-gain and precision guards. |
| Standalone empty-reference rescue | Added true matches but failed its original precision and singleton guards. |
| Macro-selected adapter checkpoints | Reached 0.9808512 against a 0.9809432 development control; the gain did not carry over from the early-selection sample. |

Historical rejection reports were retained. A later combination was evaluated as a separately declared candidate; its result did not retroactively turn the original component tests into successes.

**Lesson:** unsuccessful experiments were useful when they narrowed the error explanation, but neither more features nor a higher development point estimate automatically meant a better release.

## 11. The strongest local candidate and the final diagnosis

The final verified hybrid used an eight-layer neural correction on the original route, a four-layer correction on a disjoint route for empty predictions, calibrated decision thresholds, and post-selection exclusive ownership.

On a new, single-use 10,000-owner audit:

| Metric | Submission 05 comparator | Hybrid candidate |
| --- | ---: | ---: |
| Macro F0.5 | 0.9803742791 | **0.9811416633** |
| Link precision | 0.9956300509 | 0.9953443954 |
| Link recall | 0.9552856035 | 0.9578128141 |
| Singleton false predictions | 16 | 17 |

The paired gain was **0.0007673842**, with a 95% interval of **[0.0002402509, 0.0013411254]**. It passed its prospectively declared macro-gain and country conditions, while its precision decline and extra singleton error remained visible in the report. It did not supply any direct measurement of French accuracy.

After that audit was explicitly marked consumed, diagnostic analysis identified:

- 563 true links missing from the retained candidates.
- 906 true links retrieved but rejected.
- 156 accepted false links.
- Missing target addresses in **558 of the 906 retrieved-but-rejected links**.

The training set had only **122 missing-address negative pairs** against **1,969 missing-address positive pairs**. This was a concrete coverage weakness. A final balanced address-dropout experiment masked the target address in 10,000 positive and 10,000 negative examples, preserved keys and labels, excluded changed street decoys, and rejected contradictory text/label combinations.

The resulting eight-layer training run completed all 120,000 rows in about **646 seconds** on the Mac, and two development score sets completed. Calibration and final quality evaluation were not reviewed before shutdown. No improved macro score was claimed for this experiment.

Submission 06 export was paused and then terminated when the VMs were deallocated. It did not produce a completed, validated upload. The strongest completed portal submission remained Submission 05.

## 12. Infrastructure and operational experiences

Two Azure CPU VMs supported the larger experiments: a 64-vCPU machine with approximately 251 GB RAM and a 32-vCPU machine with approximately 125 GB RAM. GPU options were investigated, but the checked quotas were unavailable. Neural fine-tuning and scoring therefore used the Mac's MPS backend.

Available RAM did not imply unlimited throughput. Repeated queries, serial preparation steps, Python overhead, cross-machine transfers, hashing, and output validation could dominate even when memory remained free. Measured end-to-end timing was more useful than hardware specifications alone.

A checkpoint-resume failure also interrupted Submission 04. Chunk markers had stored the wrong hash even though the saved prediction files passed integrity checks. The repair changed checkpoint metadata rather than predictions, with backups and regression tests before resumption.

Parallel development required explicit ownership of files, processes, and artifacts. Separate Git worktrees and versioned directories reduced collisions. Saved manifests tied models, features, source code, and inputs together. Unfinished runs were not silently presented as completed experiments.

At project shutdown, both VMs were deallocated and their experiment resource groups, retained disks, networking, public IPs, schedules, and automatic Network Watcher resources were deleted. The active subscription's resource-group inventory was verified empty. Local files remained untouched. Remote-only temporary state was not a durable backup.

## 13. What the project taught

**Start with the error budget.** Candidate misses, rejected true links, accepted false links, and singleton errors needed different remedies. The best next experiment depended on which bucket could still deliver meaningful improvement.

**Measure the full path early.** Retrieval, feature generation, inference, arbitration, writing, and validation all contributed to submission time. A fast training run did not imply a fast submission run.

**Protect evaluation boundaries.** Whole-entity splits, out-of-fold supervised features, paired comparisons, and unused audits mattered more as gains shrank. Reusing an inspected audit would have exaggerated confidence.

**Keep feature and artifact identity explicit.** Pair keys, feature order, model versions, index versions, route definitions, hashes, and the probability anchor had to agree. Equal row counts were not enough to prove alignment.

**Treat missing data as a learning problem.** Empty addresses appeared disproportionately in the remaining matcher errors. A few handcrafted rules could not replace representative positive and negative training coverage.

**Use global constraints carefully.** Ownership improved consistency only when competing claims were available and decision order was correct.

**Separate research from release.** A candidate could be promising, audited, or fully exported; these were different states. Versioned submissions protected the last usable result while experiments continued.

**Be explicit about the outcome.** The project substantially improved both speed and measured quality, but did not reach the qualification target. Additional compute did not remove the need for better evidence, representative validation, and time reserved for complete submission generation.

## Evidence and further reading within the repository

- [Initial data audit](../reports/eda/local_download_audit.json)
- [Pilot results and larger training comparison](../reports/experiments/PILOT_RESULTS.md)
- [Retrieval redesign and runtime measurements](RUNTIME_REDESIGN.md)
- [Review-driven implementation plan](REVIEW_IMPLEMENTATION_PLAN.md)
- [Context, competition, and scaling plan](ROUND_2_MODEL_PLAN.md)
- [Final handover and release hashes](ER_HANDOVER_2026-09-27.md)
- [Submission 05 portal record](../reports/final_2h/submission_05_portal_result.json)
- [Four-layer final audit](../research/final_2h/neural_last4_final_audit_v1/report.json)
- [Hybrid extension audit](../research/final_2h/hybrid_audit/extension_evaluation_v1/report.json)
- [Consumed-audit error analysis](../reports/final_2h/hybrid_consumed_error_summary_v1.json)
- [Missing-address training coverage](../reports/final_2h/neural_training_address_coverage_v1.json)
- [Rejected macro-checkpoint experiment](../research/final_2h/macro_checkpoint_v1/report.json)

Historical planning documents describe what was intended at the time. Completed reports and the final status take precedence when they disagree with an earlier plan.
