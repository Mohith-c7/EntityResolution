# EntityResolution — Score Improvement Action Plan

Date: 26 September 2026

## Objective

Improve macro F₀.₅ while retaining a small, auditable candidate set and practical full-test runtime. A top-10 position is the goal; neither local validation nor model complexity guarantees a rank.

Execution order: finish Submission 01 with the verified `larger_alias_v4` saved baseline before starting further experiments. Its model, normalization, 50-feature schema, wide blocking configuration, and threshold `0.71` are frozen under `output/submission_01/`. Full inference, both validators with ID checks, versioned outputs, and an upload report are automated by `scripts/create_submission.py`. See [Submission 01](SUBMISSION_01.md). Resume the improvement priorities below only after this submission is complete, preserving it unchanged.

The main metric is per-S1 macro F₀.₅, including singletons. For nonempty truth:

`F0.5 = 1.25 × TP / (1.25 × TP + FP + 0.25 × FN)`

## 1. Evidence before changing the system

The latest completed run used 20,000 classifier-training entities, 2,000 tuning entities, and 2,000 held-out entities against all 10,320,219 training targets. The name channel was fitted separately on the complete training fold.

| Measurement | Completed holdout result |
|---|---:|
| Macro F₀.₅ | 0.951057 |
| Link precision | 0.984827 |
| Link recall | 0.904598 |
| Retrieval recall | 0.970115 |
| Perfect-matcher ceiling on retained candidates | 0.989379 |
| True links lost in blocking | 208 |
| Retrieved true links rejected by the matcher | 456 |
| Incorrect links accepted | 97 |
| Mean candidates per S1 | 40 |

The current retrieval ceiling is below 0.99 on this sample. A better classifier alone cannot reach 0.99 on these same candidates. Both retrieval and matching need improvement, with matching the immediate priority.

On these same 2,000 held-out entities, the earlier model scores 0.933597. The new model scores 0.951057: a paired gain of 0.017460, with an entity-bootstrap 95% interval of 0.012734–0.022544. This compares the same entities and target pools; it remains local exploratory evidence.

All further error-driven changes use tuning data. The latest tuning diagnostic found 228 blocked true links, 419 retrieved-but-rejected true links, and 100 false predictions. Only 62 of its 519 matcher errors lie within 0.05 of the threshold. Threshold adjustments alone are unlikely to solve the main problem.

Among rejected tuning matches, 368 have no learned alias evidence, 165 have high address-token similarity, and 83 have missing candidate addresses. These groups overlap; they identify investigation areas, not established causes or automatic matching rules.

## 2. Execution priorities

| Priority | Experiment | Purpose | Selection criterion |
|---|---|---|---|
| P0 | Implement targeted address/numeric and name-alignment evidence | Address the remaining matcher errors after larger-data and capacity checks | Version features, ablate changes on tuning data, then use a frozen fresh audit |
| P1 | Inspect tuning errors and targeted hard negatives | Separate similar-name branches, shared-address businesses, missing addresses, and unseen name variants | Improve tuning macro F₀.₅ without a material precision or singleton regression |
| P1 | Address and numeric evidence improvements | Recover matches with formatting changes while detecting real contradictions | Version features; evaluate each addition separately, then together |
| P1 | Out-of-fold alias features and alias robustness | Reduce reliance on name evidence learned from the same training entity | Compare on unseen-name groups and regular tuning data; never use validation labels in alias fitting |
| P2 | Retrieval-loss analysis before and after truncation | Recover missing true links with bounded work | Move the oracle macro F₀.₅ ceiling toward ≥0.995 on representative development data; this is a target, not an achieved score |
| P2 | K=5/10/20 and adaptive candidate budgets | Reduce candidate count without sacrificing final quality | Choose on development data; export exactly the candidates actually scored |
| P2 | Bounded retrieval throughput | Make full inference practical for 1,732,544 test references | Benchmark end-to-end scoring on a representative sample, with matching-quality checks |
| P3 | Small model/decision comparison | Capture complementary errors after the evidence features are stronger | Retain an ensemble or extra decision layer only for repeatable improvement over the best single model |

## 3. Concrete modeling work

### Expand useful training coverage

The 20,000-entity run is complete and improves both precision and recall on the identical evaluation cohort. Use this learning-curve evidence and the capacity experiments to decide whether 100,000 entities are justified. Increasing sample size is an experiment, not a guaranteed improvement.

Keep all observed positives among retrieved pairs. Build difficult negatives from similar names at different addresses, different businesses at the same address, numeric conflicts, and near-threshold wrong matches. Mine additional examples using predictions for training-fold entities or out-of-fold training scores. Do not turn tuning or audit mistakes into training examples and then report those same entities as independent validation.

### Improve address and name evidence

Add explicit evidence for available-versus-missing numbers, number-set conflicts, and ordered numeric fragments. Treat the first number as a proxy rather than assuming it is always a street number. Test punctuation-separated numeric variants without converting every number into an unqualified match.

Compare distinctive address components and align IDF lookups with the actual accent/token representation. Verify Unicode and leading-zero behavior. Strong agreement on a common city or street should not carry the same weight as agreement on a distinctive address component.

Test token alignment and acronym evidence for name variants that the learned alias channel does not recognize. Preserve raw and normalized views. No name-only rule should automatically merge branches of a common chain.

### Make supervised name features generalize

The current name channel excludes validation entities, so validation labels do not enter fitting. During classifier training, however, a training entity can benefit from its own contribution to that channel. Compare out-of-fold training aliases against the current approach to test whether this creates a generalization shortcut.

Also test robustness when alias evidence is absent. Alias dropout is a possible training experiment; it is not a substitute for a properly isolated evaluation. Retrieval and feature generation must use the documented fitted assets consistently.

### Tune the decision system after stronger fitting

Retain global thresholding as the baseline. Extend threshold search where score distributions justify it. Compare separate S2/S3 thresholds only if their development curves differ consistently.

Test cross-reference competition for targets that appear plausible for multiple S1 entities. The training audit found exclusive target ownership, but a greedy winner rule can still choose the wrong business. Use uncertainty and validation evidence; avoid forced assignments and transitive merges from weak links.

Fit any learned singleton gate or score calibrator using out-of-fold training predictions. Never force one match per source. Country names remain open strings; France must not receive an invented country-specific threshold.

## 4. Retrieval and runtime strategy

Keep the wide configuration as the quality baseline. Diagnose whether a missed link was never retrieved or was removed by reranking/top-K. Inspect development examples from each category and add the smallest useful retrieval path.

The compound-index experiments reduced candidates but lost too many true links. The latest unranked bounded variant processed 100 tuning references in about 5.1 seconds, but its link recall was only 0.8911. It is not selected for production. Probe timings are not a controlled full-inference benchmark.

K=5 per source reduced the pilot candidate count by 75% and gave comparable exploratory matching quality. K=20 still had the best tuning score and a higher retrieval ceiling. Verify this tradeoff on the larger development set rather than selecting K from the highest observed holdout score.

A possible adaptive strategy uses a small budget for distinct, well-supported records and expands ambiguous cases. Its expansion rules must be determined without validation labels at inference. Report pre-cap retrieval volume as well as the final scored candidate count.

## 5. Evaluation rules

1. Preserve S1-level entity isolation and all singleton rows. Targets follow their labeled owners.
2. Use tuning data for experiments. Treat previously inspected pilot holdouts as exploratory.
3. Reserve a fresh audit population from previously unused holdout entities before final model selection; exclude every entity already inspected. Prefer at least 5,000 entities, with enough singletons and both training countries represented.
4. Compare models on identical entities and target pools. Use paired entity-bootstrap score differences to assess whether small gains are stable.
5. Run unseen-name-family and country-transfer checks. Refit supervised preprocessing inside each split. These tests provide stress evidence; they cannot establish French accuracy.
6. Record macro F₀.₅, precision/recall, singleton errors, candidate ceiling, mean/p95 candidates, runtime, and memory. Do not report micro precision or an oracle score as the main achieved score.
7. Make one controlled change at a time where possible. Log rejected experiments as well as gains.

## 6. Work already started

- The 20,000-training-entity run is complete at 0.951057 held-out macro F₀.₅.
- Two capacity experiments are complete. Longer boosting/lower regularization reached 0.955451 tuning macro F₀.₅; more leaves reached 0.955490, versus the 0.954912 baseline. Both reduced precision. These gains are too small to justify promotion without stronger evidence.
- Tuning error-budget analysis is implemented and has run on the completed pilot.
- Two feature-removal experiments are complete. Removing the blocking score raised tuning macro F₀.₅ from 0.937929 to 0.939181, but reduced precision. Removing blocking and direct alias scores reached 0.938652 with a larger precision decline. These small results do not justify promotion yet.
- Complete test S2/S3 indexes are built.
- Both submission validators are available in the self-contained code package.

Reproduce current diagnostics from the repository root:

```bash
python scripts/analyze_tuning_errors.py \
  --artifact-dir models/larger_alias_v4 \
  --summary-output reports/experiments/larger_tuning_error_budget.json

python scripts/run_feature_ablations.py \
  --artifact-dir models/pilot_alias_v4 \
  --output-dir models/feature_ablations_v1 --threads 2
```

Use a fresh output directory for a completed experiment. Pair examples and trained experimental models stay under ignored `models/`; aggregate reports can be reviewed in Git.

## 7. Submission gate

Freeze the chosen model, feature schema, alias asset, retrieval settings, and thresholds. Complete fresh audit evaluation and a representative full-pipeline throughput test. Generate both TSVs for every official test S1, including France, then run the strict and organizer validators with ID checks enabled.

The final candidate file must equal the actual matching-model inference set. Preserve both output files and exact reproduction commands in the submission archive. Fill methodology results from the selected final run. No leaderboard upload or score is claimed before it actually occurs.

All identity evidence must come from the supplied data. Any added model must independently meet the MIT/Apache-2.0, ≤8B-parameter, offline, and permitted-training-data requirements. Complexity is accepted only when it improves measured quality or runtime.
