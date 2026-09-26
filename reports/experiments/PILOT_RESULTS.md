# Development experiments — 26 September 2026

## Evaluation setting

Whole-file deterministic seed-42 entity folds: 1,544,993 train, 330,816 tune, 331,012 holdout. Classifier samples: 2,000 / 500 / 500. Every reference is retrieved against all 5,034,616 S2 and 5,285,603 S3 training records. No target limit is used.

The train-only name channel is fitted separately on 1,544,990 nonempty training-fold S1 names and 5,348,430 matched targets. It contains 653,487 confident aliases and 46,580 ambiguous alias strings. No tune/holdout-owned target contributes to this channel or classifier training. Label-free retrieval frequencies use the full target corpus.

These pilot results are exploratory. Model selection and repeated diagnostic inspection can bias small holdout estimates. They are not French accuracy estimates or leaderboard scores. Freeze settings and evaluate a larger fresh population before a final claim.

## Matching results

| Measurement | Initial pilot | Improved pilot |
|---|---:|---:|
| Macro F₀.₅, holdout | 0.8462164 | 0.9267946 |
| Micro precision | 0.9681818 | 0.9758643 |
| Micro recall | 0.7447552 | 0.8717949 |
| Retrieval link recall | 0.8071096 | 0.9662005 |
| Oracle macro F₀.₅ | 0.8900307 | 0.9871680 |
| Mean candidates per S1 | 40 | 40 |
| Threshold selected on tune | 0.69 | 0.74 |

Improved holdout macro F₀.₅: US 0.9387401 (312 entities), India 0.9069702 (188 entities). Entity-bootstrap 95% interval: 0.9107670–0.9412080. Of 17 holdout singletons, two received a false prediction; the sample is too small for a stable singleton estimate.

The improved classifier scored 58,377 training pairs after excluding 21,623 retrieved pairs belonging to held-out counterpart identities. Tune and holdout each contained 20,000 pairs. Candidate retrieval took approximately 209 / 92 / 90 seconds for train / tune / holdout with four workers in that run.

## Changes and limitations

The improved run jointly changed retrieval weights, numeric/location queries, train-only alias retrieval, and the 36-to-50 feature schema. Its gain is evidence for the combined implementation; it does not isolate each component's contribution.

The main quality gap is between the 0.9872 retrieval ceiling and 0.9268 achieved matching score. Increase classifier training coverage and inspect tuning errors before adding larger model families. Blocking still misses links and retains 40 candidates per S1, so candidate-budget optimization remains necessary.

Full-test inference has not run. No portal upload or completed competition package is claimed.

## Retrieval experiments

Fixed tuning-only probes use the first 100 deterministic tuning references, against the complete target universe.

- Wide v4 retrieval with the current name channel: link recall 0.962751, oracle macro F₀.₅ 0.989922, 40 candidates per S1.
- Initial compound retrieval: link recall 0.914040, oracle 0.967183, mean 34.66 candidates.
- More selective compound retrieval: link recall 0.896848, oracle 0.962941, mean 31.29 candidates.

The compound alternatives are rejected for current model selection: smaller candidate counts did not justify the recall loss. Timing probes overlapped and are not controlled throughput comparisons. Frequency materialization preserves corpus DF values; measured accuracy must remain unchanged when only this lookup mechanism changes.

Oracle scores assume perfect matching and must never be presented as actual model scores.

## Next experiments

1. Increase classifier-training entities while retaining entity isolation and realistic hard negatives.
2. Compare smaller candidate budgets using frozen tuning references.
3. Add source-specific calibration or entity context only when measured gains justify it.
4. Test unseen-name and country-transfer behavior; do not infer French performance from US/India.
5. Freeze the chosen system, measure full inference throughput, export every official test S1, run both validators, and fill final methodology counts.

Detailed machine-readable pilot reports are stored beside this document. Local model/pair artifacts remain under ignored `models/`.

## Candidate-budget pilot

| K per source | Mean candidates | Tune macro F₀.₅ | Holdout macro F₀.₅ | Holdout retrieval ceiling |
|---|---:|---:|---:|---:|
| 1 | 2 | 0.7694 | 0.7891 | 0.8097 |
| 2 | 4 | 0.8875 | 0.8895 | 0.9264 |
| 3 | 6 | 0.9185 | 0.9139 | 0.9581 |
| 5 | 10 | 0.9303 | 0.9297 | 0.9780 |
| 10 | 20 | 0.9346 | 0.9273 | 0.9830 |
| 20 | 40 | 0.9379 | 0.9268 | 0.9872 |

K=5 is a smaller candidate-budget challenger, with comparable pilot matching quality. K=20 remains preferred by the tuning score. Selecting K=5 solely because its holdout score is highest would use the holdout for tuning; confirm budget choice on the larger development population. Pair features in wide retrieval are independent of final K, so existing predictions permit this exploratory comparison. Final inference must score and export only the configured retained set.

## Larger training run

`larger_alias_v4` uses 20,000 training, 2,000 tuning, and 2,000 held-out S1 entities. It achieved 0.9510572 holdout macro F₀.₅, 0.9848272 precision, and 0.9045977 recall at threshold 0.71. Retrieval recall is 0.9701149; the retained-candidate ceiling is 0.9893790.

On the identical 2,000-entity cohort, the earlier model scores 0.9335971. The paired gain is 0.0174601, with bootstrap 95% interval 0.0127344–0.0225441. Singleton false positives fell from 8 to 4 among 81 singleton entities. See `larger_model_comparison.json` for the complete comparison.

The larger tuning set still has 419 retrieved true links rejected by the matcher, 100 false predictions, and 228 links lost in blocking. Capacity/regularization comparisons used cached pairs and tuning data for selection. Longer boosting/lower regularization reached 0.955451 tuning macro F₀.₅; more leaves reached 0.955490, versus baseline 0.954912. Both reduced precision; neither is promoted.
