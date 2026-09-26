# Amazon ML Challenge 2026 — Business Entity Resolution

Offline business matching using the seven organizer TSVs, bounded disk indexes, a training-only name channel, pair features, and LightGBM.

## Latest completed result

The first completed submission received a **0.9399 portal score**, reported by the team. Its files are preserved under `output/submission_03/`; both strict and official validators passed with ID checking enabled. It covers all 1,732,544 test references, including France.

The larger `models/scale_v1_run/model_300k` matcher achieved **0.965888 local macro F0.5** on a fresh 10,000-entity audit, versus **0.957786** for the previous matcher on the same entities. Precision increased to 0.989011, recall to 0.927543, and singleton false matches fell from 44 to 31. Paired gain: **0.008103**, with 95% interval **[0.006429, 0.009761]**. This model uses 300,000 training references, 65 features and five-fold training aliases. It has not been submitted.

Current development experiments combine learned candidate ranking, one-hop blocking expansion and a second-stage candidate-context model. The conservative selection reaches **0.970635 on 20,000 development entities**, with candidate oracle **0.993451** and 40 candidates per reference. This is not a fresh-audit or portal score. See [the round-two model plan and research](docs/ROUND_2_MODEL_PLAN.md).

**Submission generation is paused by request while working toward 0.99.** Preserve the existing upload and frozen artifacts; do not automatically create another submission from development improvements. The earlier audit sets are consumed and must not be reused for tuning. France still has no labeled evaluation.

Historical v3 results, runtime benchmarks and stopped submission 01/02 checkpoints are documented in [the runtime plan](docs/RUNTIME_REDESIGN.md). The old 0.951057 and 0.960616 results belong to different, earlier holdout cohorts and pipelines.

## Earlier development results

The complete training target pool contains 5,034,616 S2 records and 5,285,603 S3 records. These pilots used 2,000 classifier-training S1 entities, 500 threshold-tuning entities, and 500 held-out entities.

| Pilot | Held-out macro F₀.₅ | Link precision | Link recall | Blocking recall |
|---|---:|---:|---:|---:|
| Initial lexical baseline | 0.8462 | 0.9682 | 0.7448 | 0.8071 |
| Name channel, address-led retrieval, 50 features | 0.9268 | 0.9759 | 0.8718 | 0.9662 |

These are exploratory local scores, not leaderboard results. The improved pilot's bootstrap 95% interval is 0.9108–0.9412. Its name channel is fitted on all eligible training-fold names and counterparts, separately from the classifier sample. Multiple components changed between pilots; this is not an isolated feature ablation. France is test-only, so these results do not establish French accuracy.

All seven files passed a full schema, ID, ground-truth, ownership, and train/test-ID overlap audit. Sanhitha's normalization fixes at `17c9532` are integrated, with blank-field regression fixes. Full official-test predictions are complete; final archive packaging remains separate.

## Run and review

See [the runnable package README](code/business_entity_resolution/README.md) for exact setup, training, inference, and validation commands.

- [Team execution plan](docs/TEAM_PLAN.md)
- [Full download audit](reports/eda/local_download_audit.json)
- [Experiment record](reports/experiments/PILOT_RESULTS.md)
- [Mohith issue review](reports/reviews/Mohith_issue_2_review.md)

Macro F₀.₅ is averaged over every S1 entity, including singletons. The export contains precisely the candidates scored by the matcher, and matches are subsets. No external identity lookup, geocoding, or hosted model calls are used. Countries remain unrestricted strings.

The official organizer validator is preserved at `utils/organizer_validate_submission.py`. A stricter streaming validator is available at `utils/validate_submission.py`. Run both before uploading, with `--check-ids` on the official helper. Candidate size and scalability also matter in the organizer's final ranking review.

## Branch integration

The active model, feature registry, normalization, blocking and inference modules retain
Harsha's implementation. Main's earlier 36-feature baseline is preserved under
`src/legacy/`, with its normalization and tests isolated from the active pipeline.
`run_legacy_pipeline.py` retains the earlier integration scaffold; use the runnable
package README for the active training and inference commands.

Main's data quality checks, EDA updates and grouped split utility remain available.
The grouped split generator is an optional diagnostic; existing model artifacts still
use their recorded hash split and cross-fitted aliases. A merge does not change those
artifacts or their evaluation protocol.
