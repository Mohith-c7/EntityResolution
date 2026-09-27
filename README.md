# Amazon ML Challenge 2026 — Business Entity Resolution

Offline business matching using the seven organizer TSVs, bounded disk indexes, a training-only name channel, pair features, and LightGBM.

## Final results and project status

The best completed submission, **Submission 05**, received a team-reported **0.970 portal macro F0.5**, following Submission 04 at **0.968** and Submission 03 at **0.9399**. It covered all **1,732,544** test references, including France, and both strict and official validators passed with ID checking enabled.

The strongest separately audited local candidate reached **0.9811416633** on 10,000 held-out entities. Its Submission 05 comparator scored **0.9803742791 on the same cohort**; the paired gain was **0.0007673842**, with 95% interval **[0.0002402509, 0.0013411254]**. This later hybrid did **not** become a completed submission. Local audits did not measure French accuracy, and the qualification target of 0.989279 was not achieved.

All cloud runs were stopped and the Azure experiment resources were deleted at the user's request. No training or submission generation remains active. The final address-dropout training and two development score sets completed locally, but its calibration was not reviewed or promoted. Submission 06 export was terminated before completion. Submission 05 remains the latest completed release.

Read the [project retrospective](docs/PROJECT_RETROSPECTIVE.md) for the full development experience, improvements, failures, operational lessons, and measured outcomes. The [final handover](docs/ER_HANDOVER_2026-09-27.md) records artifact locations and hashes. Large datasets, checkpoints, and output caches are intentionally outside Git; a repository clone alone does not contain the complete release package.

## Pipeline

The system combined bounded posting-list retrieval, bridge candidates, LightGBM pair and context models, competing-owner evidence, and sparse offline multilingual neural matching. Every completed submission retained the exact candidates actually scored. Development experiments remained separate from frozen release artifacts.

Historical runtime measurements and stopped Submission 01/02 attempts are documented in [the runtime plan](docs/RUNTIME_REDESIGN.md). Earlier 0.951057, 0.960616, 0.965888, and 0.978547 local results came from different cohorts and pipelines; they should not be read as one directly comparable leaderboard series.

## Earlier development results

The complete training target pool contains 5,034,616 S2 records and 5,285,603 S3 records. These pilots used 2,000 classifier-training S1 entities, 500 threshold-tuning entities, and 500 held-out entities.

| Pilot | Held-out macro F₀.₅ | Link precision | Link recall | Blocking recall |
|---|---:|---:|---:|---:|
| Initial lexical baseline | 0.8462 | 0.9682 | 0.7448 | 0.8071 |
| Name channel, address-led retrieval, 50 features | 0.9268 | 0.9759 | 0.8718 | 0.9662 |

These are exploratory local scores, not leaderboard results. The improved pilot's bootstrap 95% interval is 0.9108–0.9412. Its name channel is fitted on all eligible training-fold names and counterparts, separately from the classifier sample. Multiple components changed between pilots; this is not an isolated feature ablation. France is test-only, so these results do not establish French accuracy.

All seven files passed a full schema, ID, ground-truth, ownership, and train/test-ID overlap audit. Sanhitha's normalization fixes at `17c9532` are integrated, with blank-field regression fixes. Submission 05 predictions and its versioned archive were completed; subsequent experimental exports were not completed.

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
