# Amazon ML Challenge 2026 — Business Entity Resolution

Offline business matching using the seven organizer TSVs, bounded disk indexes, a training-only name channel, pair features, and LightGBM.

## Measured development results

The complete training target pool contains 5,034,616 S2 records and 5,285,603 S3 records. These pilots used 2,000 classifier-training S1 entities, 500 threshold-tuning entities, and 500 held-out entities.

| Pilot | Held-out macro F₀.₅ | Link precision | Link recall | Blocking recall |
|---|---:|---:|---:|---:|
| Initial lexical baseline | 0.8462 | 0.9682 | 0.7448 | 0.8071 |
| Name channel, address-led retrieval, 50 features | 0.9268 | 0.9759 | 0.8718 | 0.9662 |

These are exploratory local scores, not leaderboard results. The improved pilot's bootstrap 95% interval is 0.9108–0.9412. Its name channel is fitted on all eligible training-fold names and counterparts, separately from the classifier sample. Multiple components changed between pilots; this is not an isolated feature ablation. France is test-only, so these results do not establish French accuracy.

All seven files passed a full schema, ID, ground-truth, ownership, and train/test-ID overlap audit. Sanhitha's normalization fixes at `17c9532` are integrated, with blank-field regression fixes. Full official-test predictions and a submission archive are still pending.

## Run and review

See [the runnable package README](code/business_entity_resolution/README.md) for exact setup, training, inference, and validation commands.

- [Team execution plan](docs/TEAM_PLAN.md)
- [Full download audit](reports/eda/local_download_audit.json)
- [Experiment record](reports/experiments/PILOT_RESULTS.md)
- [Mohith issue review](reports/reviews/Mohith_issue_2_review.md)

Macro F₀.₅ is averaged over every S1 entity, including singletons. The export contains precisely the candidates scored by the matcher, and matches are subsets. No external identity lookup, geocoding, or hosted model calls are used. Countries remain unrestricted strings.

The official organizer validator is preserved at `utils/organizer_validate_submission.py`. A stricter streaming validator is available at `utils/validate_submission.py`. Run both before uploading, with `--check-ids` on the official helper. Candidate size and scalability also matter in the organizer's final ranking review.
