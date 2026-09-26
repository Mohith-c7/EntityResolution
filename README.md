# Amazon ML Challenge 2026 — Business Entity Resolution

Offline business matching using the seven organizer TSVs, bounded disk indexes, a training-only name channel, pair features, and LightGBM.

## Latest completed result

The verified `models/features_v3_audit01` model achieved **0.960616 macro F₀.₅** on **5,000 unused holdout entities**. On the identical entities, the frozen v2 model scores 0.948512. Precision improves from 0.981779 to 0.988304, recall from 0.901517 to 0.916518, and singleton false positives fall from 27 to 19. Paired macro gain: **0.012104**, with bootstrap 95% interval **[0.009178, 0.015012]**. These are local scores; there is no leaderboard score yet.

Version 3 uses 65 features, 20,000 classifier-training entities, and 10,000 separate tuning entities. Threshold 0.76 and model settings were frozen before the fresh audit. All acceptance gates passed. The old 0.951057 score belongs to a different 2,000-reference cohort.

Retrieval remains the release bottleneck. A paired 600-reference benchmark measured 17.74 references/second for baseline and 19.99 with memory mapping. Removing numeric/location retrieval reached 21.92 but lowered matching quality, so it was not adopted. The fresh candidate oracle is 0.989322 with 40 candidates per reference; better ranking is needed to make 0.99 attainable on this cohort.

Submission 01 remains stopped at 37,000 completed references, with its original frozen model and checkpoints preserved. No complete test output has been produced. See [the review implementation plan](docs/REVIEW_IMPLEMENTATION_PLAN.md) and [the aggregate audit report](reports/experiments/features_v3_fresh_audit.json).

## Earlier development results

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
