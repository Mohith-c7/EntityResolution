# EntityResolution — Next Steps

Date: 26 September 2026
Main metric: macro F₀.₅, averaged over every Source 1 entity.

## Current state

All seven official TSVs are audited. Feature version 3 now passes a fresh 5,000-entity audit at **0.960616 local macro F₀.₅**, compared with 0.948512 for the frozen baseline on those same entities. Precision and recall improve and singleton errors fall. The 10,000-entity tuning set selected threshold 0.76. There is no leaderboard score. Submission 01 remains stopped at 37,000 checkpointed references.

Retrieval is now the immediate release priority: batch key joins, improved ranking before top-K, and a measured candidate budget. Memory mapping alone improves the short benchmark by about 13%; removing numeric/location queries lowers matching quality and is not adopted. See [the review implementation plan](REVIEW_IMPLEMENTATION_PLAN.md) for the measured results and team deliverables.

## Ownership and immediate work

| Member | Next deliverable |
|---|---|
| Harsha | Larger classifier sample, hard-negative analysis, retrieval/decision experiments, candidate-budget curve, full-test inference |
| Mohith | Generic loader string preservation, executable cross-file integrity checks, measured EDA regeneration, feature/provenance review, package integration |
| Sanhitha | Review merged preprocessing regressions and inspect recurring normalization errors on supplied data |
| Sahasra | Independent metric checks, singleton/output QA, experiment scorecards, methodology verification |

## Release order

1. Implement batch retrieval and learned reranking against the verified version 3 matcher; keep the completed audit unused for further selection.
2. Select the smallest candidate budget that preserves matching quality and retrieval coverage.
3. Freeze model, feature version, alias asset, retrieval settings, and threshold.
4. Evaluate a larger fresh audit population and unseen-name/transfer scenarios.
5. Generate complete `matching_results.tsv` and the exact scored `candidate_pairs.tsv`.
6. Run the strict and organizer validators, including the organizer's `--check-ids`.
7. Fill the methodology template using the final measured system and package the reproducible code.
8. Upload the complete matching file to the portal; retain the identical file in the final archive.

Do not submit limited smoke outputs, report an oracle ceiling as model accuracy, or assume a local score guarantees a leaderboard rank.

See `code/business_entity_resolution/README.md` for runnable commands and `reports/experiments/PILOT_RESULTS.md` for measured evidence.
