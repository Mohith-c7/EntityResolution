# EntityResolution — Next Steps

Date: 26 September 2026
Main metric: macro F₀.₅, averaged over every Source 1 entity.

## Current state

All seven official TSVs are downloaded and audited. Sanhitha's Unicode, null-handling, and preprocessing fixes are integrated. The latest full-target run reached 0.9511 local macro F₀.₅ on 2,000 held-out entities; this is not a leaderboard score. The runnable pipeline includes disk retrieval, 36/50 pair features, train-only name-channel fitting, LightGBM, threshold tuning, inference, and strict output validation.

## Ownership and immediate work

| Member | Next deliverable |
|---|---|
| Harsha | Larger classifier sample, hard-negative analysis, retrieval/decision experiments, candidate-budget curve, full-test inference |
| Mohith | Generic loader string preservation, executable cross-file integrity checks, measured EDA regeneration, feature/provenance review, package integration |
| Sanhitha | Review merged preprocessing regressions and inspect recurring normalization errors on supplied data |
| Sahasra | Independent metric checks, singleton/output QA, experiment scorecards, methodology verification |

## Release order

1. Implement targeted address/numeric and name-alignment features based on the remaining tuning errors.
2. Select the smallest candidate budget that preserves matching quality and retrieval coverage.
3. Freeze model, feature version, alias asset, retrieval settings, and threshold.
4. Evaluate a larger fresh audit population and unseen-name/transfer scenarios.
5. Generate complete `matching_results.tsv` and the exact scored `candidate_pairs.tsv`.
6. Run the strict and organizer validators, including the organizer's `--check-ids`.
7. Fill the methodology template using the final measured system and package the reproducible code.
8. Upload the complete matching file to the portal; retain the identical file in the final archive.

Do not submit limited smoke outputs, report an oracle ceiling as model accuracy, or assume a local score guarantees a leaderboard rank.

See `code/business_entity_resolution/README.md` for runnable commands and `reports/experiments/PILOT_RESULTS.md` for measured evidence.
