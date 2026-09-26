# Submission 01 — Frozen Baseline

## Selection

Use the saved `models/larger_alias_v4` model for the first complete submission. Its local held-out macro F0.5 is **0.9510571967** on 2,000 entities. Its official leaderboard score is unknown until the portal scores the uploaded file.

Freeze the model, training-fitted name channel, normalization, ordered 50-feature schema, wide retrieval settings, and threshold `0.71`. These frozen assets remain unchanged. The run was stopped at user request with 37,000 completed references; there are no upload-ready TSVs. Feature and runtime experiments now run separately. Resume this frozen job only after an explicit decision.

## Compatibility evidence

The nine critical inference modules match the SHA-256 hashes recorded during training. A replay of 120 saved tuning pairs reproduced the feature vectors and model probabilities exactly. Complete test S2/S3 indexes match their source-file fingerprints.

The frozen code and model copies are under `output/submission_01/`, alongside `compatibility_report.json`, `frozen_assets_sha256.json`, and input/configuration metadata. Generated outputs and fitted assets are ignored by Git.

## Run and resume

```bash
python scripts/create_submission.py \
  --submission-dir output/submission_01 \
  --test-dir dataset/test --index-dir models \
  --workers 12 --batch-size 100
```

Run one process per submission directory. A completed chunk commits both output fragments and their checksums before its completion marker. Resume verifies input order, assets, configuration, and completed fragments. Completed work survives an interrupted process.

Read `output/submission_01/progress.json` for current counts and runtime estimates. These counts are partial until the run finishes; checkpoint fragments must not be uploaded.

## Finalization

The runner performs the following steps automatically:

1. Cover all 1,732,544 official test Source 1 entities, including France, in original input order.
2. Export exactly the candidates scored by the frozen matching model, with matches as subsets and empty predictions preserved.
3. Assemble `output/submission_01/matching_results.tsv` and `candidate_pairs.tsv`.
4. Run the strict validator with ID membership and subset checks.
5. Run the unmodified official validator with `--check-ids` and both output files.
6. Publish byte-identical copies at `output/matching_results.tsv` and `output/candidate_pairs.tsv` after both validators pass.
7. Save prediction counts, empty-prediction rate, candidate histogram/mean/p95, country counts, output hashes, environment, and validation results in `submission_report.json`.

The upload gate is `ready_for_upload: true` in `submission_report.json`. The rate of empty predictions is a predicted no-match rate; true singleton prevalence in the unlabeled test set is unknown.

## Submission 02

Preserve Submission 01 unchanged. Continue the score-improvement plan separately while this run remains stopped: diagnose missed candidates, false merges, and rejected true matches; select changes using tuning data; freeze a configuration and evaluate it on a fresh S1-level holdout. Country-transfer checks provide stress evidence, while French accuracy remains unmeasured locally.

All identity evidence and training examples must come from the supplied challenge data. Each additional model must independently satisfy the organizer's model-license, parameter-count, offline-inference, and training-data requirements.
