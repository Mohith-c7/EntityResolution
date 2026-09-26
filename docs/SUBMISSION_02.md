# Submission 02 — Feature Version 3

The full official test run uses `models/features_v3_audit01`, feature schema `pairwise-v3-65`, and threshold `0.76`. Its local fresh-audit macro F0.5 is **0.9606164625**. No leaderboard score is claimed.

The model, aliases, audited inference source and runner are frozen under `output/submission_02/`. The earlier `output/submission_01/` checkpoints remain separate and stopped. Predictions from different models are never combined.

## Generate or resume

From the repository root, using the installed Python environment:

```bash
/tmp/entity-review-venv/bin/python output/submission_02/create_submission.py \
  --submission-dir output/submission_02 \
  --test-dir dataset/test --index-dir models \
  --workers 8 --batch-size 100 --mmap-bytes 2147418112
```

An exclusive run lock prevents a second active process. Resume with identical batch size and memory-mapping settings. Completed chunks are checked against input IDs and SHA-256 file hashes before reuse. The live job was launched detached with idle-sleep prevention; closing a terminal does not intentionally stop it. Explicit sleep, shutdown or reboot can interrupt computation; committed checkpoints permit resumption.

`progress.json` and `inference.log` report actual progress. `ready_for_upload` stays false until every reference is processed, both output files are assembled, the strict validator passes, and the official validator passes with `--check-ids`.

## Completed outputs

After both validators pass, byte-identical copies are published to:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

Versioned copies remain under `output/submission_02/`. The generated `SUBMISSION_REPORT.md` contains the upload link, prediction counts, empty-prediction rate, candidate counts and country summaries. `submission_report.json` records validation results, hashes and the model/configuration used. A partial checkpoint file is not a submission.

Every test Source 1 entity is included exactly once, including France and predicted singletons. The candidate export contains exactly the candidates scored by the matcher; final matches are a subset. Memory mapping changes file access only; it does not change retrieval queries, feature definitions, the model or the threshold.

## Prepare another frozen run

Use a new directory. The preparation helper refuses to overwrite an existing submission:

```bash
python scripts/prepare_submission.py \
  --artifact-dir models/features_v3_audit01 \
  --submission-dir output/submission_02
```

This verifies model/alias hashes, the selected threshold, the audited source snapshot and test-index fingerprints, then copies the required assets. Preparation is already complete for Submission 02; do not repeat it on the existing directory.
