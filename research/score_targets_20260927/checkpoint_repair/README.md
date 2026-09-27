# Submission checkpoint repair

The resume failure is metadata-only. `score_chunk` correctly scores the current 250 Source 1 rows, but it reuses `ids` for S3 candidates before writing `input_ids_sha256`. A resumed process compares that S3-candidate hash with the current Source 1 slice and rejects every populated marker. The test input SHA still matches `universe_counts_test.sqlite`; all 1,630 saved parquet files matched their recorded hashes; sampled legacy hashes matched the observed sorted S3 candidate IDs.

`run_frozen_pipeline.patch` renames the reference and candidate variables and does not change retrieval, features, model inference, probabilities, or decisions. Because the runner is pinned in `frozen.json`, the old descriptor must remain untouched. `prepare_repair_frozen.py` accepts only the exact expected patch, verifies that the original runner matches the old descriptor and that the accepted audit report belongs to that descriptor, then proposes a new descriptor whose runner hash pins the patched file. It records both descriptor/report hashes and the metadata-only parity proof.

Review and dry-run locally:

```bash
python3 -m unittest discover -s research/score_targets_20260927/checkpoint_repair -p 'test_*.py'
patch --dry-run -p1 < research/score_targets_20260927/checkpoint_repair/run_frozen_pipeline.patch
python3 research/score_targets_20260927/checkpoint_repair/migrate_checkpoint_markers.py \
  --source1 dataset/test/test_source1.tsv \
  --universe-counts output/submission_04/universe_counts_test.sqlite \
  --chunks output/submission_04/chunks
```

After approval, first preserve the original runner outside the source tree, apply the patch, and dry-run the new descriptor. Neither helper writes without `--apply`:

```bash
mkdir /mnt/er/repair_backup_20260927_e70007
cp scripts/run_frozen_pipeline.py /mnt/er/repair_backup_20260927_e70007/run_frozen_pipeline.original.py
patch -p1 < research/score_targets_20260927/checkpoint_repair/run_frozen_pipeline.patch
python3 research/score_targets_20260927/checkpoint_repair/prepare_repair_frozen.py \
  --original-descriptor models/frozen_v4/frozen.json \
  --original-runner /mnt/er/repair_backup_20260927_e70007/run_frozen_pipeline.original.py \
  --patched-runner scripts/run_frozen_pipeline.py \
  --audit-report models/audit_v2/run/report.json \
  --output models/frozen_v4_checkpoint_repair/frozen.json
```

Only after both dry-runs pass, apply into new backup/output paths:

```bash
python3 research/score_targets_20260927/checkpoint_repair/migrate_checkpoint_markers.py \
  --source1 dataset/test/test_source1.tsv \
  --universe-counts output/submission_04/universe_counts_test.sqlite \
  --chunks output/submission_04/chunks \
  --apply --backup-dir output/submission_04/checkpoint_markers_before_repair
python3 research/score_targets_20260927/checkpoint_repair/prepare_repair_frozen.py \
  --original-descriptor models/frozen_v4/frozen.json \
  --original-runner /mnt/er/repair_backup_20260927_e70007/run_frozen_pipeline.original.py \
  --patched-runner scripts/run_frozen_pipeline.py \
  --audit-report models/audit_v2/run/report.json \
  --output models/frozen_v4_checkpoint_repair/frozen.json --apply
```

Resume with the same dataset/output and the new descriptor: `python scripts/run_frozen_pipeline.py test --frozen models/frozen_v4_checkpoint_repair/frozen.json --workers 62`. The original `models/frozen_v4/frozen.json`, accepted audit report, marker backups, and every parquet remain preserved. Chunks 1629 and 1630 are absent and will be computed normally. Final merge still verifies each parquet hash before writing and validating submission files.
