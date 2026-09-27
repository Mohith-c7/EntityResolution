#!/bin/bash
# After a clean parity replay: submission_03 baseline and the one-time audit, alongside
# test scoring. The submission is published only if the audit accepts it.
set -uo pipefail
cd /mnt/er/entity
PY=/mnt/er/venv/bin/python
L=/mnt/er/logs
stamp() { echo "$(date -u +%H:%M:%S) $*"; }
(
  $PY scripts/audit_submission03_baseline.py --workers 16 > $L/audit_baseline.log 2>&1 && stamp "baseline scored" &&
  $PY scripts/run_frozen_pipeline.py audit --workers 24 > $L/audit.log 2>&1 && stamp "AUDIT_DONE" || stamp "AUDIT_FAILED"
) &
$PY scripts/run_frozen_pipeline.py test --workers 48 > $L/test.log 2>&1 && stamp "TEST_DONE" || stamp "TEST_FAILED"
wait
stamp "SPRINT_MAIN_DONE"
