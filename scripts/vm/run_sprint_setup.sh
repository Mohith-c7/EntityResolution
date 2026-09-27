#!/bin/bash
# Native extension, train and test indexes, then the development parity replay.
set -euo pipefail
cd /mnt/er/entity
PY=/mnt/er/venv/bin/python
L=/mnt/er/logs
stamp() { echo "$(date -u +%H:%M:%S) $*"; }
mkdir -p models/runtime output/submission_03/code/business_entity_resolution/native
$PY scripts/build_postings_native.py --output models/runtime/erpostings.dylib
cp models/runtime/erpostings.dylib output/submission_03/code/business_entity_resolution/native/erpostings.dylib
stamp "native extension built"
export PYTHONPATH=/mnt/er/entity/code/business_entity_resolution
for split in train test; do
  for n in 2 3; do
    $PY -c "from src.blocking.disk_index import build_disk_index; print(build_disk_index('dataset/$split/${split}_source$n.tsv', 'models/index_${split}_S$n.sqlite', 'S$n'))" \
      > $L/index_${split}_S$n.log 2>&1 &
  done
done
wait
stamp "indexes built"
$PY scripts/run_frozen_pipeline.py replay --references 250 > $L/replay.log 2>&1
stamp "REPLAY_DONE"
