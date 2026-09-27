#!/bin/bash
# Stacked second stage at scale: out-of-fold first-stage scores for 300k training
# references, their context and competition features, then four second-stage fits.
set -euo pipefail
cd /mnt/er/entity
PY=/mnt/er/venv/bin/python
L=/mnt/er/logs
stamp() { echo "$(date -u +%H:%M:%S) $*"; }
size() { stat -c %s "$1" 2>/dev/null || echo 0; }
wait_size() { while [ "$(size "$1")" != "$2" ]; do sleep 10; done; }

verify_cache() {
  $PY - "$1" <<'EOF'
import hashlib, json, sys
from pathlib import Path
cache = Path(sys.argv[1])
progress = json.loads((cache / "progress.json").read_text())
for n in range(progress["chunks"]):
    marker = json.loads((cache / f"{n:06d}.json").read_text())
    if hashlib.sha256((cache / f"{n:06d}.parquet").read_bytes()).hexdigest() != marker["parquet_sha256"]:
        sys.exit(1)
EOF
}

M=models/next_round/extended_model
wait_size $M/train_features.npy 2268575008
wait_size $M/train_groups.npy 34901280
wait_size $M/train_labels.npy 8725416
wait_size $M/train_weights.npy 34901280
stamp "training arrays ready"

FOLDS=()
for k in 0 1 2 3 4; do
  $PY scripts/train_oof_first_stage.py --fold $k --threads 12 > $L/oof_fold_$k.log 2>&1 &
  FOLDS+=($!)
done
stamp "out-of-fold models started"

until verify_cache models/scale_v1_run/cache_train 2>/dev/null; do sleep 15; done
stamp "training cache verified"
$PY scripts/build_competition_features.py --cache models/scale_v1_run/cache_train --reference-split train \
  --output models/next_round/competition_train > $L/competition_train.log 2>&1 &
COMP=$!

for pid in "${FOLDS[@]}"; do wait $pid; done
$PY scripts/train_oof_first_stage.py --merge > $L/oof_merge.log 2>&1
stamp "out-of-fold scores merged"

$PY scripts/build_context_features.py --split train --cache models/scale_v1_run/cache_train \
  --first-stage-model $M --probabilities models/next_round/oof_first_stage/oof_probabilities_train.npy \
  --output models/next_round/context_train_oof --workers 56 > $L/context_train.log 2>&1
stamp "training context features built"
wait $COMP
stamp "training competition features built"

R=reports/experiments/round3
mkdir -p $R
FITS=()
for spec in "a31:31:0.05:100:" "b31dev:31:0.05:100:--add-development-training" "c127:127:0.05:200:" "d127dev:127:0.05:200:--add-development-training"; do
  IFS=: read -r name leaves lr child extra <<< "$spec"
  $PY scripts/train_stacked_context.py --output models/next_round/stacked_$name --report-output $R/stacked_$name.json \
    --leaves $leaves --learning-rate $lr --min-child $child --trees 3000 --threads 16 $extra > $L/stacked_$name.log 2>&1 &
  FITS+=($!)
done
stamp "second-stage fits started"
for pid in "${FITS[@]}"; do wait $pid || stamp "a fit failed: $pid"; done
stamp "ALL_DONE"
