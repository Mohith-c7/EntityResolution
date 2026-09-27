#!/bin/bash
# Both stages refit on bridge-retrieval candidates. Cache builds resume from their
# checkpoints. Second stage: the configurations that won on the original candidates
# (127 leaves, two seeds) plus 255 leaves, then decision rules, the mixture, losses
# and bridge-aware exclusivity on the same 20k selection references.
set -euo pipefail
cd /mnt/er/entity
PY=/mnt/er/venv/bin/python
L=/mnt/er/logs
R=reports/experiments/round3
B=models/bridge_v1
stamp() { echo "$(date -u +%H:%M:%S) $*"; }
run() { nice -n 5 "$PY" "$@"; }
# /mnt is the VM's temporary disk and is wiped on deallocation; results must reach the OS disk.
keep() {
  mkdir -p ~/results
  rsync -a "$R" $L ~/results/ || true
  rsync -a --exclude='cache_*/' --exclude='context_*/' --exclude='competition_*/' --include='*/' --include='model*.txt' \
    --include='*.json' --include='selection_probabilities.npy' --include='adjusted_*.npy' --exclude='*' $B/ ~/results/bridge_v1/ || true
}
trap keep EXIT
mkdir -p "$B" "$R"

run scripts/build_scale_cache.py --split tune --bridge --workers 60 --idle-workers 60 --output $B/cache_tune >> $L/bridge_cache_tune.log 2>&1
run scripts/build_scale_cache.py --split train --bridge --workers 60 --idle-workers 60 --output $B/cache_train >> $L/bridge_cache_train.log 2>&1
stamp "bridge caches built"

run scripts/build_competition_features.py --cache $B/cache_tune --reference-split tune --output $B/competition_tune > $L/bridge_competition_tune.log 2>&1 &
COMP_TUNE=$!
run scripts/build_competition_features.py --cache $B/cache_train --reference-split train --output $B/competition_train > $L/bridge_competition_train.log 2>&1 &
COMP_TRAIN=$!

run scripts/train_scale_model.py --train-cache $B/cache_train --tune-cache $B/cache_tune --output $B/first_stage --threads 56 --trees 3000 > $L/bridge_first_stage.log 2>&1
stamp "first stage fitted"
FOLDS=()
for k in 0 1 2 3 4; do
  run scripts/train_oof_first_stage.py --model-dir $B/first_stage --output $B/oof --fold $k --threads 12 > $L/bridge_oof_$k.log 2>&1 &
  FOLDS+=($!)
done
for pid in "${FOLDS[@]}"; do wait "$pid"; done
run scripts/train_oof_first_stage.py --model-dir $B/first_stage --output $B/oof --merge > $L/bridge_oof_merge.log 2>&1
stamp "out-of-fold scores merged"

run scripts/build_context_features.py --split train --cache $B/cache_train --first-stage-model $B/first_stage \
  --probabilities $B/oof/oof_probabilities_train.npy --output $B/context_train --workers 56 > $L/bridge_context_train.log 2>&1
run scripts/build_context_features.py --split tune --cache $B/cache_tune --first-stage-model $B/first_stage \
  --output $B/context_tune --workers 56 > $L/bridge_context_tune.log 2>&1
stamp "context features built"
wait $COMP_TUNE; wait $COMP_TRAIN
stamp "competition features built"

FIRST_T=$($PY -c "import json;print(json.load(open('$B/first_stage/report.json'))['chosen']['threshold'])")
COMMON=(--train-context $B/context_train --train-competition $B/competition_train --tune-context $B/context_tune
        --tune-competition $B/competition_tune --incumbent "" --baseline-threshold "$FIRST_T" --trees 4000 --learning-rate .05 --threads 21)
FITS=()
for spec in "l127s42:127:200:42" "l127s7:127:200:7" "l255s42:255:400:42"; do
  IFS=: read -r name leaves child seed <<< "$spec"
  run scripts/train_stacked_context.py "${COMMON[@]}" --leaves $leaves --min-child $child --seed $seed \
    --output $B/stacked_$name --report-output $R/bridge_stacked_$name.json > $L/bridge_stacked_$name.log 2>&1 &
  FITS+=($!)
done
for pid in "${FITS[@]}"; do wait "$pid" || stamp "a fit failed: $pid"; done
stamp "second stage fitted"

MODELS=()
for d in $B/stacked_*; do [ -e "$d/selection_probabilities.npy" ] && MODELS+=("$d"); done
for d in "${MODELS[@]}"; do
  run scripts/evaluate_decision_rules.py --cache $B/context_tune --model "$d" --report-output $R/bridge_decision_rules_$(basename "$d").json > $L/bridge_decision_$(basename "$d").log 2>&1 || true
done
run scripts/evaluate_ensemble.py --cache $B/context_tune --models "${MODELS[@]}" --incumbent "" --baseline-threshold "$FIRST_T" \
  --report-output $R/bridge_ensemble.json > $L/bridge_ensemble.log 2>&1 || true
BEST=$($PY - "${MODELS[@]}" <<'EOF'
import json, sys
from pathlib import Path
scores = {d: (json.loads((Path(d) / "report.json").read_text())["protocol_selection"] or {"macro_f05": 0})["macro_f05"] for d in sys.argv[1:]}
print(max(scores, key=scores.get))
EOF
)
NAME=$(basename "$BEST")
stamp "best second stage: $NAME"
run scripts/analyze_selection_losses.py --cache $B/context_tune --model "$BEST" > $R/bridge_losses_$NAME.json 2> $L/bridge_losses_$NAME.log || true
run scripts/evaluate_exclusive_assignment.py --model "$BEST" --first-stage $B/first_stage --cache $B/context_tune \
  --competition $B/competition_tune --bridge-cache $B/cache_tune --output $B/exclusive_$NAME \
  --report-output $R/bridge_exclusive_$NAME.json --workers 56 > $L/bridge_exclusive_$NAME.log 2>&1
run scripts/evaluate_decision_rules.py --cache $B/context_tune --model "$BEST" --probabilities $B/exclusive_$NAME/adjusted_all_competitors.npy \
  --report-output $R/bridge_decision_rules_exclusive_$NAME.json > $L/bridge_decision_exclusive_$NAME.log 2>&1
stamp "BRIDGE_PIPELINE_DONE"
