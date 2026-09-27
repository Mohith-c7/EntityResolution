#!/bin/bash
# Both stages refit on bridge-retrieval candidates: cross-fitted bridge caches for the
# 300k training and 50k tuning references, a new first stage with out-of-fold scores,
# context and competition features, second-stage fits, then the decision-rule and
# exclusivity evaluations on the same 20k selection references.
set -euo pipefail
cd /mnt/er/entity
PY=/mnt/er/venv/bin/python
L=/mnt/er/logs
R=reports/experiments/round3
B=models/bridge_v1
stamp() { echo "$(date -u +%H:%M:%S) $*"; }
run() { nice -n 19 "$PY" "$@"; }
mkdir -p "$B" "$R"

run scripts/build_scale_cache.py --split tune --bridge --workers 60 --idle-workers 60 --output $B/cache_tune > $L/bridge_cache_tune.log 2>&1
stamp "tuning bridge cache built"
run scripts/build_scale_cache.py --split train --bridge --workers 60 --idle-workers 60 --output $B/cache_train > $L/bridge_cache_train.log 2>&1
stamp "training bridge cache built"

run scripts/build_competition_features.py --cache $B/cache_tune --reference-split tune --output $B/competition_tune > $L/bridge_competition_tune.log 2>&1 &
COMP_TUNE=$!
run scripts/build_competition_features.py --cache $B/cache_train --reference-split train --output $B/competition_train > $L/bridge_competition_train.log 2>&1 &
COMP_TRAIN=$!

run scripts/train_scale_model.py --train-cache $B/cache_train --tune-cache $B/cache_tune --output $B/first_stage --threads 56 --trees 3000 > $L/bridge_first_stage.log 2>&1
stamp "first stage fitted"
FOLDS=()
for k in 0 1 2 3 4; do
  run scripts/train_oof_first_stage.py --model-dir $B/first_stage --output $B/oof --fold $k --threads 11 > $L/bridge_oof_$k.log 2>&1 &
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
        --tune-competition $B/competition_tune --incumbent "" --baseline-threshold "$FIRST_T" --trees 3000 --learning-rate .05 --threads 16)
run scripts/train_stacked_context.py "${COMMON[@]}" --leaves 31 --min-child 100 --output $B/stacked_a31 --report-output $R/bridge_stacked_a31.json > $L/bridge_stacked_a31.log 2>&1 &
S1=$!
run scripts/train_stacked_context.py "${COMMON[@]}" --leaves 31 --min-child 100 --add-development-training --output $B/stacked_b31dev --report-output $R/bridge_stacked_b31dev.json > $L/bridge_stacked_b31dev.log 2>&1 &
S2=$!
run scripts/train_stacked_context.py "${COMMON[@]}" --leaves 127 --min-child 200 --output $B/stacked_c127 --report-output $R/bridge_stacked_c127.json > $L/bridge_stacked_c127.log 2>&1 &
S3=$!
run scripts/train_stacked_context.py "${COMMON[@]}" --leaves 127 --min-child 200 --add-development-training --output $B/stacked_d127dev --report-output $R/bridge_stacked_d127dev.json > $L/bridge_stacked_d127dev.log 2>&1 &
S4=$!
wait $S1; wait $S2; wait $S3; wait $S4
stamp "second stage fitted"

BEST=$($PY - <<'EOF'
import json
from pathlib import Path
reports = {p.parent.name: json.loads((p).read_text()) for p in Path("models/bridge_v1").glob("stacked_*/report.json")}
print(max(reports, key=lambda k: (reports[k]["protocol_selection"] or {"macro_f05": 0})["macro_f05"]))
EOF
)
stamp "best second stage: $BEST"
run scripts/evaluate_decision_rules.py --cache $B/context_tune --model $B/$BEST --report-output $R/bridge_decision_rules_$BEST.json > $L/bridge_decision_$BEST.log 2>&1
run scripts/analyze_selection_losses.py --cache $B/context_tune --model $B/$BEST > $R/bridge_losses_$BEST.json 2> $L/bridge_losses_$BEST.log
run scripts/evaluate_exclusive_assignment.py --model $B/$BEST --first-stage $B/first_stage --cache $B/context_tune \
  --competition $B/competition_tune --bridge-cache $B/cache_tune --output $B/exclusive_$BEST \
  --report-output $R/bridge_exclusive_$BEST.json --workers 56 > $L/bridge_exclusive_$BEST.log 2>&1
run scripts/evaluate_decision_rules.py --cache $B/context_tune --model $B/$BEST --probabilities $B/exclusive_$BEST/adjusted_all_competitors.npy \
  --report-output $R/bridge_decision_rules_exclusive_$BEST.json > $L/bridge_decision_exclusive_$BEST.log 2>&1
stamp "BRIDGE_PIPELINE_DONE"
