#!/usr/bin/env bash
# Isolated experiment VM only. This script never creates submission outputs.
set -euo pipefail
ROOT=/home/azureuser/er-experiments/e70007
cd "$ROOT/entity"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PY="$ROOT/venv/bin/python"
jobs=()
for seed in 70007 70008 70009; do
    destination="models/cpu_feature_results_20260927_e70007_seed${seed}"
    if [[ -e "$destination" ]]; then
        echo "Refusing to overwrite $destination" >&2
        exit 1
    fi
done
for seed in 70007 70008 70009; do
    "$PY" research/score_targets_20260927/cpu_feature_experiment.py train \
        --data models/cpu_feature_pilot_20260927_e70007 \
        --output "models/cpu_feature_results_20260927_e70007_seed${seed}" \
        --threads 8 --trees 1200 --early-stopping-rounds 75 --seed "$seed" \
        > "$ROOT/logs/cpu_seed${seed}.log" 2>&1 &
    pid=$!
    jobs+=("$pid")
    echo "seed=$seed pid=$pid"
done
status=0
for pid in "${jobs[@]}"; do
    if ! wait "$pid"; then
        echo "FAILED pid=$pid" >&2
        status=1
    fi
done
echo "CPU_TRIALS_DONE status=$status"
exit "$status"
