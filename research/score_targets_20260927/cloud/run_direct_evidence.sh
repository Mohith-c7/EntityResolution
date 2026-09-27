#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/azureuser/er-experiments/e70007
cd "$ROOT/entity"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=6
export PYTHONUNBUFFERED=1
exec "$ROOT/venv/bin/python" research/score_targets_20260927/train_direct_evidence_expert.py \
    --data models/cpu_scale_stage2_20260927_e70007 \
    --baseline models/cpu_scale_results_20260927_e70007_seed70007 \
    --output models/direct_evidence_expert_20260927_e70010 \
    --threads 6 --trees 2400
