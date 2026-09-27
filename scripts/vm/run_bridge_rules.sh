#!/bin/bash
# Rank-one decision rule on bridge candidates for every stacked fit, once follow-ups finish.
set -euo pipefail
cd /mnt/er/entity
until grep -q FOLLOWUPS_DONE /mnt/er/logs/run_followups.log; do sleep 20; done
for model in models/next_round/stacked_*; do
  name=$(basename $model)
  /mnt/er/venv/bin/python scripts/evaluate_decision_rules.py --model $model --bridge \
    --report-output reports/experiments/round3/decision_rules_bridge_$name.json > /mnt/er/logs/decision_bridge_$name.log 2>&1
done
echo "$(date -u +%H:%M:%S) BRIDGE_RULES_DONE"
