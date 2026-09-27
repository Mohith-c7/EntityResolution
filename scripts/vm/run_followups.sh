#!/bin/bash
# After the stacked fits: bridge candidates, decision rules and exclusivity for the best fit.
set -euo pipefail
cd /mnt/er/entity
PY=/mnt/er/venv/bin/python
L=/mnt/er/logs
R=reports/experiments/round3
stamp() { echo "$(date -u +%H:%M:%S) $*"; }
until grep -q ALL_DONE $L/run_stacked_stage.log; do sleep 20; done
while pgrep -f "evaluate_bridge_stacked.py --workers 8" > /dev/null; do sleep 10; done
BEST=$($PY - <<'EOF'
import json
from pathlib import Path
scores = {}
for name in ("a31", "b31dev", "c127", "d127dev"):
    path = Path(f"models/next_round/stacked_{name}/report.json")
    if path.exists() and json.loads(path.read_text())["protocol_selection"]:
        scores[name] = json.loads(path.read_text())["protocol_selection"]["macro_f05"]
print(max(scores, key=scores.get))
EOF
)
stamp "best stacked fit: $BEST"
MODELS=$(ls -d models/next_round/stacked_* | tr '\n' ' ')
$PY scripts/evaluate_bridge_stacked.py --models $MODELS --workers 32 > $L/bridge_eval.log 2>&1 &
BRIDGE=$!
$PY scripts/evaluate_decision_rules.py --model models/next_round/stacked_$BEST --report-output $R/decision_rules_$BEST.json > $L/decision_$BEST.log 2>&1
stamp "decision rules done"
$PY scripts/evaluate_exclusive_assignment.py --model models/next_round/stacked_$BEST --output models/next_round/exclusive_$BEST \
  --report-output $R/exclusive_$BEST.json --workers 56 > $L/exclusive_$BEST.log 2>&1
stamp "exclusivity done"
$PY scripts/evaluate_decision_rules.py --model models/next_round/stacked_$BEST \
  --probabilities models/next_round/exclusive_$BEST/adjusted_all_competitors.npy \
  --report-output $R/decision_rules_exclusive_$BEST.json > $L/decision_exclusive_$BEST.log 2>&1
wait $BRIDGE
stamp "FOLLOWUPS_DONE"
