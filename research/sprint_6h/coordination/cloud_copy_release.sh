#!/usr/bin/env bash
# Copy immutable completed baseline for isolated gate/runtime experiments.
set -euo pipefail
cd /mnt/er/entity
/mnt/er/venv/bin/python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('output/submission_04/submission_report.json').read_text())
assert r['status'] == 'complete' and not r['validation']['strict_issues']
assert r['validation']['official_returncode'] == 0
print('Copying verified, complete baseline and test indexes', flush=True)
PY
ionice -c 3 nice -n 19 rsync -aRrL --partial-dir=.sprint-rsync-partial --stats --bwlimit=102400 \
  --files-from=/mnt/er/sprint_transfer/cloud_release_assets.txt \
  -e 'ssh -i /mnt/er/sprint_transfer/id_ed25519 -o UserKnownHostsFile=/mnt/er/sprint_transfer/vm2_known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15' \
  ./ azureuser@20.205.231.10:/home/azureuser/er-experiments/e70007/sprint_6h/entity/
date -u
touch /mnt/er/sprint_transfer/cloud_release_copy.complete
