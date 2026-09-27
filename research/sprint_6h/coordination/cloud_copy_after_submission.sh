#!/usr/bin/env bash
# Run on VM 1. Wait for the existing submission PID to exit before bulk I/O.
set -euo pipefail
cd /mnt/er/entity
while kill -0 84351 2>/dev/null; do sleep 20; done
/mnt/er/venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('output/submission_04/submission_report.json')
r=json.loads(p.read_text())
assert r['status']=='complete',r
assert not r['validation']['strict_issues']
assert r['validation']['official_returncode']==0
print('Submission 04 complete; starting direct cloud asset transfer',flush=True)
PY
nice -n 15 rsync -aRrL --partial-dir=.sprint-rsync-partial --stats \
  --files-from=/mnt/er/sprint_transfer/cloud_assets.txt \
  -e 'ssh -i /mnt/er/sprint_transfer/id_ed25519 -o UserKnownHostsFile=/mnt/er/sprint_transfer/vm2_known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15' \
  ./ azureuser@20.205.231.10:/home/azureuser/er-experiments/e70007/sprint_6h/entity/
date -u
touch /mnt/er/sprint_transfer/cloud_copy.complete
