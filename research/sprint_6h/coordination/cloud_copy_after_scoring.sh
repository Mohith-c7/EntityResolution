#!/usr/bin/env bash
# Immutable input copy at idle I/O priority after all scoring workers finish.
# This job does not read, alter, signal, or resume submission output files.
set -euo pipefail
cd /mnt/er/entity
/mnt/er/venv/bin/python - <<'PY'
import json
from pathlib import Path
events = [json.loads(s) for s in Path('/mnt/er/logs/test_checkpoint_repair.log').read_text().splitlines() if s.startswith('{')]
assert any(e.get('stage') == 'test_scored' and e.get('pairs') == 69301760 for e in events)
print('Scoring workers finished; copying immutable inputs at idle I/O priority', flush=True)
PY
ionice -c 3 nice -n 19 rsync -aRrL --partial-dir=.sprint-rsync-partial --stats --bwlimit=102400 \
  --files-from=/mnt/er/sprint_transfer/cloud_assets.txt \
  -e 'ssh -i /mnt/er/sprint_transfer/id_ed25519 -o UserKnownHostsFile=/mnt/er/sprint_transfer/vm2_known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15' \
  ./ azureuser@20.205.231.10:/home/azureuser/er-experiments/e70007/sprint_6h/entity/
date -u
touch /mnt/er/sprint_transfer/cloud_copy.complete
