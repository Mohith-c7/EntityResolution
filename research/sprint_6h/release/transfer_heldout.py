#!/usr/bin/env python3
"""Run on VM1: copy sealed root-level heldout assets directly to isolated VM2.

Preserves source bytes and omits feature/score chunks. Hashes every selected file
before transfer and verifies destination bytes plus unchanged source manifest.
Uses the parent-provided temporary restricted SSH identity and pinned host key.
"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

SOURCE_ROOT = Path('/mnt/er/entity')
DESTINATION_ROOT = '/home/azureuser/er-experiments/e70007/sprint_6h/entity'
COHORT = Path('models/sprint_6h/heldout_cohort')
TRANSPORT = ['ssh','-i','/mnt/er/sprint_transfer/id_ed25519',
             '-o','UserKnownHostsFile=/mnt/er/sprint_transfer/vm2_known_hosts',
             '-o','StrictHostKeyChecking=yes','-o','BatchMode=yes']
TARGET = 'azureuser@20.205.231.10'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8 << 20),b''): h.update(block)
    return h.hexdigest()


def remote_inventory(relative_paths):
    code = "from pathlib import Path\nimport hashlib,json\nroot=Path("+repr(DESTINATION_ROOT)+")\npaths="+repr(relative_paths)+"\n"
    code += "def digest(path):\n h=hashlib.sha256()\n with path.open('rb') as f:\n  for b in iter(lambda:f.read(8<<20),b''):h.update(b)\n return h.hexdigest()\n"
    code += "print(json.dumps({p: digest(root/p) if (root/p).is_file() else None for p in paths}))\n"
    result = subprocess.run(TRANSPORT+[TARGET,'python3 -'],input=code,text=True,capture_output=True,check=True)
    return json.loads(result.stdout)


def transfer(output):
    output=Path(output)
    if output.exists():raise ValueError('Transfer report must be new/versioned')
    started=time.perf_counter()
    manifest_path=SOURCE_ROOT / COHORT / 'manifest.json'
    manifest_sha=digest(manifest_path); manifest=json.loads(manifest_path.read_text())
    if manifest.get('status')!='complete' or manifest.get('verified_current_pipeline') is not True:
        raise ValueError('Heldout cohort has not passed source seal')
    paths=sorted(str(p.relative_to(SOURCE_ROOT)) for p in (SOURCE_ROOT/COHORT).iterdir() if p.is_file() and not p.is_symlink())
    paths += ['research/sprint_6h/coordination/'+name for name in
              ('vm2_capture_parity.json','assets_verified_vm1.json','score_workbench.py','seal_workbench.py','verify_workbench_assets.py')]
    paths += ['models/frozen_v4_checkpoint_repair/frozen.json']
    files={p:{'bytes':(SOURCE_ROOT/p).stat().st_size,'sha256':digest(SOURCE_ROOT/p)} for p in paths}
    if files[str(COHORT/'pairs.parquet')]['sha256']!=manifest['pairs_sha256']:
        raise ValueError('Pair table differs from sealed manifest')
    proofs={'research/sprint_6h/coordination/vm2_capture_parity.json':manifest['parity_proof_sha256'],
            'research/sprint_6h/coordination/assets_verified_vm1.json':manifest['asset_proof_sha256'],
            'research/sprint_6h/coordination/score_workbench.py':manifest['capture_script_sha256'],
            'models/frozen_v4_checkpoint_repair/frozen.json':manifest['frozen_sha256']}
    if any(files[path]['sha256']!=sha for path,sha in proofs.items()):
        raise ValueError('Proof/code/frozen evidence differs from sealed cohort')
    existing=remote_inventory(paths)
    differing=[p for p,sha in existing.items() if sha is not None and sha!=files[p]['sha256']]
    if differing:raise ValueError('Destination already contains differing immutable files: '+repr(differing))
    print(json.dumps({'status':'copying','files':len(files),'bytes':sum(v['bytes'] for v in files.values()),'manifest_sha256':manifest_sha}),flush=True)
    command=['ionice','-c','3','nice','-n','19','rsync','-a','--partial','--protect-args','--bwlimit=102400',
             '--files-from=-','-e',shlex.join(TRANSPORT),str(SOURCE_ROOT)+'/',TARGET+':'+DESTINATION_ROOT+'/']
    subprocess.run(command,input='\n'.join(paths)+'\n',text=True,check=True)
    copied=remote_inventory(paths)
    if any(copied[p]!=files[p]['sha256'] for p in paths):raise ValueError('Destination copy checksum mismatch')
    if digest(manifest_path)!=manifest_sha:raise ValueError('Source manifest changed during copy')
    result={'status':'complete','copy_byte_parity':True,'manifest_sha256':manifest_sha,
            'source_root':str(SOURCE_ROOT),'destination_root':DESTINATION_ROOT,
            'entities':manifest['entities'],'pairs':manifest['pairs'],'pair_order_sha256':manifest['pair_order_sha256'],
            'feature_chunks_copied':False,'files':files,'elapsed_seconds':time.perf_counter()-started}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.partial');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(output)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(transfer(p.parse_args().output),indent=2),flush=True)
