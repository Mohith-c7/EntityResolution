"""Offline entry point for packaged verification and inference reproduction."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
KEYS = ['source1_entity_id','candidate_entity_id']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''): h.update(part)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def setup(root):
    sys.path[:0] = [str(root/'scripts'),str(root/'research/sprint_6h'),
                    str(root/'research/final_2h'),str(root/'code/business_entity_resolution')]


def verify_package(package):
    marker_path=package/'package_manifest.json'
    marker=json.loads(marker_path.read_text())
    archive_root=package.parents[1]
    for name,expected in marker['files'].items():
        if sha(archive_root/name)!=expected: raise ValueError('Packaged file changed: '+name)
    return marker_path,marker


def score_fixture(root, fixture):
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    setup(root)
    import build_neural_last4_submission as exporter
    import build_gated_submission as reuse
    fm=json.loads((fixture/'manifest.json').read_text())
    for file,expected in fm['files'].items():
        if sha(fixture/file)!=expected: raise ValueError('Numeric fixture changed')
    base=pd.read_parquet(fixture/'pairs.parquet')
    features=pd.read_parquet(fixture/'features37.parquet')
    fine=pd.read_parquet(fixture/'fine_scores.parquet')
    adapter=root/'research/final_2h/neural_last4_fine_v1/adapter.txt'
    if sha(adapter)!=exporter.ADAPTER_SHA: raise ValueError('Selected native adapter changed')
    if sha(root/'models/final_2h/neural_last4_continue_v1/manifest.json')!=fm['fine_head_manifest_sha256']:
        raise ValueError('Numeric fixture belongs to another fine head')
    if sha(root/'models/sprint_6h/neural/frozen_head120k_v1/manifest.json')!=fm['old_head_manifest_sha256']:
        raise ValueError('Numeric fixture belongs to another frozen head')
    assembled=exporter.assemble_fine(features,fine)
    changed=exporter.apply_adapter(base,assembled,lgb.Booster(model_file=str(adapter)),threads=1)
    if not base[[*KEYS,'candidate_order']].equals(changed[[*KEYS,'candidate_order']]):
        raise ValueError('Candidate membership/order changed')
    positions=exporter.route_positions(base,assembled)
    outside=np.ones(len(base),dtype=bool); outside[positions]=False
    if not np.array_equal(base.probability.to_numpy()[outside],changed.probability.to_numpy()[outside]):
        raise ValueError('Outside-route probabilities changed')
    config=json.loads((root/'research/sprint_6h/coordination/baseline_freeze/frozen.json').read_text())
    config={k:config[k] for k in ('threshold','t_first','t_rest')}
    chosen,lost=reuse.load_original_decide(root/'scripts/run_frozen_pipeline.py')(changed,changed,config)
    return changed,chosen,lost,{'owners':fm['owners'],'pairs':len(base),'routed_pairs':len(features),
        'unchanged_unrouted_pairs':int(outside.sum()),'accepted_pairs':int(chosen.sum()),
        'candidate_order_preserved':True,'frozen_probability_anchored_once':True,
        'labels_read':False,'raw_business_records_included':False,
        'python':sys.version.split()[0],'lightgbm':lgb.__version__,
        'adapter_sha256':sha(adapter),'exporter_sha256':sha(exporter.__file__)}


def reference(args, root):
    import numpy as np
    fixture=args.fixture
    if (fixture/'expected_probabilities.npy').exists(): raise FileExistsError('Reference already sealed')
    changed,chosen,lost,report=score_fixture(root,fixture)
    np.save(fixture/'expected_probabilities.npy',changed.probability.to_numpy())
    np.save(fixture/'expected_chosen.npy',chosen)
    np.save(fixture/'expected_lost.npy',lost)
    marker=json.loads((fixture/'manifest.json').read_text())
    for file in ('expected_probabilities.npy','expected_chosen.npy','expected_lost.npy'):
        marker['files'][file]=sha(fixture/file)
    marker['reference']=report
    write(fixture/'manifest.json',marker)
    print(json.dumps(report))


def sample(args, root, package):
    import numpy as np
    if args.output.exists(): raise FileExistsError(args.output)
    manifest_path,_=verify_package(package)
    fixture=root/'verification_fixture'
    changed,chosen,lost,report=score_fixture(root,fixture)
    expected=np.load(fixture/'expected_probabilities.npy',allow_pickle=False)
    max_error=float(np.max(np.abs(expected-changed.probability.to_numpy()),initial=0.))
    if max_error>1e-12 or not np.array_equal(chosen,np.load(fixture/'expected_chosen.npy',allow_pickle=False)) or not np.array_equal(lost,np.load(fixture/'expected_lost.npy',allow_pickle=False)):
        raise ValueError('Packaged scoring/decisions differ from sealed actual-subset reference')
    report.update(status='packaged_actual_subset_passed',package_manifest_sha256=sha(manifest_path),
                  max_probability_error=max_error,selected_masks_exact=True,
                  scope='Actual exposed numeric subset; checks adapter/keyed scatter and original decisions. No cold full-test reproduction or neural/retrieval runtime claim.')
    args.output.mkdir(parents=True)
    changed.to_parquet(args.output/'scored_pairs.parquet',index=False)
    write(args.output/'verification.json',report)
    print(json.dumps(report,indent=2))


def command(python, script, *args, cwd):
    subprocess.run([str(python),str(script),*map(str,args)],cwd=cwd,check=True)


def workspace(root, work):
    """Expose immutable source/weights without writing into the extracted package."""
    work.mkdir(parents=True,exist_ok=True)
    for name in ('scripts','code'):
        dest=work/name
        if not dest.exists(): dest.symlink_to(root/name,target_is_directory=True)
    (work/'research').mkdir(exist_ok=True)
    for name in ('sprint_6h','final_2h'):
        dest=work/'research'/name
        if not dest.exists(): dest.symlink_to(root/'research'/name,target_is_directory=True)
    (work/'models').mkdir(exist_ok=True)
    for name in ('bridge_v1','redesign','scale_v1_plan','sprint_6h','final_2h'):
        dest=work/'models'/name
        if not dest.exists(): dest.symlink_to(root/'models'/name,target_is_directory=True)
    (work/'models/runtime').mkdir(exist_ok=True)


def indexes(args, root):
    setup(root)
    from src.blocking.disk_index import build_disk_index
    import reverse_competition
    workspace(root,args.work_dir)
    for source in (2,3):
        build_disk_index(args.test_dir/f'test_source{source}.tsv',args.work_dir/f'models/index_test_S{source}.sqlite',f'S{source}')
    reverse_competition.build_index(args.test_dir/'test_source1.tsv',args.work_dir/'test_s1.sqlite','test')
    options=['--output',args.work_dir/'models/runtime/erpostings.dylib']
    if args.sqlite_include: options.extend(['--sqlite-include',args.sqlite_include])
    command(args.cpu_python,root/'scripts/build_postings_native.py',*options,cwd=args.work_dir)


def base(args, root):
    workspace(root,args.work_dir)
    command(args.cpu_python,root/'scripts/run_frozen_pipeline.py','test',
        '--frozen',root/'research/sprint_6h/coordination/baseline_freeze/frozen.json',
        '--test-dir',args.test_dir,'--workers',args.workers,'--output',args.work_dir/'base',cwd=args.work_dir)


def features(args, root):
    setup(root)
    import build_gated_submission as reuse
    import train_sibling_matcher as sibling
    work=args.work_dir
    paths=[work/f'models/index_test_S{s}.sqlite' for s in (2,3)]
    paths += [Path(str(p)+'.frequencies.sqlite') for p in list(paths)]
    paths += [work/'test_s1.sqlite',work/'models/runtime/erpostings.dylib']
    baseline=root/'research/sprint_6h/coordination/baseline_freeze/frozen.json'
    reuse_path=work/'reuse_manifest.json'
    reuse.seal_reuse_manifest(work/'base',args.test_dir,baseline,{str(p):sha(p) for p in paths},reuse_path,root=work)
    rows,frame=reuse.load_verified_chunks(json.loads(reuse_path.read_text()),args.test_dir,root=work)
    full=work/'full_claims'; full.mkdir(exist_ok=False)
    frame.to_parquet(full/'pairs.parquet',index=False)
    write(full/'manifest.json',{'status':'complete','reference_ids':[r['entity_id'] for r in rows],
        'verified_current_pipeline':True,
        'entities':len(rows),'pairs':len(frame),'pairs_sha256':sha(full/'pairs.parquet'),
        'pair_order_sha256':sibling.pair_order_hash(frame),'frozen_sha256':sha(baseline),
        'split':'test','role':'test','labels_read':False})
    del frame
    command(args.cpu_python,root/'research/sprint_6h/parallel_reverse_features.py',
        '--pairs',full/'pairs.parquet','--manifest',full/'manifest.json','--index',work/'test_s1.sqlite',
        '--target-index-prefix',work/'models/index_test','--workers',args.workers,
        '--output',work/'reverse',cwd=work)
    command(args.cpu_python,root/'research/sprint_6h/sibling/train_reverse_adapter.py','join',
        '--pairs',full/'pairs.parquet','--manifest',full/'manifest.json',
        '--reverse-features',work/'reverse/features.parquet','--reverse-manifest',work/'reverse/manifest.json',
        '--route-plan',work/'reverse/global_route','--output',work/'combined_reverse',cwd=work)
    raw=work/'provided_test_inputs'; raw.mkdir()
    for n in (1,2,3): (raw/f'train_source{n}.tsv').symlink_to(args.test_dir/f'test_source{n}.tsv')
    command(args.cpu_python,root/'research/sprint_6h/prepare_neural_inputs.py',
        '--features',work/'combined_reverse','--data-dir',raw,
        '--output',work/'neural/neural_inputs/test',cwd=work)


def neural_scores(args, root):
    """Bound both offline neural scorers into sealed 10k-row inference batches."""
    work=args.work_dir
    inputs=work/'neural/neural_inputs/test'
    full=json.loads((inputs/'manifest.json').read_text())
    pieces=work/'neural/pieces'; pieces.mkdir(exist_ok=False)
    old_parts=[]; fine_parts=[]
    def process(number,lines):
        part=pieces/str(number); part.mkdir()
        input_path=part/'pairs.jsonl'; input_path.write_text(''.join(lines))
        im={**full,'rows':len(lines),'input_sha256':sha(input_path),'parent_input_manifest_sha256':sha(inputs/'manifest.json')}
        write(part/'manifest.json',im)
        command(args.neural_python,root/'research/sprint_6h/neural_frozen_head.py','score',
            '--input',input_path,'--manifest',part/'manifest.json',
            '--base',root/'models/sprint_6h/neural/minilm_base',
            '--head',root/'models/sprint_6h/neural/frozen_head120k_v1',
            '--output',part/'old_scores',cwd=work)
        command(args.neural_python,root/'research/final_2h/neural_layers.py','score',
            '--input',input_path,'--input-manifest',part/'manifest.json',
            '--model',root/'models/final_2h/neural_last4_continue_v1/checkpoint',
            '--max-minutes',30,'--output',part/'fine_scores',cwd=work)
        old_parts.append(part/'old_scores'); fine_parts.append(part/'fine_scores')
    lines=[]; total=0; number=0
    with (inputs/'pairs.jsonl').open() as stream:
        for line in stream:
            lines.append(line); total+=1
            if len(lines)==10000:
                process(number,lines); number+=1; lines=[]
    if lines: process(number,lines)
    if total!=full['rows']: raise ValueError('Neural input coverage changed')
    for name,parts,head_field,head in [('test',old_parts,'head_manifest_sha256',root/'models/sprint_6h/neural/frozen_head120k_v1/manifest.json'),
                                    ('fine_test',fine_parts,'checkpoint_manifest_sha256',root/'models/final_2h/neural_last4_continue_v1/manifest.json')]:
        output=work/'neural'/name; output.mkdir()
        with (output/'scores.jsonl').open('x') as dest:
            for part in parts:
                pm=json.loads((part/'manifest.json').read_text())
                if pm['status']!='complete' or pm.get('labels_read') is not False or pm[head_field]!=sha(head) or sha(part/'scores.jsonl')!=pm['scores_sha256']:
                    raise ValueError('Neural part is incomplete or unsealed')
                with (part/'scores.jsonl').open() as source:
                    for line in source: dest.write(line)
        write(output/'manifest.json',{'status':'complete','labels_read':False,'rows':total,
            'scores_sha256':sha(output/'scores.jsonl'),head_field:sha(head),
            'input_sha256':full['input_sha256'],'input_manifest_sha256':sha(inputs/'manifest.json'),
            'bounded_part_manifest_sha256':[sha(p/'manifest.json') for p in parts],
            'score_code_sha256':sha(__file__)})
    command(args.cpu_python,root/'research/sprint_6h/neural_adapter.py','join',
        '--features',work/'combined_reverse','--neural',work/'neural/test',
        '--head-manifest',root/'models/sprint_6h/neural/frozen_head120k_v1/manifest.json',
        '--output',work/'combined_neural',cwd=work)


def export(args, root):
    work=args.work_dir
    original=root/'selected_release_freeze.json'
    frozen=json.loads(original.read_text())
    # SQLite metadata includes build paths/timings, so rebuilding is semantically
    # reproducible but cannot retain the original database byte hashes. Keep
    # all trained-model, routing, schema and source pins; attest new derived data.
    indexes=[work/f'models/index_test_S{s}.sqlite' for s in (2,3)]
    indexes += [Path(str(p)+'.frequencies.sqlite') for p in list(indexes)]
    indexes += [work/'test_s1.sqlite']
    frozen['index_sha256']={str(p):sha(p) for p in indexes}
    native=work/'models/runtime/erpostings.dylib'
    frozen['native_sha256']={str(native):sha(native)}
    frozen['reproduction_of_selected_release_sha256']=sha(original)
    frozen['frozen_at']=datetime.now(timezone.utc).isoformat()
    frozen['reproduction_note']='Identical selected model/code/routing; derived provided-data indices/native build freshly attested. No audit evaluation or retraining.'
    release=work/'reproduction_freeze.json'; write(release,frozen)
    command(args.cpu_python,root/'scripts/build_neural_last4_submission.py',
        '--freeze',release,'--reuse-manifest',work/'reuse_manifest.json',
        '--features',work/'combined_neural','--fine-scores',work/'neural/fine_test',
        '--fine-input-manifest',work/'neural/neural_inputs/test/manifest.json',
        '--fine-head-manifest',root/'models/final_2h/neural_last4_continue_v1/manifest.json',
        '--old-head-manifest',root/'models/sprint_6h/neural/frozen_head120k_v1/manifest.json',
        '--model',root/'research/final_2h/neural_last4_fine_v1',
        '--route-manifest',work/'reverse/global_route/manifest.json',
        '--test-dir',args.test_dir,'--threads',args.workers,'--output',work/'final_output',cwd=work)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['verify-assets','verify-sample','reference','indexes','base','features','neural-scores','export','reproduce'])
    p.add_argument('--runtime-root',type=Path)
    p.add_argument('--fixture',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--test-dir',type=Path)
    p.add_argument('--work-dir',type=Path)
    p.add_argument('--cpu-python',type=Path,default=Path(sys.executable))
    p.add_argument('--neural-python',type=Path)
    p.add_argument('--sqlite-include',type=Path)
    p.add_argument('--workers',type=int,default=16)
    a=p.parse_args()
    package=Path(__file__).resolve().parent.parent
    root=(a.runtime_root or package/'src/final_pipeline').resolve()
    if a.command=='reference':
        if a.fixture is None: p.error('Reference needs --fixture')
        reference(a,root); return
    if a.command=='verify-assets':
        _,marker=verify_package(package); print(json.dumps({'verified_files':len(marker['files']),'offline':True})); return
    if a.command=='verify-sample':
        if a.output is None: p.error('Sample needs unused --output directory')
        sample(a,root,package); return
    if a.work_dir is None or a.test_dir is None or not 1<=a.workers<=16:
        p.error('Inference stages require --work-dir, --test-dir, workers1..16')
    a.work_dir=a.work_dir.resolve(); a.test_dir=a.test_dir.resolve()
    a.cpu_python=a.cpu_python.resolve()
    if a.neural_python: a.neural_python=a.neural_python.resolve()
    if a.command in ('neural-scores','reproduce') and a.neural_python is None:
        p.error('Offline MPS scoring requires --neural-python')
    # Verification happens before any inference stage. No labels are needed.
    verify_package(package)
    stages={'indexes':indexes,'base':base,'features':features,'neural-scores':neural_scores,'export':export}
    for name in (list(stages) if a.command=='reproduce' else [a.command]):
        print(json.dumps({'stage':name,'work_dir':str(a.work_dir)}),flush=True)
        stages[name](a,root)


if __name__=='__main__': main()
