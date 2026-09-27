"""Build a self-contained offline archive without raw datasets or fitted changes.

All original runtime module bytes and relative repository paths are preserved
inside src/final_pipeline. Staging is separate from final archive publication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CORE = Path('code/business_entity_resolution')
RUNTIME = CORE/'src/final_pipeline'
OUTPUT_NAMES = ('matching_results.tsv', 'candidate_pairs.tsv')
ASSETS = (
    'models/bridge_v1/first_stage/model.txt',
    'models/bridge_v1/stacked_l255s42/model.txt',
    'models/redesign/cheap_ranker_v1/model.txt',
    'models/scale_v1_plan/aliases/full.json',
    'models/runtime/erpostings.dylib',
    'models/sprint_6h/neural/frozen_head120k_v1/head.pt',
    'models/sprint_6h/neural/frozen_head120k_v1/scaling.npz',
    'models/sprint_6h/neural/frozen_head120k_v1/manifest.json',
    'research/final_2h/neural_last4_fine_v1/adapter.txt',
    'research/final_2h/neural_last4_fine_v1/report.json',
    'research/sprint_6h/coordination/baseline_freeze/frozen.json',
    'research/sprint_6h/neural_v2_schema.json',
)
EVIDENCE = (
    'reports/final_2h/neural_last4_final_audit_freeze.json',
    'reports/final_2h/neural_last4_release_v3_validation_proof.json',
    'research/final_2h/snapshots/build_neural_last4_submission_audited_fe1cb234.py',
    'research/final_2h/neural_last4_runtime_final_v2/runtime.json',
    'research/final_2h/neural_last4_final_audit_v1/report.json',
    'reports/final_2h/neural_last4_final_freeze_verification.json',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''): h.update(part)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def relative_file(repo, name):
    name = Path(name)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Package assets must have repository-relative paths')
    if name.parts[0] in ('dataset', '.git'):
        raise ValueError('Raw datasets and repository internals are forbidden')
    path = repo/name
    if not path.is_file(): raise FileNotFoundError(path)
    return path


def copy_file(repo, name, destination):
    source = relative_file(repo, name)
    target = destination/Path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if digest(source) != digest(target): raise ValueError('Copy differs: '+str(name))


def asset_file(repo, name, overlay):
    if overlay is not None and (overlay/Path(name)).is_file():
        return relative_file(overlay,name)
    return relative_file(repo,name)


def source_files(repo):
    """Source only: no research caches, labels, private keys, or raw records."""
    paths = list((repo/CORE/'src').rglob('*.py')) + list((repo/CORE/'src').rglob('*.c'))
    paths += list((repo/'scripts').glob('*.py'))
    paths += list((repo/'research/sprint_6h').glob('*.py'))
    paths += list((repo/'research/sprint_6h/sibling').glob('*.py'))
    paths += list((repo/'research/final_2h').glob('*.py'))
    # Source nested under our own package would recurse on a second assembly.
    return sorted({p.relative_to(repo) for p in paths if 'final_pipeline' not in p.parts})


def prepare_fixture(args):
    """Read an exposed numeric runtime block; retain complete owner groups."""
    import numpy as np
    import pandas as pd
    if args.output.exists(): raise FileExistsError(args.output)
    base = pd.read_parquet(args.fixture_base)
    features = pd.read_parquet(args.fixture_features)
    scores = pd.read_json(args.fixture_scores/'scores.jsonl', lines=True)
    score_marker = json.loads((args.fixture_scores/'manifest.json').read_text())
    feature_marker_path = args.fixture_features.parent/'manifest.json'
    feature_marker = json.loads(feature_marker_path.read_text())
    input_marker = json.loads(args.fixture_input_manifest.read_text())
    if (feature_marker.get('status') != 'complete'
            or feature_marker['features_sha256'] != digest(args.fixture_features)
            or score_marker.get('status') != 'complete' or score_marker.get('labels_read') is not False
            or score_marker['scores_sha256'] != digest(args.fixture_scores/'scores.jsonl')
            or score_marker['checkpoint_manifest_sha256'] != digest(args.fine_head_manifest)
            or score_marker['input_sha256'] != input_marker['input_sha256']
            or feature_marker['neural_head_manifest_sha256'] != digest(args.old_head_manifest)):
        raise ValueError('Fixture runtime provenance differs')
    keys = ['source1_entity_id','candidate_entity_id']
    if any(f.duplicated(keys).any() for f in (base,features,scores)):
        raise ValueError('Duplicate fixture pair keys')
    if set(map(tuple,features[keys].to_numpy())) != set(map(tuple,scores[keys].to_numpy())):
        raise ValueError('Fine score coverage differs from full37 route')
    owners = sorted(features.source1_entity_id.unique())[:args.fixture_owners]
    if len(owners) != args.fixture_owners: raise ValueError('Too few routed owners')
    pairs = base.loc[base.source1_entity_id.isin(owners)].copy().reset_index(drop=True)
    features = features.loc[features.source1_entity_id.isin(owners)].copy().reset_index(drop=True)
    scores = scores.loc[scores.source1_entity_id.isin(owners)].copy().reset_index(drop=True)
    counts = pairs.groupby('source1_entity_id').size()
    if len(counts) != len(owners) or not counts.eq(40).all():
        raise ValueError('Fixture must preserve every original40 candidate')
    if 'candidate_order' not in pairs:
        pairs['candidate_order'] = pairs.groupby('source1_entity_id',sort=False).cumcount()
    allowed_base = keys+['first_stage','second_stage','probability','candidate_order']
    pairs = pairs[[c for c in allowed_base if c in pairs]]
    feature_names = feature_marker['features']
    features = features[[*keys,*feature_names]]
    scores = scores[[*keys,'neural_probability']]
    if not np.isfinite(features[feature_names].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite fixture values')
    args.output.mkdir(parents=True)
    for name, frame in [('pairs.parquet',pairs),('features37.parquet',features),('fine_scores.parquet',scores)]:
        frame.to_parquet(args.output/name,index=False)
    write_json(args.output/'manifest.json',{
        'status':'complete_exposed_numeric_fixture', 'owners':len(owners), 'pairs':len(pairs),
        'routed_pairs':len(features), 'features':feature_names,
        'labels_read':False, 'raw_business_records_included':False,
        'source_bounds':{'base_pairs_sha256':digest(args.fixture_base),
            'full37_features_sha256':feature_marker['features_sha256'],
            'full37_manifest_sha256':digest(feature_marker_path),
            'fine_scores_sha256':score_marker['scores_sha256'],
            'fine_score_manifest_sha256':digest(args.fixture_scores/'manifest.json'),
            'input_manifest_sha256':digest(args.fixture_input_manifest)},
        'old_head_manifest_sha256':digest(args.old_head_manifest),
        'fine_head_manifest_sha256':digest(args.fine_head_manifest),
        'files':{n:digest(args.output/n) for n in ('pairs.parquet','features37.parquet','fine_scores.parquet')},
        'owner_ids_sha256':hashlib.sha256('\n'.join(owners).encode()).hexdigest(),
        'preparation_code_sha256':digest(__file__),
    })
    print(json.dumps({'fixture':str(args.output),'owners':len(owners),'pairs':len(pairs),'routed_pairs':len(features)}))


def stage(args):
    started = time.monotonic()
    if args.output.exists(): raise FileExistsError(args.output)
    repo = args.repo.resolve()
    freeze = json.loads(args.freeze.read_text())
    if freeze.get('status') != 'candidate_frozen': raise ValueError('Selected candidate freeze required')
    for field in ('audit_freeze', 'release_validation_proof', 'audit_report', 'runtime_evidence'):
        evidence = freeze.get(field)
        if evidence and digest(relative_file(repo,evidence['path'])) != evidence['sha256']:
            raise ValueError('Release lineage evidence changed: '+field)
    sources = source_files(repo)
    sources = set(sources)
    for name, expected in freeze.get('code_sha256',{}).items():
        source=relative_file(repo,name)
        if digest(source)!=expected: raise ValueError('Frozen source changed: '+name)
        sources.add(Path(name))
    assets = set(ASSETS)
    for directory in ('models/sprint_6h/neural/minilm_base','models/final_2h/neural_last4_continue_v1/checkpoint'):
        assets.update(p.relative_to(repo).as_posix() for p in (repo/directory).iterdir() if p.is_file())
    assets.add('models/final_2h/neural_last4_continue_v1/manifest.json')
    # Selected freeze hashes every required model/schema/native asset. Indexes
    # are reproducibly derived from supplied TSVs and deliberately not bundled.
    for group in ('model_sha256','schema_sha256','native_sha256'):
        if not freeze.get(group): raise ValueError('Missing freeze group '+group)
        for name, expected in freeze[group].items():
            file = asset_file(repo,name,args.asset_root)
            if digest(file) != expected: raise ValueError('Frozen asset changed: '+name)
            assets.add(name)
    for name in assets: asset_file(repo,name,args.asset_root)
    for name in EVIDENCE: relative_file(repo,name)
    fixture = json.loads((args.fixture/'manifest.json').read_text())
    if fixture.get('status') != 'complete_exposed_numeric_fixture' or fixture.get('raw_business_records_included') is not False:
        raise ValueError('Require exposed numeric-only fixture')
    if not fixture.get('reference') or not {'expected_probabilities.npy','expected_chosen.npy','expected_lost.npy'} <= set(fixture['files']):
        raise ValueError('Actual-subset reference scores and decisions must be sealed before staging')
    args.output.mkdir(parents=True)
    runtime = args.output/RUNTIME
    for name in sorted(sources): copy_file(repo,name,runtime)
    # The required src directory contains the complete original core source too.
    for name in sorted(sources):
        if CORE/'src' in name.parents: copy_file(repo,name,args.output)
    for name in assets:
        source=asset_file(repo,name,args.asset_root)
        destination=runtime/Path(name)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        if digest(source)!=digest(destination): raise ValueError('Asset copy differs: '+str(name))
    copy_file(repo,'code/business_entity_resolution/src/model/MODEL_LICENSE.txt',runtime)
    copy_file(repo,'code/business_entity_resolution/src/model/MODEL_LICENSE.txt',args.output)
    shutil.copyfile(args.freeze, runtime/'selected_release_freeze.json')
    # Aggregate audit/runtime evidence only: no owner truth or raw datasets.
    for name in EVIDENCE: copy_file(repo,name,runtime)
    shutil.copytree(args.fixture, runtime/'verification_fixture')
    docs = repo/'docs/final_submission'
    for name,target in [('README.md',args.output/CORE/'README.md'),
                        ('requirements-cpu.txt',args.output/CORE/'requirements.txt'),
                        ('requirements-neural.txt',args.output/CORE/'requirements-neural.txt'),
                        ('Documentation_template.md',args.output/'Documentation_template.md')]:
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(docs/name,target)
    shutil.copyfile(repo/'scripts/run_packaged_submission.py',args.output/CORE/'src/run_final_submission.py')
    files = {p.relative_to(args.output).as_posix():digest(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    write_json(args.output/CORE/'package_manifest.json',{
        'status':'staged_requires_packaged_sample_and_final_outputs',
        'created_at':datetime.now(timezone.utc).isoformat(), 'selected_freeze_sha256':digest(args.freeze),
        'files':files, 'raw_datasets_included':False, 'offline_weights_included':True,
        'indices':'Rebuild from organizer-provided source TSVs using included source; not included in archive',
        'assembly_seconds':time.monotonic()-started,
    })
    print(json.dumps({'staged':str(args.output),'files':len(files),'bytes':sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()),'seconds':time.monotonic()-started}))


def archive(args):
    """Archive only after the final output validators and package smoke pass."""
    if args.output.exists(): raise FileExistsError(args.output)
    package = args.stage/CORE
    marker = json.loads((package/'package_manifest.json').read_text())
    verification = json.loads(args.verification.read_text())
    report = json.loads((args.submission/'submission_report.json').read_text())
    valid = report.get('validation',{})
    if report.get('status') != 'complete' or valid.get('strict') != 'PASS' or valid.get('official') != 'PASS':
        raise ValueError('Complete strict/official final output validation required')
    if verification.get('status') != 'packaged_actual_subset_passed' or verification.get('package_manifest_sha256') != digest(package/'package_manifest.json'):
        raise ValueError('Actual packaged subset verification is missing or belongs to another package')
    if report.get('candidate_frozen_sha256') != marker['selected_freeze_sha256']:
        raise ValueError('Output belongs to another frozen candidate')
    for name, expected in marker['files'].items():
        if Path(name).suffix in ('.py', '.c') and not Path(name).is_relative_to(CORE/'src'):
            raise ValueError('All source must be under code/business_entity_resolution/src: '+name)
        if digest(args.stage/name) != expected: raise ValueError('Staged artifact changed: '+name)
    for name in OUTPUT_NAMES:
        if digest(args.submission/name) != report['files_sha256'][name]: raise ValueError('Validated output changed: '+name)
    # No write to Submission04, Submission05, or any inference working files.
    temporary = args.output.with_suffix(args.output.suffix+'.partial')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
        for name in sorted(marker['files']): archive.write(args.stage/name,name)
        archive.write(package/'package_manifest.json',(CORE/'package_manifest.json').as_posix())
        for name in OUTPUT_NAMES: archive.write(args.submission/name,'output/'+name)
        archive.write(args.submission/'submission_report.json',(CORE/'release_report.json').as_posix())
        archive.write(args.verification,'code/business_entity_resolution/packaged_verification.json')
    with zipfile.ZipFile(temporary) as archive:
        names = archive.namelist()
        if len(names)!=len(set(names)) or any(n.startswith(('dataset/','/')) or '..' in Path(n).parts for n in names):
            raise ValueError('Unsafe or duplicate archive members')
        if archive.testzip() is not None: raise ValueError('Archive integrity failed')
    temporary.replace(args.output)
    write_json(args.output.with_suffix(args.output.suffix+'.manifest.json'),{
        'status':'validated_offline_submission_archive', 'zip_sha256':digest(args.output),
        'bytes':args.output.stat().st_size, 'files':len(names),
        'selected_freeze_sha256':marker['selected_freeze_sha256'],
        'output_sha256':{n:report['files_sha256'][n] for n in OUTPUT_NAMES},
        'packaged_subset_verification_sha256':digest(args.verification),
    })
    print(json.dumps({'archive':str(args.output),'bytes':args.output.stat().st_size,'files':len(names)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare-fixture','stage','archive'])
    p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--asset-root',type=Path,help='Read-only overlay for exact frozen platform-specific assets')
    p.add_argument('--output',type=Path,required=True)
    for arg in ['freeze','fixture','stage','submission','verification','fixture_base','fixture_features',
                'fixture_scores','fixture_input_manifest','fine_head_manifest','old_head_manifest']:
        p.add_argument('--'+arg.replace('_','-'),type=Path)
    p.add_argument('--fixture-owners',type=int,default=100)
    a = p.parse_args()
    required={'prepare-fixture':['fixture_base','fixture_features','fixture_scores','fixture_input_manifest','fine_head_manifest','old_head_manifest'],
              'stage':['freeze','fixture'], 'archive':['stage','submission','verification']}[a.command]
    if any(getattr(a,k) is None for k in required): p.error('Missing required inputs: '+','.join(required))
    {'prepare-fixture':prepare_fixture,'stage':stage,'archive':archive}[a.command](a)


if __name__ == '__main__': main()
