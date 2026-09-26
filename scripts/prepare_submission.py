"""Freeze a verified model and its audited source snapshot for resumable inference."""
import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-dir', type=Path, required=True)
    p.add_argument('--submission-dir', type=Path, required=True)
    p.add_argument('--test-dir', type=Path, default=Path('dataset/test'))
    p.add_argument('--index-dir', type=Path, default=Path('models'))
    args = p.parse_args()
    artifact, submission = args.artifact_dir, args.submission_dir
    if submission.exists(): raise FileExistsError('Use a fresh submission directory; never overwrite checkpoints')
    report = json.loads((artifact / 'report.json').read_text())
    compatibility = json.loads((artifact / 'inference_compatibility.json').read_text())
    selection = json.loads((artifact / 'frozen_selection.json').read_text())
    if not compatibility['compatible']: raise ValueError('Saved model compatibility failed')
    if not report['audit']['accepted']: raise ValueError('Saved model did not pass its audit')
    if digest(artifact / 'model.txt') != selection['model_sha256']: raise ValueError('Model changed after audit')
    if digest(artifact / 'name_aliases.json') != selection['alias_sha256']: raise ValueError('Aliases changed after audit')
    if report['threshold'] != selection['threshold']: raise ValueError('Threshold changed after audit')
    if report['feature_version'] != compatibility['feature_version']: raise ValueError('Feature version changed')
    snapshot = artifact / 'source_snapshot'
    source_hashes = json.loads((artifact / 'source_snapshot_sha256.json').read_text())
    for rel, expected in source_hashes.items():
        if digest(snapshot / rel) != expected: raise ValueError('Audited source snapshot changed: ' + rel)
    indexes = {}
    for n, source in ((2, 'S2'), (3, 'S3')):
        raw = (args.test_dir / f'test_source{n}.tsv').resolve()
        path = (args.index_dir / f'index_test_{source}.sqlite').resolve()
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as conn:
            meta = json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
        fp = meta['fingerprint']
        stat = raw.stat()
        if not meta['complete'] or (fp['path'], fp['size'], fp['mtime_ns'], fp['source']) != (str(raw), stat.st_size, stat.st_mtime_ns, source):
            raise ValueError('Test index fingerprint mismatch: ' + source)
        indexes[source] = meta
    package = submission / 'code/business_entity_resolution'
    package.parent.mkdir(parents=True)
    shutil.copytree('code/business_entity_resolution', package, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Active inference modules must be the exact audited bytes, including older
    # harmless formatting revisions. Current entry points/tests remain available.
    for rel in source_hashes:
        if rel.startswith('code/business_entity_resolution/'):
            destination = submission / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot / rel, destination)
    model = submission / 'model'; model.mkdir()
    for filename in ('model.txt', 'report.json', 'name_aliases.json', 'feature_registry.json', 'MODEL_LICENSE.txt',
                     'reproduction.json', 'frozen_selection.json', 'audit_report.json', 'audit_protocol.json',
                     'inference_compatibility.json', 'feature_replay_check.json', 'threshold_sweep.json'):
        shutil.copyfile(artifact / filename, model / filename)
    compatibility.update(test_indexes=indexes, model_sha256=selection['model_sha256'],
        local_holdout_macro_f05=report['holdout']['macro_f05'], leaderboard_score=None,
        inference_source_policy='Exact audited Python modules restored from the saved source snapshot')
    write_json(submission / 'compatibility_report.json', compatibility)
    shutil.copyfile('scripts/create_submission.py', submission / 'create_submission.py')
    frozen = {str(path.relative_to(submission)): digest(path) for path in sorted(submission.rglob('*')) if path.is_file()}
    write_json(submission / 'frozen_assets_sha256.json', frozen)
    write_json(submission / 'runner_sha256.json', {'create_submission.py': digest(submission / 'create_submission.py')})
    write_json(submission / 'progress.json', {'status': 'prepared', 'ready_for_upload': False,
        'model': report['experiment'], 'feature_version': report['feature_version'], 'threshold': report['threshold'],
        'local_validation_macro_f05': report['holdout']['macro_f05'], 'leaderboard_score': None})
    print(json.dumps({'prepared': str(submission), 'model': report['experiment'], 'frozen_assets': len(frozen), 'threshold': report['threshold']}))


if __name__ == '__main__': main()
