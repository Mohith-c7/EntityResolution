"""CPU-only driver checks; audited neural executables are mocked, not changed."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def setup_driver(tmp_path, monkeypatch, *, corrupt_old_head=False, old_timecap=False):
    spec = importlib.util.spec_from_file_location('full_neural_driver', ROOT / 'research/final_2h/score_full_test_neural.py')
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    raw = inputs / 'pairs.jsonl'
    rows = [{'source1_entity_id': f'S1-{i}', 'candidate_entity_id': f'S2-{i}',
        'text_left': 'provided left', 'text_right': 'provided right'} for i in range(1124)]
    raw.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    full = {'status': 'complete', 'labels_read': False, 'source_split': 'test',
        'rows': len(rows), 'input_sha256': driver.sha(raw), 'feature_manifest_sha256': 'sealed-route'}
    (inputs / 'manifest.json').write_text(json.dumps(full))
    for relative in ['models/sprint_6h/neural/frozen_head120k_v1/manifest.json',
                     'models/final_2h/neural_last4_continue_v1/manifest.json']:
        p = tmp_path / relative; p.parent.mkdir(parents=True); p.write_text('{}\n')
    for relative in ['research/sprint_6h/neural_frozen_head.py', 'research/final_2h/neural_layers.py']:
        p = tmp_path / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# immutable scorer fixture\n')
    calls = []

    def mock_score(command, *, cwd, check):
        assert cwd == tmp_path and check is True
        input_path = Path(command[command.index('--input') + 1])
        out = Path(command[command.index('--output') + 1]); out.mkdir()
        old = command[1].endswith('neural_frozen_head.py')
        calls.append(('old' if old else 'fine', input_path))
        if old_timecap and old:
            (out / 'status.json').write_text(json.dumps({'status': 'feature_time_limit_rejected'}))
            return
        manifest_arg = '--manifest' if old else '--input-manifest'
        manifest_path = Path(command[command.index(manifest_arg) + 1])
        part = [json.loads(line) for line in input_path.read_text().splitlines()]
        scores = [{'source1_entity_id': r['source1_entity_id'], 'candidate_entity_id': r['candidate_entity_id'],
                   'neural_probability': .8 if old else .9} for r in part]
        score_path = out / 'scores.jsonl'; score_path.write_text(''.join(json.dumps(r) + '\n' for r in scores))
        head = tmp_path / ('models/sprint_6h/neural/frozen_head120k_v1' if old else 'models/final_2h/neural_last4_continue_v1')
        marker = {'status': 'complete', 'labels_read': False, 'rows': len(part),
            'head_manifest_sha256' if old else 'checkpoint_manifest_sha256': driver.sha(head / 'manifest.json'),
            'input_sha256': driver.sha(input_path), 'input_manifest_sha256': driver.sha(manifest_path),
            'code_sha256': driver.sha(command[1]), 'scores_sha256': driver.sha(score_path)}
        if old and corrupt_old_head:
            marker['head_manifest_sha256'] = 'wrong-head'
        (out / 'manifest.json').write_text(json.dumps(marker))

    monkeypatch.setattr(driver.subprocess, 'run', mock_score)
    monkeypatch.setattr(sys, 'argv', ['score_full_test_neural', '--inputs', str(inputs),
        '--parts', str(tmp_path / 'parts'), '--old-output', str(tmp_path / 'old'),
        '--fine-output', str(tmp_path / 'fine'), '--part-rows', '512'])
    return driver, inputs, rows, calls


def test_partition_boundaries_count_and_aggregate_provenance(tmp_path, monkeypatch):
    driver, inputs, original, calls = setup_driver(tmp_path, monkeypatch)
    driver.main()
    assert [kind for kind, _ in calls] == ['old', 'fine'] * 3
    parts = [tmp_path / 'parts' / str(i) for i in range(3)]
    markers = [json.loads((p / 'manifest.json').read_text()) for p in parts]
    assert [m['rows'] for m in markers] == [512, 512, 100]
    assert all(m['batch_boundary'] == 512 and m['execution_subset'] for m in markers)
    assert all(m['parent_input_manifest_sha256'] == driver.sha(inputs / 'manifest.json') for m in markers)
    assert all(m['parent_input_sha256'] == driver.sha(inputs / 'pairs.jsonl') for m in markers)
    for kind in ['old', 'fine']:
        target = tmp_path / kind
        scores = [json.loads(line) for line in (target / 'scores.jsonl').read_text().splitlines()]
        assert [(r['source1_entity_id'], r['candidate_entity_id']) for r in scores] == [
            (r['source1_entity_id'], r['candidate_entity_id']) for r in original]
        marker = json.loads((target / 'manifest.json').read_text())
        assert marker['rows'] == 1124 and marker['labels_read'] is False
        assert marker['input_sha256'] == driver.sha(inputs / 'pairs.jsonl')
        assert marker['input_manifest_sha256'] == driver.sha(inputs / 'manifest.json')
        assert marker['scores_sha256'] == driver.sha(target / 'scores.jsonl')
        assert marker['partition_manifest_sha256'] == [driver.sha(p / kind / 'manifest.json') for p in parts]
        assert marker['execution_driver_sha256'] == driver.sha(driver.__file__)


def test_aggregate_rejects_wrong_partition_head(tmp_path, monkeypatch):
    driver, _, _, _ = setup_driver(tmp_path, monkeypatch, corrupt_old_head=True)
    with pytest.raises(ValueError, match='seal|source|partition|head'):
        driver.main()


def test_timecap_return_zero_stops_before_fine_job(tmp_path, monkeypatch):
    driver, _, _, calls = setup_driver(tmp_path, monkeypatch, old_timecap=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        driver.main()
    assert len(calls) == 1 and calls[0][0] == 'old'


@pytest.mark.parametrize('field,value', [
    ('rows', 511), ('code_sha256', 'changed-scorer'),
    ('scores_sha256', 'changed-score-bytes'), ('input_sha256', 'foreign-input'),
    ('input_manifest_sha256', 'foreign-manifest'),
    ('head_manifest_sha256', 'foreign-head'), ('status', 'time_limit_rejected'),
    ('labels_read', True),
])
def test_partition_provenance_rejects_changed_count_or_source(tmp_path, monkeypatch, field, value):
    driver, _, _, _ = setup_driver(tmp_path, monkeypatch)
    driver.main()
    part = tmp_path / 'parts/0'; marker_path = part / 'old/manifest.json'
    marker = json.loads(marker_path.read_text()); marker[field] = value
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='provenance'):
        driver.verify_part(part / 'old', tmp_path / 'models/sprint_6h/neural/frozen_head120k_v1',
            'head_manifest_sha256', tmp_path / 'research/sprint_6h/neural_frozen_head.py',
            part / 'pairs.jsonl', part / 'manifest.json', 512)
