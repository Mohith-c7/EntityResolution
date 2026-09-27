"""Standalone regression gates for the incomplete historical benchmark draft."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.final_2h import benchmark_last4 as bench
from research.final_2h.score_neural_last4_heldout import FEATURES


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def freeze_fixture(tmp_path, monkeypatch):
    """All model/index/runner seals are synthetic; no private assets are read."""
    source = Path(bench.__file__).read_bytes()
    monkeypatch.setattr(bench, 'ROOT', tmp_path)
    draft = tmp_path / 'research/final_2h/benchmark_last4.py'
    draft.parent.mkdir(parents=True); draft.write_bytes(source)
    monkeypatch.setattr(bench, '__file__', str(draft))
    for relative in (bench.ADAPTER, bench.FINE_HEAD, bench.OLD_HEAD, bench.RUNNER):
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('synthetic ' + relative.as_posix())
    decision = {'threshold': .83, 't_first': .70, 't_rest': .83}
    write(tmp_path / bench.BASELINE_FREEZE, decision)
    write(tmp_path / bench.REPORT, {
        'model_sha256': bench.export.digest(tmp_path / bench.ADAPTER),
        'original_neural_head_sha256': bench.export.digest(tmp_path / bench.OLD_HEAD),
        'fine_head_sha256': bench.export.digest(tmp_path / bench.FINE_HEAD), 'features': FEATURES})
    pins = lambda paths: {p.as_posix(): bench.export.digest(tmp_path / p) for p in paths}
    payload = {'status': 'candidate_frozen', 'frozen_at': '2026-09-27T00:00:00Z',
        'mode': 'R2', 'candidate': bench.CANDIDATE,
        'model_sha256': pins([bench.ADAPTER, bench.REPORT, bench.FINE_HEAD, bench.OLD_HEAD]),
        'code_sha256': pins([bench.RUNNER, Path('research/final_2h/benchmark_last4.py')]),
        'schema_sha256': pins([bench.REPORT]), 'index_sha256': pins([bench.OLD_HEAD]),
        'native_sha256': pins([bench.ADAPTER]),
        'original_runner': {'path': bench.RUNNER.as_posix(), 'sha256': bench.export.digest(tmp_path / bench.RUNNER)},
        'parent_frozen_sha256': bench.export.digest(tmp_path / bench.BASELINE_FREEZE),
        'decision_config': decision}
    path = write(tmp_path / 'freeze.json', payload)
    return path, payload


@pytest.fixture
def scores_fixture(tmp_path):
    inputs = write(tmp_path / 'input.json', {'status': 'complete', 'labels_read': False,
        'source_split': 'test', 'rows': 3, 'input_sha256': 'sealed-input',
        'feature_manifest_sha256': 'sealed-features'})
    head = write(tmp_path / 'head.json', {'synthetic': 'old'})
    other_head = write(tmp_path / 'other-head.json', {'synthetic': 'fine'})
    scores = tmp_path / 'scores'; scores.mkdir()
    (scores / 'scores.jsonl').write_text(''.join(json.dumps({'source1_entity_id': 'S1-'+str(i),
        'candidate_entity_id': 'S2-'+str(i), 'neural_probability': .8})+'\n' for i in range(3)))
    marker = {'status': 'complete', 'labels_read': False, 'rows': 3, 'seconds': .5,
        'input_sha256': 'sealed-input', 'input_manifest_sha256': bench.export.digest(inputs),
        'head_manifest_sha256': bench.export.digest(head),
        'checkpoint_manifest_sha256': bench.export.digest(other_head),
        'scores_sha256': bench.export.digest(scores / 'scores.jsonl')}
    write(scores / 'manifest.json', marker)
    return scores, inputs, head, other_head, marker


def test_requires_new_last4_freeze_not_old_nnv2_proof(tmp_path):
    old = write(tmp_path/'old.json', {'status': 'candidate_frozen', 'mode': 'R2',
        'candidate': 'neural_v2', 'frozen_at': '2026-09-27T00:00:00Z'})
    with pytest.raises(ValueError, match='last4|candidate'): bench.verify_freeze(old)


@pytest.mark.parametrize('tamper', ['adapter', 'report', 'runner', 'decision'])
def test_frozen_binding_rejects_changed_provenance(freeze_fixture, tamper):
    path, payload = freeze_fixture
    assert bench.verify_freeze(path)[1] == bench.export.digest(path)
    if tamper in ('adapter', 'report'):
        artifact = bench.ADAPTER if tamper == 'adapter' else bench.REPORT
        payload['model_sha256'][artifact.as_posix()] = '0'*64
    elif tamper == 'runner': payload['original_runner']['sha256'] = '0'*64
    else: payload['decision_config']['threshold'] = 2.
    write(path, payload)
    with pytest.raises(ValueError, match='adapter|head|report|runner|decision'): bench.verify_freeze(path)


def test_fine_score_accepts_synthetic_sealed_input_and_rejects_wrong_input(scores_fixture):
    scores, inputs, _, head, _ = scores_fixture
    assert bench.verify_score(scores, inputs, head, 'sealed-features')['rows'] == 3
    marker = json.loads(inputs.read_text()); marker['input_sha256'] = 'different-input'; write(inputs, marker)
    with pytest.raises(ValueError, match='score|input'): bench.verify_score(scores, inputs, head, 'sealed-features')


@pytest.mark.parametrize('tamper', ['head', 'input_manifest', 'score_bytes', 'rows', 'nonfinite_seconds'])
def test_old_head_runtime_score_requires_exact_seals(scores_fixture, tamper):
    scores, inputs, head, other_head, marker = scores_fixture
    assert bench.verify_old_score(scores, inputs, head)['rows'] == 3
    if tamper == 'head': head = other_head
    elif tamper == 'input_manifest':
        value = json.loads(inputs.read_text()); value['role'] = 'changed'; write(inputs, value)
    elif tamper == 'score_bytes': (scores/'scores.jsonl').write_text('tampered\n')
    elif tamper == 'rows': marker['rows'] = 2; write(scores/'manifest.json', marker)
    else: marker['seconds'] = float('nan'); write(scores/'manifest.json', marker)
    with pytest.raises(ValueError, match='head|input|score'): bench.verify_old_score(scores, inputs, head)


def complete_plan():
    return {'status': 'complete', 'labels_read': False,
        'blocks': [{'references': 3, 'pairs': 3, 'countries': {'india': 1, 'us': 1, 'france': 1}} for _ in range(2)],
        'full_references': 100, 'full_pairs': 400, 'routed_pairs': 7, 'full_route_setup_seconds': 1.}


def block_placeholders(blocks):
    for number in (0, 1):
        block = blocks/f'block_{number}'; (block/'combined_neural').mkdir(parents=True)
        for name in ('pairs.parquet', 'manifest.json', 'references.json', 'combined_neural/manifest.json',
                     'combined_neural/features.parquet', 'feature_timing.json'): (block/name).touch()


def test_missing_actual_blocks_cannot_create_runtime_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'verify_freeze', lambda _: ({'candidate': bench.CANDIDATE}, 'new-hash'))
    output = tmp_path/'result'
    with pytest.raises(ValueError, match='block_0|actual blocks'):
        bench.benchmark(SimpleNamespace(freeze=tmp_path/'freeze', blocks=tmp_path/'missing', output=output))
    assert not output.exists()


@pytest.mark.parametrize('tamper', ['incomplete', 'country_count', 'full_pairs', 'nonfinite_timing'])
def test_invalid_plan_cannot_create_report(tmp_path, monkeypatch, tamper):
    monkeypatch.setattr(bench, 'verify_freeze', lambda _: ({'candidate': bench.CANDIDATE}, 'new-hash'))
    blocks=tmp_path/'blocks'; block_placeholders(blocks); plan=complete_plan()
    if tamper == 'incomplete': plan['status'] = 'incomplete'
    elif tamper == 'country_count': plan['blocks'][0]['countries']['us'] = 2
    elif tamper == 'full_pairs': plan['full_pairs'] = 3
    else: plan['full_route_setup_seconds'] = float('inf')
    write(blocks/'plan.json', plan); output=tmp_path/'unused'
    with pytest.raises(ValueError, match='plan|complete'):
        bench.benchmark(SimpleNamespace(freeze=tmp_path/'freeze', blocks=blocks, output=output))
    assert not output.exists()


def test_valid_preflight_explicitly_rejects_unimplemented_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'verify_freeze', lambda _: ({'candidate': bench.CANDIDATE}, 'new-hash'))
    blocks=tmp_path/'blocks'; block_placeholders(blocks); write(blocks/'plan.json', complete_plan())
    output=tmp_path/'unused'
    with pytest.raises(NotImplementedError, match='benchmark_neural_last4.py'):
        bench.benchmark(SimpleNamespace(freeze=tmp_path/'freeze', blocks=blocks, output=output))
    assert not output.exists()


def test_block_rejects_truncated_claim_graph_before_model_or_output(tmp_path):
    block=tmp_path/'block_0'; block.mkdir()
    refs=[{'entity_id':'S1-'+str(i), 'country':country} for i,country in enumerate(['India','US','France'])]
    write(block/'references.json', refs)
    frame=pd.DataFrame({'source1_entity_id':[r['entity_id'] for r in refs], 'candidate_entity_id':['S2-'+str(i) for i in range(3)],
        'candidate_order':[0]*3, 'first_stage':[.8]*3, 'probability':[.8]*3})
    frame.to_parquet(block/'pairs.parquet', index=False)
    write(block/'manifest.json', {'status':'complete','split':'test','role':'runtime_block','audit_labels_read':False,
        'verified_current_pipeline':True,'reference_ids':[r['entity_id'] for r in refs], 'pairs':4,
        'pairs_sha256':bench.export.digest(block/'pairs.parquet'),'pair_order_sha256':bench.export.pair_order_digest(frame),
        'frozen_sha256':'baseline-hash','routing_universe_pair_order_sha256':'full-graph-hash'})
    plan=complete_plan(); plan['blocks'][0]['pairs']=4
    with pytest.raises(ValueError, match='full|claim|pair|block'):
        bench.verify_block(block, plan, 0, 'baseline-hash', 'full-graph-hash')


def test_scatter_changes_only_keyed_route_and_anchors_once():
    frame=pd.DataFrame({'source1_entity_id':['S1-1','S1-1'],'candidate_entity_id':['S2-2','S2-1'],
        'first_stage':[.4,.8],'probability':[.2,.8],'candidate_order':[1,0]})
    routed=pd.DataFrame({'source1_entity_id':['S1-1'],'candidate_entity_id':['S2-1'],
        'first_stage':[.8],'probability':[.8],'candidate_rank':[1]})
    result=bench.scatter(frame,routed,np.array([np.log(2.)]))
    assert result.probability.iloc[0]==.2
    assert result.probability.iloc[1]==pytest.approx(8./9.)
    assert result[bench.export.PAIR_KEYS+['candidate_order']].equals(frame[bench.export.PAIR_KEYS+['candidate_order']])
    with pytest.raises(ValueError,match='anchor|rank|pair'):bench.scatter(frame,routed.assign(candidate_rank=2),np.array([0.]))
