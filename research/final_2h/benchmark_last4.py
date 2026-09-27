"""Historical, incomplete last-four benchmark draft; provenance preflight only.

This draft never emits runtime evidence. For the completed measurement driver,
use research/final_2h/benchmark_neural_last4.py with its sealed runtime assets.
"""
import json
import math
from pathlib import Path
import sys
from collections import Counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'research/sprint_6h'), str(ROOT / 'research/final_2h')]
import build_gated_submission as export
from score_neural_last4_heldout import scatter_corrections as scatter

CANDIDATE = 'neural_last4_fine_v1'
ADAPTER = Path('research/final_2h/neural_last4_fine_v1/adapter.txt')
FINE_HEAD = Path('models/final_2h/neural_last4_continue_v1/manifest.json')
OLD_HEAD = Path('models/sprint_6h/neural/frozen_head120k_v1/manifest.json')
REPORT = ADAPTER.with_name('report.json')
BASELINE_FREEZE = Path('research/sprint_6h/coordination/baseline_freeze/frozen.json')
RUNNER = Path('scripts/run_frozen_pipeline.py')


def verify_freeze(path):
    freeze = json.loads(Path(path).read_text())
    if (freeze.get('status') != 'candidate_frozen' or freeze.get('mode') != 'R2'
            or freeze.get('candidate') != CANDIDATE or not freeze.get('frozen_at')):
        raise ValueError('Requires new frozen last4 candidate (not NNv2)')
    required = [ADAPTER, REPORT, FINE_HEAD, OLD_HEAD]
    group = freeze.get('model_sha256') or {}
    for artifact in required:
        if group.get(artifact.as_posix()) != export.digest(ROOT / artifact):
            raise ValueError('Last4 adapter/head/report not bound to frozen candidate: ' + str(artifact))
    for name in ('code_sha256', 'model_sha256', 'schema_sha256', 'index_sha256', 'native_sha256'):
        pins = freeze.get(name)
        if not isinstance(pins, dict) or not pins:
            raise ValueError('Missing frozen provenance group: ' + name)
        for artifact, sha in pins.items():
            file = (ROOT / artifact).resolve()
            if not file.is_relative_to(ROOT) or not file.is_file() or sha != export.digest(file):
                raise ValueError('Frozen provenance mismatch: ' + name + ': ' + artifact)
    runner = freeze.get('original_runner') or {}
    if (runner.get('path') != RUNNER.as_posix()
            or runner.get('sha256') != export.digest(ROOT / RUNNER)
            or freeze['code_sha256'].get(RUNNER.as_posix()) != runner['sha256']
            or freeze['code_sha256'].get(Path(__file__).resolve().relative_to(ROOT).as_posix()) != export.digest(__file__)):
        raise ValueError('Frozen original runner or benchmark code differs')
    baseline = json.loads((ROOT / BASELINE_FREEZE).read_text())
    decision = {key: baseline[key] for key in ('threshold', 't_first', 't_rest')}
    if (freeze.get('parent_frozen_sha256') != export.digest(ROOT / BASELINE_FREEZE)
            or freeze.get('decision_config') != decision):
        raise ValueError('Frozen baseline decision config or parent differs')
    report = json.loads((ROOT / REPORT).read_text())
    if (report.get('model_sha256') != export.digest(ROOT / ADAPTER)
            or report.get('original_neural_head_sha256') != export.digest(ROOT / OLD_HEAD)
            or report.get('fine_head_sha256') != export.digest(ROOT / FINE_HEAD)
            or report.get('features') != __import__('score_neural_last4_heldout').FEATURES):
        raise ValueError('Frozen adapter report/head/schema lineage differs')
    return freeze, export.digest(path)


def verify_score(scores, input_manifest, head_manifest, reverse_feature_hash):
    scores = Path(scores)
    im = json.loads(Path(input_manifest).read_text())
    sm = json.loads((scores / 'manifest.json').read_text())
    if (im.get('status') != 'complete' or im.get('labels_read') is not False
            or im.get('source_split') != 'test'
            or sm.get('status') != 'complete' or sm.get('labels_read') is not False
            or sm.get('input_sha256') != im.get('input_sha256')
            or sm.get('rows') != im.get('rows')
            or sm.get('checkpoint_manifest_sha256') != export.digest(head_manifest)
            or sm.get('scores_sha256') != export.digest(scores / 'scores.jsonl')
            or im.get('feature_manifest_sha256') != reverse_feature_hash
            or not isinstance(sm.get('seconds'), (int, float)) or sm['seconds'] <= 0):
        raise ValueError('Sealed last4 score/input/feature provenance differs')
    return sm


def verify_old_score(scores, input_manifest, head_manifest):
    """Bind the old-head timing to this exact sealed, label-free test input."""
    scores = Path(scores)
    im = json.loads(Path(input_manifest).read_text())
    sm = json.loads((scores / 'manifest.json').read_text())
    seconds = sm.get('seconds')
    if (im.get('status') != 'complete' or im.get('labels_read') is not False
            or im.get('source_split') != 'test'
            or sm.get('status') != 'complete' or sm.get('labels_read') is not False
            or sm.get('input_sha256') != im.get('input_sha256')
            or not im.get('input_sha256')
            or sm.get('input_manifest_sha256') != export.digest(input_manifest)
            or sm.get('rows') != im.get('rows')
            or not isinstance(sm.get('rows'), int) or isinstance(sm['rows'], bool) or sm['rows'] <= 0
            or sm.get('head_manifest_sha256') != export.digest(head_manifest)
            or sm.get('scores_sha256') != export.digest(scores / 'scores.jsonl')
            or not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('Sealed old-head score/input/head provenance differs')
    return sm


def verify_plan(path):
    """Reject incomplete or incoherent projection metadata before any output."""
    plan = json.loads(Path(path).read_text())
    if not isinstance(plan, dict):
        raise ValueError('Benchmark plan must be a complete metadata object')
    positive_int = lambda value: isinstance(value, int) and not isinstance(value, bool) and value > 0
    blocks = plan.get('blocks')
    if (plan.get('status') != 'complete' or plan.get('labels_read') is not False
            or not isinstance(blocks, list) or len(blocks) != 2
            or any(not positive_int(plan.get(k)) for k in ('full_references', 'full_pairs', 'routed_pairs'))):
        raise ValueError('Benchmark plan must be complete and label-free with two actual blocks')
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError('Invalid benchmark block plan')
        countries = block.get('countries')
        if (not positive_int(block.get('references')) or not positive_int(block.get('pairs'))
                or block['pairs'] < block['references']
                or not isinstance(countries, dict) or set(countries) != {'india', 'us', 'france'}
                or any(not positive_int(n) for n in countries.values())
                or sum(countries.values()) != block['references']):
            raise ValueError('Benchmark block plan counts or country coverage differ')
    seconds = plan.get('full_route_setup_seconds')
    if (plan['full_pairs'] < plan['full_references']
            or plan['routed_pairs'] > plan['full_pairs']
            or sum(b['references'] for b in blocks) > plan['full_references']
            or sum(b['pairs'] for b in blocks) > plan['full_pairs']
            or not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
            or not math.isfinite(seconds) or seconds < 0):
        raise ValueError('Benchmark plan full graph counts or setup timing differ')
    return plan


def verify_block(block, plan, number, baseline_hash, graph_hash):
    """Verify the complete *block* claimant graph, never just routed pairs."""
    marker = json.loads((block / 'manifest.json').read_text())
    refs = json.loads((block / 'references.json').read_text())
    frame = pd.read_parquet(block / 'pairs.parquet')
    ids = [row['entity_id'] for row in refs]
    countries = dict(Counter(row['country'].strip().casefold() for row in refs))
    declared = plan['blocks'][number]
    if (plan.get('status') != 'complete' or plan.get('labels_read') is not False
            or len(plan.get('blocks', [])) != 2
            or marker.get('status') != 'complete' or marker.get('audit_labels_read') is not False
            or marker.get('verified_current_pipeline') is not True
            or marker.get('split') != 'test' or marker.get('role') != 'runtime_block'
            or marker.get('frozen_sha256') != baseline_hash
            or marker.get('routing_universe_pair_order_sha256') != graph_hash
            or marker.get('pairs_sha256') != export.digest(block / 'pairs.parquet')
            or marker.get('pairs') != len(frame) or declared.get('pairs') != len(frame)
            or declared.get('references') != len(refs) or marker.get('reference_ids') != ids
            or declared.get('countries') != countries
            or len(ids) != len(set(ids)) or set(countries) != {'india', 'us', 'france'}
            or not set(frame.source1_entity_id).issubset(ids)
            or frame.duplicated(export.PAIR_KEYS).any()
            or marker.get('pair_order_sha256') != export.pair_order_digest(frame)):
        raise ValueError('Actual full block claim graph, pair order or provenance differs')
    return frame, refs, countries


def benchmark(args):
    freeze, freeze_hash = verify_freeze(args.freeze)
    if args.output.exists():
        raise ValueError('Use unused versioned benchmark output')
    for number in (0, 1):
        block = args.blocks / f'block_{number}'
        for name in ('pairs.parquet', 'manifest.json', 'references.json',
                     'combined_neural/manifest.json', 'combined_neural/features.parquet',
                     'feature_timing.json'):
            if not (block / name).is_file():
                raise ValueError(f'Actual blocks unavailable: {block / name}')
    if not (args.blocks / 'plan.json').is_file():
        raise ValueError('Actual blocks plan unavailable')
    verify_plan(args.blocks / 'plan.json')
    raise NotImplementedError(
        'This historical draft implements provenance preflight only. '
        'Use research/final_2h/benchmark_neural_last4.py for the completed benchmark; '
        'no runtime evidence was produced.')
