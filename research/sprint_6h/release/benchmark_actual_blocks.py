#!/usr/bin/env python3
"""Measure frozen R1 release mechanics on exact actual country-stratified blocks.

Block generation/selection is separate. This consumes the gate worker's measured
blocks, never labels. Full target ID inputs are used for both validators. A fixed
empty-reference validator measurement separates target-universe loading from
reference-dependent processing. No complete test submission is generated here.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('release_builder', ROOT / 'scripts/build_gated_submission.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
GATE_STAGES = ('reads', 'lookups', 'normalization', 'gate_or_features', 'predict')
ALL_STAGES = GATE_STAGES + ('global_decision', 'diagnostic_write', 'tsv_export', 'strict_validator', 'official_validator')


def finite_seconds(values, required):
    if not set(required) <= set(values):
        raise ValueError('Missing measured runtime stages')
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('Non-finite/negative measured runtime seconds')
    return dict(values)


def block_path(evidence, base):
    return r.resolve_evidence(evidence, base)


def test_inputs(directory, source1, official_test):
    directory.mkdir()
    (directory / 'test_source1.tsv').symlink_to(source1.resolve())
    for number in (2, 3):
        (directory / f'test_source{number}.tsv').symlink_to((official_test / f'test_source{number}.tsv').resolve())
    return directory


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def benchmark(freeze_path, metadata_paths, official_test, copy_proof_path, output,
              processing_budget=7200, safety=.15, root=ROOT):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    freeze_path, official_test, copy_proof_path, output = map(Path, (freeze_path, official_test, copy_proof_path, output))
    if output.exists() or not math.isfinite(safety) or safety < .15 or not math.isfinite(processing_budget) or processing_budget <= 0:
        raise ValueError('New output and finite budget/safety >= 0.15 required')
    forbidden = [Path(root) / 'output' / name for name in ('submission_03', 'submission_04')]
    if any(output.resolve() == p.resolve() or p.resolve() in output.resolve().parents for p in forbidden):
        raise ValueError('Preserved baseline is not a benchmark destination')
    freeze = json.loads(freeze_path.read_text()); frozen_sha = r.digest(freeze_path)
    if freeze.get('status') != 'candidate_frozen' or freeze.get('mode') != 'R1' or not freeze.get('frozen_at'):
        raise ValueError('Actual runtime execution requires a frozen R1 candidate')
    pinned = freeze['code_sha256']
    required_code = [ROOT / 'scripts/build_gated_submission.py',
                     ROOT / 'code/business_entity_resolution/src/pipeline/export.py',
                     ROOT / 'code/business_entity_resolution/src/utils/organizer_validate_submission.py']
    if freeze['policy'] == 'ownership_v2':
        required_code.append(ROOT / 'scripts/decision_policy_v2.py')
    for path in required_code:
        key = str(path.relative_to(ROOT))
        if pinned.get(key, pinned.get(str(path))) != r.digest(path):
            raise ValueError('Runtime code differs from frozen implementation: ' + key)
    for path, sha in pinned.items():
        r.resolve_evidence({'path':path,'sha256':sha},root)
    proof = json.loads(copy_proof_path.read_text())
    if proof.get('status') != 'complete' or proof.get('copy_byte_parity') is not True or proof['baseline_frozen_sha256'] != freeze['parent_frozen_sha256']:
        raise ValueError('Complete baseline copy proof with matching lineage required')
    if len(metadata_paths) < 2:
        raise ValueError('At least two actual measured blocks required')
    output.mkdir(parents=True)
    # This measures loading both complete target ID universes without sampled
    # Source1/output processing. The full run pays that fixed cost once.
    empty = output / 'empty_source1.tsv'; empty.write_text('entity_id\tcountry\tbusiness_name\tbusiness_address\n')
    fixed_test = test_inputs(output / 'fixed_test', empty, official_test)
    fixed_output = output / 'fixed_validator'; fixed_output.mkdir()
    r.export_outputs([], pd.DataFrame(columns=r.PAIR_KEYS + r.SCORE_COLUMNS), np.zeros(0, dtype=bool), fixed_output)
    fixed_validation = r.validate_outputs(fixed_output, fixed_test)
    if fixed_validation['strict'] != fixed_validation['official'] or fixed_validation['strict'] != 'PASS':
        raise ValueError('Fixed target-ID validator preflight failed')
    blocks = []; sampled = set()
    for number, metadata_path in enumerate(map(Path, metadata_paths)):
        metadata = json.loads(metadata_path.read_text()); base = metadata_path.parent
        if metadata.get('candidate_frozen_sha256') != frozen_sha or metadata.get('measured') is not True:
            raise ValueError('Block does not belong to exact frozen measured candidate')
        stages = finite_seconds(metadata['seconds'], GATE_STAGES)
        start = time.perf_counter()
        source1 = block_path(metadata['source1'], base)
        score_path = block_path(metadata['scores'], base)
        gate_path = block_path(metadata['gate_table'], base)
        rows = r.read_source1(source1); frame = pd.read_parquet(score_path)
        r.validate_pairs(frame)
        ids = {row['entity_id'] for row in rows}
        if sampled.intersection(ids) or not set(frame.source1_entity_id) <= ids:
            raise ValueError('Blocks overlap or have unknown score references')
        sampled.update(ids)
        countries = {}
        for row in rows:
            country = row['country'].strip().casefold()
            countries[country] = countries.get(country, 0) + 1
        if set(countries) != {'france', 'india', 'us'} or any(v <= 0 for v in countries.values()):
            raise ValueError('Every actual block must include France, India, US')
        if not len(frame) or len(rows) != metadata['references'] or len(frame) != metadata['pairs'] or countries != metadata['country_counts']:
            raise ValueError('Actual block count/country mismatch')
        before = frame[r.SCORE_COLUMNS].copy()
        frame = r.attach_vetoes(frame, pd.read_parquet(gate_path), metadata['pair_order_sha256'])
        stages['reads'] += time.perf_counter() - start
        start = time.perf_counter()
        chosen, lost, details = r.make_decisions(frame, freeze, root)
        if not frame[r.SCORE_COLUMNS].equals(before):
            raise ValueError('Runtime gate changed base scores')
        stages['global_decision'] = time.perf_counter() - start
        destination = output / f'block_{number}'; destination.mkdir()
        start = time.perf_counter()
        statistics = r.export_outputs(rows, frame, chosen, destination)
        stages['tsv_export'] = time.perf_counter() - start
        start = time.perf_counter()
        diagnostics = frame.copy(); diagnostics['accepted'] = chosen; diagnostics['ownership_lost'] = lost
        diagnostics.to_parquet(destination / 'pair_decisions.parquet', index=False)
        high = frame.loc[frame.probability >= float(freeze['decision_config'].get('threshold', freeze['decision_config']['t_rest']))]
        collisions = high.groupby('candidate_entity_id', sort=False).source1_entity_id.nunique()
        histogram = statistics['candidate_histogram']; cumulative = 0; p95 = 0
        for count, frequency in sorted(histogram.items()):
            cumulative += frequency
            if cumulative >= .95 * len(rows): p95 = count; break
        stages['diagnostic_write'] = time.perf_counter() - start
        test = test_inputs(output / f'block_{number}_test', source1, official_test)
        validation = r.validate_outputs(destination, test)
        stages.update(validation['stage_seconds'])
        if validation['strict'] != validation['official'] or validation['strict'] != 'PASS':
            raise ValueError('Actual block validator failed')
        blocks.append({'references':len(rows), 'pairs':len(frame), 'country_stratified':True,
                       'country_counts':countries, 'seconds':finite_seconds(stages, ALL_STAGES),
                       'metadata_sha256':r.digest(metadata_path), 'pair_order_sha256':metadata['pair_order_sha256'],
                       'eligible_signal_rows':metadata['eligible_signal_rows'], 'skipped_signal_rows':metadata['skipped_signal_rows'],
                       'mean_candidates':len(frame)/len(rows), 'p95_candidates':p95, 'max_candidates':max(histogram, default=0),
                       'high_band_collision_targets':int((collisions > 1).sum()), 'decision':details,
                       'deep_frame_bytes':int(frame.memory_usage(deep=True).sum()), 'peak_rss_bytes':rss_bytes(),
                       'fixed_preflight_seconds':finite_seconds(metadata.get('fixed_preflight_seconds', {}), ()),
                       'validation':{'strict':'PASS','official':'PASS','id_checking':True}})
    full_pairs, full_refs = proof['pairs'], proof['entities']
    estimates = {}
    for stage in ALL_STAGES:
        if stage in ('strict_validator','official_validator'):
            fixed = fixed_validation['stage_seconds'][stage]
            variable = max(max(0.0, block['seconds'][stage] - fixed) / block['references'] for block in blocks) * full_refs
            estimates[stage] = fixed + variable
        else:
            estimates[stage] = max(block['seconds'][stage] / block['pairs'] for block in blocks) * full_pairs
            if stage == 'global_decision':
                # Account for sorting/grouping growth rather than assuming a
                # strictly linear full-universe arbitration cost.
                estimates[stage] *= max(math.log2(max(2,full_pairs)) / math.log2(max(2,block['pairs'])) for block in blocks)
    # Baseline proof measured the full sealed asset/chunk byte checks. Double
    # that warm-cache time, then add signal-state/lookup initialization once.
    estimates['fixed_seal_verification'] = 2 * proof['elapsed_seconds']
    estimates['fixed_gate_preflight'] = max(sum(block['fixed_preflight_seconds'].values()) for block in blocks)
    raw_eta = sum(estimates.values()); inflated_eta = raw_eta * (1 + safety)
    result = {'kind':'actual_frozen_r1_runtime', 'mode':'R1', 'measured':True,
              'synthetic_fixture_runtime':False, 'full_test_eta_supported':True,
              'candidate_frozen_sha256':frozen_sha, 'passed':inflated_eta <= processing_budget,
              'estimated_processing_seconds':inflated_eta, 'estimated_before_safety_seconds':raw_eta,
              'processing_budget_seconds':processing_budget, 'safety_margin_fraction':safety,
              'stage_full_test_estimates_seconds':estimates, 'blocks':blocks,
              'fixed_validator_stage_seconds':fixed_validation['stage_seconds'],
              'full_pairs':full_pairs, 'full_references':full_refs, 'runtime_workers':1,
              'peak_rss_bytes':rss_bytes(), 'copy_proof_sha256':r.digest(copy_proof_path),
              'benchmark_code_sha256':r.digest(__file__),
              'estimator':'Worst measured serial per-pair stage rate; full target-ID validator cost once plus worst per-reference excess; global decision n-log-n growth; double measured seal IO; all stages inflated by declared safety fraction.',
              'scope':'Actual runtime blocks contain their complete candidate populations. Full release still arbitrates over all full-test claimants. No labels used, no full-test release or portal upload.'}
    r.write_json(output / 'runtime.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--blocks', type=Path, nargs='+', required=True)
    parser.add_argument('--official-test-dir', type=Path, required=True)
    parser.add_argument('--copy-proof', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--processing-budget', type=float, default=7200)
    parser.add_argument('--safety-margin', type=float, default=.15)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.freeze, args.blocks, args.official_test_dir, args.copy_proof,
                               args.output, args.processing_budget, args.safety_margin), indent=2))
