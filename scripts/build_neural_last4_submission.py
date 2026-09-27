#!/usr/bin/env python3
"""Export the frozen four-layer, anchored 38-feature candidate.

Reuses the complete verified submission_04 score graph. Sparse evidence cannot
change candidate membership, order, the fixed global route, or unrouted scores.
No labels are read; retrieval, feature generation and neural scoring precede this
command. The original decision function runs once on the complete graph.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'research/sprint_6h')]
import build_gated_submission as reuse
import neural_adapter as nn

KEYS = reuse.PAIR_KEYS
FEATURES = [*nn.FEATURES, 'neural_fine_probability']
CANDIDATE = 'neural_last4_fine_v1'
ADAPTER_SHA = '625268050ec1774ec2eb1c0c8d38b89bf13c284d3e7474b564bb8264eec0dac1'
FINE_HEAD_SHA = '60bca75bf41628f43eff487f8bdbb9b2f2f5ad54967af467807d7bc55417083c'
OLD_HEAD_SHA = 'f2b75c8a1ca5bcb7293b893c61791b619c02d5d7f38e75c7da197423b87dfd94'
ROUTED_ROWS = 801380
TEST_OWNERS = 1732544


def assemble_fine(frame, values):
    """Exact-key sparse join, independent of score file ordering."""
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError('Duplicate/missing old37 pair keys')
    if set(values.columns) != {*KEYS, 'neural_probability'}:
        raise ValueError('Fine scores must contain only pair keys and probability')
    if (values[KEYS].isna().any().any() or values.duplicated(KEYS).any()
            or not np.isfinite(values.neural_probability.to_numpy()).all()
            or not values.neural_probability.between(0, 1).all()):
        raise ValueError('Invalid fine score keys/probabilities')
    left = pd.MultiIndex.from_frame(frame[KEYS])
    right = pd.MultiIndex.from_frame(values[KEYS])
    positions = right.get_indexer(left)
    if len(frame) != len(values) or (positions < 0).any():
        raise ValueError('Fine scores must exactly cover old37 route keys')
    result = frame.copy(deep=False)
    result['neural_fine_probability'] = values.neural_probability.to_numpy()[positions]
    if not np.isfinite(result[FEATURES].to_numpy(dtype=np.float32)).all():
        raise ValueError('Nonfinite 38-feature input')
    return result


def distinct_route_pairs(edges, features, *, expected_rows=ROUTED_ROWS):
    """Validate peer edges, then compare their distinct scored-pair universe.

    Two different predicted peers may support one routed candidate. Their
    triplets must be unique, but the scored pair is represented exactly once.
    """
    triplets = [*KEYS, 'peer_entity_id']
    if (not set(triplets) <= set(edges) or edges[triplets].isna().any().any()
            or edges.duplicated(triplets).any()):
        raise ValueError('Duplicate/missing global route peer triplets')
    for column in triplets:
        if not edges[column].map(lambda value: isinstance(value, str) and bool(value) and value == value.strip()).all():
            raise ValueError('Malformed global route peer triplets')
    if (not edges.source1_entity_id.str.startswith('S1-').all()
            or not edges.candidate_entity_id.str.startswith(('S2-', 'S3-')).all()
            or not edges.peer_entity_id.str.startswith(('S2-', 'S3-')).all()
            or (edges.candidate_entity_id == edges.peer_entity_id).any()
            or (edges.groupby('source1_entity_id').peer_entity_id.nunique() > nn.sibling.RouteConfig().seeds).any()
            or (edges.groupby('source1_entity_id').candidate_entity_id.nunique() > nn.sibling.RouteConfig().candidates).any()
            or edges.source1_entity_id.nunique() > int(nn.sibling.RouteConfig().fraction * TEST_OWNERS)):
        raise ValueError('Global route peer edges violate frozen seed policy')
    routed = edges[KEYS].drop_duplicates(ignore_index=True)
    if (len(routed) != expected_rows or len(features) != expected_rows
            or features[KEYS].isna().any().any() or features.duplicated(KEYS).any()
            or not pd.MultiIndex.from_frame(features[KEYS]).isin(pd.MultiIndex.from_frame(routed[KEYS])).all()):
        raise ValueError('Evidence must exactly cover distinct global route pair keys')
    return routed


def route_positions(base, features, chunk_rows=2000000, base_verified=False):
    """Index only routed owners, retaining their complete candidate sets.

    Owner filtering can span chunk boundaries. Ranking the union has exactly
    the same tie semantics as ranking the full graph, without a full-graph sort.
    The caller may skip duplicate validation only after load_verified_chunks.
    """
    if chunk_rows <= 0:
        raise ValueError('Require positive scatter chunk size')
    if (features[KEYS].isna().any().any() or features.duplicated(KEYS).any()
            or (not base_verified and base.duplicated(KEYS).any())):
        raise ValueError('Duplicate/missing pair keys')
    owners = set(features.source1_entity_id)
    chunks = []
    for start in range(0, len(base), chunk_rows):
        mask = base.source1_entity_id.iloc[start:start + chunk_rows].isin(owners).to_numpy()
        chunks.append(np.flatnonzero(mask) + start)
    original = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    claims = base.iloc[original][[*KEYS, 'probability', 'first_stage']].reset_index(drop=True)
    positions = pd.MultiIndex.from_frame(claims[KEYS]).get_indexer(pd.MultiIndex.from_frame(features[KEYS]))
    if (positions < 0).any():
        raise ValueError('Foreign routed pair')
    if (not np.array_equal(claims.probability.to_numpy()[positions], features.probability.to_numpy())
            or not np.array_equal(claims.first_stage.to_numpy()[positions], features.first_stage.to_numpy())
            or not np.array_equal(nn.reverse.frozen_p2_rank(claims)[positions], features.candidate_rank.to_numpy())):
        raise ValueError('Frozen p2/first-stage anchor or full-candidate rank changed')
    return original[positions]


def scatter_corrections(base, features, correction, *, chunk_rows=2000000, base_verified=False):
    """Equivalent to heldout scatter; no full-graph MultiIndex or rank sort."""
    correction = np.asarray(correction, dtype=np.float64)
    if correction.shape != (len(features),) or not np.isfinite(correction).all():
        raise ValueError('Nonfinite/incomplete correction')
    positions = route_positions(base, features, chunk_rows, base_verified)
    original = base.probability.to_numpy()
    changed = original.copy()
    changed[positions] = nn.reverse.corrected_probability(original[positions], correction)
    if not np.isfinite(changed).all() or ((changed < 0) | (changed > 1)).any():
        raise ValueError('Invalid corrected probability')
    # Only keyed positions were assigned; original remains untouched.
    result = base.copy(deep=False)
    result['probability_original'] = original
    result['probability'] = changed
    if not result[[*KEYS, 'candidate_order']].equals(base[[*KEYS, 'candidate_order']]):
        raise ValueError('Candidate membership/order changed')
    return result


def apply_adapter(base, features, model, *, threads=8, chunk_rows=2000000, base_verified=False):
    if model.feature_name() != FEATURES or model.num_feature() != len(FEATURES):
        raise ValueError('Native adapter feature names/order differs')
    inputs = features[FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(inputs).all():
        raise ValueError('Nonfinite 38-feature input')
    # Native output is a residual margin. Frozen p2 enters exactly once here.
    correction = model.predict(inputs, raw_score=True, num_threads=threads)
    return scatter_corrections(base, features, correction, chunk_rows=chunk_rows, base_verified=base_verified)


def verify_freeze(path):
    frozen = json.loads(path.read_text())
    if (frozen.get('status') != 'candidate_frozen' or frozen.get('mode') != 'R2'
            or frozen.get('candidate') != CANDIDATE or not frozen.get('frozen_at')):
        raise ValueError('Requires frozen last4 anchored candidate')
    for group in ['code_sha256', 'model_sha256', 'schema_sha256', 'index_sha256', 'native_sha256']:
        pins = frozen.get(group)
        if not isinstance(pins, dict) or not pins:
            raise ValueError('Missing immutable evidence group: ' + group)
        for file, digest in pins.items():
            reuse.resolve_evidence({'path': file, 'sha256': digest}, ROOT)
    own_path = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if frozen['code_sha256'].get(own_path) != reuse.digest(__file__):
        raise ValueError('Exporter differs from frozen implementation')
    for module in [reuse, nn, nn.reverse, nn.sibling]:
        file = Path(module.__file__).resolve().relative_to(ROOT).as_posix()
        if frozen['code_sha256'].get(file) != reuse.digest(module.__file__):
            raise ValueError('Exporter dependency is not frozen: ' + file)
    return frozen


def verify_model(args, frozen, reference_ids, feature_manifest):
    report_path = args.model / 'report.json'
    adapter_path = args.model / 'adapter.txt'
    report = json.loads(report_path.read_text())
    old_head = json.loads(args.old_head_manifest.read_text())
    fine_head = json.loads(args.fine_head_manifest.read_text())
    for path in [report_path, adapter_path, args.old_head_manifest, args.fine_head_manifest]:
        name = path.resolve().relative_to(ROOT).as_posix()
        if frozen['model_sha256'].get(name) != reuse.digest(path):
            raise ValueError('CLI adapter/report/head is not frozen: ' + name)
    if (reuse.digest(adapter_path) != ADAPTER_SHA or report.get('model_sha256') != ADAPTER_SHA
            or reuse.digest(args.old_head_manifest) != OLD_HEAD_SHA
            or reuse.digest(args.fine_head_manifest) != FINE_HEAD_SHA
            or report.get('original_neural_head_sha256') != OLD_HEAD_SHA
            or report.get('fine_head_sha256') != FINE_HEAD_SHA
            or feature_manifest.get('neural_head_manifest_sha256') != OLD_HEAD_SHA
            or report.get('features') != FEATURES or report.get('mode') not in (None, 'anchored_neural')
            or report.get('calibration_logit_shift', 0) != 0
            or report.get('peer_head_sha256') is not None
            or report.get('source_sha256') != reuse.digest(ROOT / 'research/final_2h/neural_peer.py')):
        raise ValueError('Selected anchored adapter, feature schema or neural lineage differs')
    if old_head.get('status') != 'complete' or fine_head.get('status') != 'complete':
        raise ValueError('Incomplete neural head')
    if (fine_head.get('trainable_encoder_layers') != 4 or fine_head.get('initializer_epochs') != 3
            or fine_head.get('completed_epochs') != 1
            or any(fine_head.get(k) is not False for k in ['audit_labels_read', 'development_labels_read', 'test_records_read'])):
        raise ValueError('Four-layer outer-training lineage differs')
    fitted = set(report['fit_owners']) | set(old_head['training_owners']) | set(fine_head['training_owners'])
    if set(reference_ids) & fitted:
        raise ValueError('A fitted owner entered submission universe')
    return report


def load_fine(args, features, fm, reuse_manifest):
    im = json.loads(args.fine_input_manifest.read_text())
    score_manifest_path = args.fine_scores / 'manifest.json'
    sm = json.loads(score_manifest_path.read_text())
    input_path = args.fine_input_manifest.parent / 'pairs.jsonl'
    score_path = args.fine_scores / 'scores.jsonl'
    if (im.get('status') != 'complete' or sm.get('status') != 'complete'
            or im.get('labels_read') is not False or sm.get('labels_read') is not False
            or im.get('source_split') != 'test'
            or im.get('feature_manifest_sha256') != fm.get('reverse_feature_manifest_sha256')
            or im.get('input_sha256') != reuse.digest(input_path)
            or sm.get('input_sha256') != im.get('input_sha256')
            or sm.get('checkpoint_manifest_sha256') != FINE_HEAD_SHA
            or sm.get('scores_sha256') != reuse.digest(score_path)
            or sm.get('rows') != len(features) or im.get('rows') != len(features)):
        raise ValueError('Fine score/input/route provenance differs')
    if ('input_manifest_sha256' in sm
            and sm['input_manifest_sha256'] != reuse.digest(args.fine_input_manifest)):
        raise ValueError('Fine score input manifest differs')
    sources = {f'S{i}': reuse_manifest['test_files'][f'test_source{i}.tsv']['sha256'] for i in (1, 2, 3)}
    if im.get('source_tsv_sha256') != sources:
        raise ValueError('Fine input used different raw test sources')
    values = pd.read_json(score_path, lines=True, dtype={k: str for k in KEYS})
    return assemble_fine(features, values), sm


def build(args):
    started = time.perf_counter()
    stages = {}
    if args.threads <= 0 or args.scatter_chunk_rows <= 0:
        raise ValueError('Require positive threads/chunk size')
    for name in ['submission_03', 'submission_04']:
        preserved = (ROOT / 'output' / name).resolve()
        if args.output.resolve() == preserved or preserved in args.output.resolve().parents:
            raise ValueError('Preserved baseline cannot be a destination')
    if args.output.exists():
        raise ValueError('Require unused versioned output')
    frozen = verify_freeze(args.freeze)
    manifest = json.loads(args.reuse_manifest.read_text())
    if manifest['baseline_frozen']['sha256'] != frozen['parent_frozen_sha256']:
        raise ValueError('Baseline freeze differs')
    baseline_path = reuse.resolve_evidence(manifest['baseline_frozen'], ROOT)
    baseline = json.loads(baseline_path.read_text())
    if frozen['decision_config'] != {k: baseline[k] for k in ['threshold', 't_first', 't_rest']}:
        raise ValueError('Original decision thresholds changed')
    t = time.perf_counter()
    rows, base = reuse.load_verified_chunks(manifest, args.test_dir)
    stages['verified_base_load'] = time.perf_counter() - t
    if len(rows) != TEST_OWNERS:
        raise ValueError('Incomplete full test owner universe')
    t = time.perf_counter()
    features, fm = nn.load_features(args.features)
    if (fm.get('input_pair_order_sha256') != manifest['pair_order_sha256']
            or fm.get('split') != 'test' or fm.get('labels_read') is not False
            or len(features) != ROUTED_ROWS
            or fm.get('route_plan_sha256') != reuse.digest(args.route_manifest)):
        raise ValueError('Old37 features are not sealed to the fixed global test route')
    route = json.loads(args.route_manifest.read_text())
    route_path = reuse.resolve_evidence({'path': str(args.route_manifest.parent / 'route.parquet'),
                                        'sha256': route['route_sha256']}, ROOT)
    if (route.get('status') != 'complete' or route.get('labels_read', False) is not False
            or route.get('global_pair_order_sha256') != manifest['pair_order_sha256']
            or route.get('global_references') != len(rows)
            or route.get('route') != nn.sibling.asdict(nn.sibling.RouteConfig())):
        raise ValueError('Routing policy or full graph changed')
    edges = pd.read_parquet(route_path, columns=[*KEYS, 'peer_entity_id'])
    routed = distinct_route_pairs(edges, features)
    report = verify_model(args, frozen, [row['entity_id'] for row in rows], fm)
    combined, score_manifest = load_fine(args, features, fm, manifest)
    stages['sparse_evidence_verification_join'] = time.perf_counter() - t
    import lightgbm as lgb
    t = time.perf_counter()
    frame = apply_adapter(base, combined, lgb.Booster(model_file=str(args.model / 'adapter.txt')),
                          threads=args.threads, chunk_rows=args.scatter_chunk_rows, base_verified=True)
    stages['native_adapter_and_keyed_scatter'] = time.perf_counter() - t
    runner = reuse.resolve_evidence(frozen['original_runner'], ROOT)
    runner_name = runner.resolve().relative_to(ROOT).as_posix()
    if frozen['code_sha256'].get(runner_name) != reuse.digest(runner):
        raise ValueError('Original decision implementation is not frozen')
    t = time.perf_counter()
    chosen, lost = reuse.load_original_decide(runner)(frame, frame, frozen['decision_config'])
    chosen, lost = np.asarray(chosen, dtype=bool), np.asarray(lost, dtype=bool)
    if len(chosen) != len(frame) or len(lost) != len(frame):
        raise ValueError('Global decision mask alignment changed')
    stages['original_global_decision'] = time.perf_counter() - t
    args.output.mkdir(parents=True)
    shutil.copy2(args.freeze, args.output / 'frozen.json')
    t = time.perf_counter()
    statistics = reuse.export_outputs(rows, frame, chosen, args.output)
    stages['tsv_export'] = time.perf_counter() - t
    # The exact original candidate export must be reproduced, including order.
    baseline_report = json.loads(reuse.resolve_evidence(manifest['baseline_report'], ROOT).read_text())
    candidate_sha = reuse.digest(args.output / 'candidate_pairs.tsv')
    expected_candidate = baseline_report.get('output_sha256', baseline_report.get('files_sha256', {})).get('candidate_pairs.tsv')
    if not expected_candidate or candidate_sha != expected_candidate:
        raise ValueError('Candidate TSV differs from verified submission_04')
    t = time.perf_counter()
    validation = reuse.validate_outputs(args.output, args.test_dir)
    stages['validators'] = time.perf_counter() - t
    result = {'status': 'complete' if validation['strict'] == validation['official'] == 'PASS' else 'validation_failed',
        'mode': 'R2', 'candidate': CANDIDATE, 'labels_read': False, 'audit_labels_read': False,
        'candidate_frozen_sha256': reuse.digest(args.freeze), 'entities': len(rows),
        'scored_candidates': len(frame), 'auxiliary_scored_pairs': len(combined),
        'predicted_links': int(chosen.sum()), 'ownership_lost_pairs': int(lost.sum()),
        'candidate_pair_order_sha256': manifest['pair_order_sha256'], 'candidate_tsv_identical_to_submission_04': True,
        'statistics': statistics, 'validation': validation, 'stage_seconds': stages,
        'seconds': time.perf_counter() - started, 'source_sha256': reuse.digest(__file__),
        'evidence_sha256': {'reuse_manifest': reuse.digest(args.reuse_manifest), 'adapter': report['model_sha256'],
            'adapter_report': reuse.digest(args.model / 'report.json'), 'features': fm['features_sha256'],
            'features_manifest': reuse.digest(args.features / 'manifest.json'), 'route_manifest': reuse.digest(args.route_manifest),
            'fine_scores': score_manifest['scores_sha256'], 'fine_scores_manifest': reuse.digest(args.fine_scores / 'manifest.json'),
            'fine_input_manifest': reuse.digest(args.fine_input_manifest), 'old_head_manifest': OLD_HEAD_SHA,
            'fine_head_manifest': FINE_HEAD_SHA},
        'files_sha256': {n: reuse.digest(args.output / n) for n in ['matching_results.tsv', 'candidate_pairs.tsv']}}
    reuse.write_json(args.output / 'submission_report.json', result)
    if result['status'] != 'complete':
        raise ValueError('Submission failed validation')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['freeze', 'reuse_manifest', 'features', 'fine_scores', 'fine_input_manifest',
                 'fine_head_manifest', 'old_head_manifest', 'model', 'route_manifest', 'test_dir', 'output']:
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--scatter-chunk-rows', type=int, default=2000000)
    print(json.dumps(build(parser.parse_args()), indent=2))
