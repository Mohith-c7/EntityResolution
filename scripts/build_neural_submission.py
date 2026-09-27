"""Export a frozen R2 challenger from verified base chunks and keyed evidence.

This command never retrieves a different candidate set or edits submission_04.
Neural and reverse evidence must already be sealed against the complete source
score graph. The saved native adapter is evaluated here, then all ownership
decisions are recomputed globally before either TSV is written.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/sprint_6h'))
import neural_adapter as nn
import build_gated_submission as reuse


def apply_adapter(base, features, model):
    """Join by IDs, verify the frozen anchor/rank, and change only routed rows."""
    if features.duplicated(reuse.PAIR_KEYS).any() or not np.isfinite(features[nn.FEATURES]).all().all():
        raise ValueError('Invalid sparse evidence')
    if model.feature_name() != nn.FEATURES:
        raise ValueError('Native adapter feature order differs')
    positions = pd.MultiIndex.from_frame(base[reuse.PAIR_KEYS]).get_indexer(
        pd.MultiIndex.from_frame(features[reuse.PAIR_KEYS]))
    if (positions < 0).any(): raise ValueError('Foreign routed pair')
    if not np.array_equal(base.probability.to_numpy()[positions], features.probability.to_numpy()) or not np.array_equal(base.first_stage.to_numpy()[positions], features.first_stage.to_numpy()):
        raise ValueError('Frozen first-stage or final anchor changed')
    if not np.array_equal(nn.reverse.frozen_p2_rank(base)[positions], features.candidate_rank.to_numpy()):
        raise ValueError('Rank was calculated on a truncated candidate set')
    correction = model.predict(features[nn.FEATURES].to_numpy(dtype=np.float32), raw_score=True, num_threads=1)
    result = base.copy()
    result['probability_original'] = base.probability
    result.loc[positions, 'probability'] = nn.reverse.corrected_probability(features.probability, correction)
    if not result[reuse.PAIR_KEYS+['candidate_order']].equals(base[reuse.PAIR_KEYS+['candidate_order']]):
        raise ValueError('Candidate set or order changed')
    return result


def build(args):
    started = time.perf_counter()
    freeze = json.loads(args.freeze.read_text())
    if freeze.get('status') != 'candidate_frozen' or freeze.get('mode') != 'R2':
        raise ValueError('Requires an audited, frozen R2 candidate')
    for group in ['code_sha256', 'model_sha256', 'schema_sha256', 'index_sha256', 'native_sha256']:
        if not freeze.get(group): raise ValueError('Missing immutable evidence group: '+group)
        for file, digest in freeze[group].items():
            reuse.resolve_evidence({'path':file, 'sha256':digest}, ROOT)
    if freeze['code_sha256'].get(str(Path(__file__).resolve().relative_to(ROOT))) != reuse.digest(__file__):
        raise ValueError('Exporter differs from frozen implementation')
    for name in ['submission_03', 'submission_04']:
        preserved = (ROOT/'output'/name).resolve()
        if args.output.resolve() == preserved or preserved in args.output.resolve().parents:
            raise ValueError('Preserved baseline cannot be a destination')
    if args.output.exists(): raise ValueError('Require unused versioned output')
    manifest = json.loads(args.reuse_manifest.read_text())
    if manifest['baseline_frozen']['sha256'] != freeze['parent_frozen_sha256']:
        raise ValueError('Baseline freeze differs')
    rows, base = reuse.load_verified_chunks(manifest, args.test_dir)
    features, fm = nn.load_features(args.features)
    mm = json.loads((args.model/'manifest.json').read_text())
    native_path = str((args.model/'adapter.txt').resolve().relative_to(ROOT))
    marker_path = str((args.model/'manifest.json').resolve().relative_to(ROOT))
    if freeze['model_sha256'].get(native_path) != mm['model_sha256'] or freeze['model_sha256'].get(marker_path) != reuse.digest(args.model/'manifest.json'):
        raise ValueError('CLI model is not the frozen selected adapter')
    if nn.sibling.digest(args.model/'adapter.txt') != mm['model_sha256'] or fm['neural_head_manifest_sha256'] != mm['neural_head_manifest_sha256']:
        raise ValueError('Native model or neural feature lineage differs')
    if fm['input_pair_order_sha256'] != manifest['pair_order_sha256'] or fm['split'] != 'test':
        raise ValueError('Sparse features came from another full candidate universe')
    route = json.loads(args.route_manifest.read_text())
    if fm.get('route_plan_sha256') != reuse.digest(args.route_manifest):
        raise ValueError('Sparse evidence belongs to another route plan')
    if route['global_pair_order_sha256'] != manifest['pair_order_sha256'] or route['global_references'] != len(rows):
        raise ValueError('Route was not selected on the full test universe')
    if route['route'] != nn.sibling.asdict(nn.sibling.RouteConfig()):
        raise ValueError('Routing policy changed')
    routed = pd.read_parquet(reuse.resolve_evidence({'path':str(args.route_manifest.parent/'route.parquet'), 'sha256':route['route_sha256']}, ROOT))
    if set(map(tuple,routed[reuse.PAIR_KEYS].to_numpy())) != set(map(tuple,features[reuse.PAIR_KEYS].to_numpy())):
        raise ValueError('Scored evidence does not exactly cover the global route')
    frame = apply_adapter(base, features, lgb.Booster(model_file=str(args.model/'adapter.txt')))
    runner = reuse.resolve_evidence(freeze['original_runner'], ROOT)
    chosen, lost = reuse.load_original_decide(runner)(frame, frame, freeze['decision_config'])
    args.output.mkdir(parents=True)
    shutil.copy2(args.freeze,args.output/'frozen.json')
    statistics = reuse.export_outputs(rows, frame, chosen, args.output)
    validation = reuse.validate_outputs(args.output,args.test_dir)
    report = {'status':'complete' if validation['strict']==validation['official']=='PASS' else 'validation_failed',
        'mode':'R2','candidate_frozen_sha256':reuse.digest(args.freeze),
        'entities':len(rows),'scored_candidates':len(frame),'auxiliary_scored_pairs':len(features),
        'predicted_links':int(chosen.sum()),'ownership_lost_pairs':int(lost.sum()),
        'candidate_pair_order_sha256':reuse.pair_order_digest(frame),'statistics':statistics,
        'validation':validation,'seconds':time.perf_counter()-started,
        'source_sha256':{'reuse_manifest':reuse.digest(args.reuse_manifest),'model':mm['model_sha256'],
            'features':fm['features_sha256'],'route_manifest':reuse.digest(args.route_manifest)},
        'files_sha256':{n:reuse.digest(args.output/n) for n in ['matching_results.tsv','candidate_pairs.tsv']}}
    reuse.write_json(args.output/'submission_report.json', report)
    if report['status'] != 'complete': raise ValueError('Submission failed validation')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['freeze','reuse_manifest','features','model','route_manifest','test_dir','output']:
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    print(json.dumps(build(parser.parse_args()),indent=2))
