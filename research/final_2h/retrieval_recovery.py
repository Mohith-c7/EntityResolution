"""Rescore the sealed, exposed5k upstream-quota cache with broad evidence.

This is a bounded combination diagnostic, never a release or fresh-label gate.
Preparation reads supplied text, read-only indices, and already scored candidates.
No candidate is inferred from truth and no original candidate can be removed.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3
import sys
import time
import multiprocessing
from dataclasses import replace
from types import SimpleNamespace
import shutil
import importlib.util

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'research/final_2h'), str(ROOT / 'scripts'),
               str(ROOT / 'code/business_entity_resolution')]
import broad_evidence as broad
from src.blocking.disk_index import normalize_record
from src.blocking.contracts import Candidate
from src.preprocessing.normalize import accent_fold
from rapidfuzz import fuzz

KEYS = broad.sibling.KEYS
SAMPLE_HASH = '43db565a3dfa04e83a707d3350c2a9619981af20fe517265e226be53e12e996c'
BENCH_INDEXES = []
BENCH_EMPTY = []
BENCH_EMPTY_BITMAP = []
BENCH_INIT_SECONDS = 0.
ALIGNMENT_RECORDS = {}
ALIGNMENT_STATS = None


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def pair_set(frame):
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError('Missing or duplicate candidate pair')
    return set(frame[KEYS].itertuples(index=False, name=None))


def validate_extension(base, extended, expected):
    bp, ep = pair_set(base), pair_set(extended)
    if not bp <= ep:
        raise ValueError('Extension removed an original candidate')
    if bp != {(e, c) for e, cs in expected.items() for c in cs}:
        raise ValueError('Original40 candidates do not replay the sealed sample')
    additions = extended.loc[[p not in bp for p in extended[KEYS].itertuples(index=False, name=None)], KEYS].copy()
    if len(additions) and (additions.groupby('source1_entity_id').size() > 4).any():
        raise ValueError('More than four added candidates for a source')
    return additions


def read_scored(cache, mode):
    files = sorted((cache / ('scored_' + mode)).glob('*.parquet'))
    if not files:
        raise FileNotFoundError('No sealed scored candidate chunks')
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def feature_jobs(frame, countries):
    mask, r1, r2 = broad.route(frame)
    seeds = {}
    for owner, group in frame.groupby('source1_entity_id', sort=False):
        strong = group[group.probability >= .9].sort_values(
            ['probability', 'candidate_entity_id'], ascending=[False, True]).head(2)
        seeds[owner] = strong.candidate_entity_id.tolist(), strong.probability.tolist()
    tasks = []
    for i in np.flatnonzero(mask):
        e = frame.source1_entity_id.iat[i]
        tasks.append((e, frame.candidate_entity_id.iat[i], countries[e].casefold(),
                      float(frame.first_stage.iat[i]), float(frame.probability.iat[i]),
                      int(r1[i]), int(r2[i]), *seeds[e]))
    return tasks


def load_cached_targets(cache, frames):
    targets = {}
    provenance = []
    cached_pairs = {m: set() for m in frames}
    for path in sorted(cache.glob('cached_*.pickle')):
        chunks = pickle.loads(path.read_bytes())
        for mode in frames:
            for owner, candidates in chunks[mode].items():
                for candidate, record, source in candidates:
                    cached_pairs[mode].add((owner, record.entity_id))
                    value = (record.name, record.address, source, record.country)
                    if record.entity_id in targets and targets[record.entity_id] != value:
                        raise ValueError('Cached target text differs across modes')
                    targets[record.entity_id] = value
                    if mode == 'upstreamquota' and 'name_only_missing_address' in candidate.blocking_paths:
                        provenance.append((owner, record.entity_id, source, '|'.join(candidate.blocking_paths), candidate.rank_within_source))
    for mode, frame in frames.items():
        if cached_pairs[mode] != pair_set(frame):
            raise ValueError('Cached raw candidates differ from actually scored pairs')
    return targets, pd.DataFrame(provenance, columns=[*KEYS, 'candidate_source', 'blocking_paths', 'rank_within_source'])


def load_raw_additions(ids, directory):
    targets = {}
    for source in ('S2', 'S3'):
        with (directory / ('train_source' + source[1:] + '.tsv')).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                if row['entity_id'] in ids:
                    targets[row['entity_id']] = row
    if set(targets) != set(ids):
        raise ValueError('Missing supplied raw added-target records')
    return targets


def prepare(args):
    started = time.monotonic()
    sample = json.loads(args.sample.read_text())
    references = {r['entity_id']: r for r in sample['references']}
    ids_hash = hashlib.sha256('\n'.join(sorted(references)).encode()).hexdigest()
    if len(references) != 5000 or ids_hash != SAMPLE_HASH:
        raise ValueError('Only the previously exposed fixed5k is allowed')
    seal = json.loads((args.cache / 'report.json').read_text())
    if not seal['all_reported_candidate_pairs_model_scored'] or seal['sample_ids_sha256'] != ids_hash:
        raise ValueError('Candidate cache is unsealed or not actually scored')
    frames = {m: read_scored(args.cache, m) for m in ('baseline', 'upstreamquota')}
    additions = validate_extension(frames['baseline'], frames['upstreamquota'], sample['baseline_candidates'])
    saved = pd.read_parquet(args.sample.with_name('saved_sample_scores.parquet'))
    parity = frames['baseline'].merge(saved, on=KEYS, suffixes=('_new', '_saved'), validate='one_to_one')
    if len(parity) != len(saved) or any(not np.array_equal(parity[n+'_new'], parity[n+'_saved']) for n in ('first_stage', 'second_stage', 'probability')):
        raise ValueError('Frozen baseline scores fail exact parity')
    targets, provenance = load_cached_targets(args.cache, frames)
    if pair_set(additions) != pair_set(provenance):
        raise ValueError('Added-pair provenance does not match actual scored additions')
    raw_targets = load_raw_additions(set(additions.candidate_entity_id), args.raw_directory)
    raw_parity = 0
    for cid, raw in raw_targets.items():
        record = normalize_record(raw)
        name, address, source, country = targets[cid]
        if (record.name, record.address, record.country) != (name, address, country):
            raise ValueError('Added-target raw/index normalization mismatch: ' + cid)
        raw_parity += 1
    broad.COUNTS, broad.WORD_COUNTS, broad.REFS = Counter(), Counter(), {}
    with sqlite3.connect(args.source1_index.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        meta = json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
        if not meta.get('complete') or meta['records'] != 2206821:
            raise ValueError('Wrong full supplied Source1 index')
        for e, name, core, address, country in conn.execute('SELECT id,name,namecore,primary_address,country FROM records'):
            country, core = country.casefold(), accent_fold(core)
            if core:
                broad.COUNTS[country, core] += 1
            broad.WORD_COUNTS.update((country, t) for t in set(name.split()))
            if e in references:
                record = normalize_record(references[e])
                if (name, address, country) != (record.name, record.address, record.country.casefold()):
                    raise ValueError('Raw query/source-index normalization mismatch: ' + e)
                broad.REFS[e] = broad.clean(name, address, 'S1')
    if set(broad.REFS) != set(references):
        raise ValueError('Missing source queries')
    broad.TABLES = {e: broad.clean(name, address, source) for e, (name, address, source, country) in targets.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    raw_rows = []
    for e, cid in additions.itertuples(index=False, name=None):
        raw, ref = raw_targets[cid], references[e]
        raw_rows.append({'source1_entity_id': e, 'candidate_entity_id': cid,
                         'source1_name_raw': ref['business_name'], 'source1_address_raw': ref['business_address'],
                         'target_name_raw': raw['business_name'], 'target_address_raw': raw['business_address'],
                         'source1_name_normalized': broad.REFS[e]['name'], 'source1_address_normalized': broad.REFS[e]['address'],
                         'target_name_normalized': targets[cid][0], 'target_address_normalized': targets[cid][1]})
    pd.DataFrame(raw_rows).merge(provenance, on=KEYS, validate='one_to_one').to_parquet(args.output / 'added_raw_text.parquet', index=False)
    additions.to_parquet(args.output / 'added_pairs.parquet', index=False)
    routed = {}
    feature_seconds = {}
    for mode, frame in frames.items():
        frame['country'] = frame.source1_entity_id.map(sample['countries'])
        frame.to_parquet(args.output / (mode + '_pairs.parquet'), index=False)
        tasks = feature_jobs(frame, sample['countries'])
        t0 = time.monotonic()
        chunks = [tasks[i:i+4000] for i in range(0, len(tasks), 4000)]
        arrays = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for values in pool.map(broad.rows_job, chunks):
                arrays.extend(values)
        features = pd.DataFrame(arrays, columns=[*KEYS, *broad.NAMES])
        pair_set(features)
        if not np.isfinite(features[broad.NAMES]).all().all():
            raise ValueError('Non-finite features')
        anchors = features.merge(frame, on=KEYS, validate='one_to_one')
        if not np.array_equal(anchors.p2, anchors.probability) or not np.array_equal(anchors.p1, anchors.first_stage):
            raise ValueError('Pair-key or probability anchor changed')
        features.to_parquet(args.output / (mode + '_features.parquet'), index=False)
        routed[mode] = len(features)
        feature_seconds[mode] = time.monotonic() - t0
        print(json.dumps({'mode': mode, 'routed': len(features), 'feature_seconds': feature_seconds[mode]}), flush=True)
    manifest = {'status': 'ready_for_new_model_scoring', 'scope': 'previously_exposed_fixed5k_only',
                'models_fitted': False, 'fresh_or_audit_labels_read': False, 'preparation_truth_accessed': False,
                'sample_ids_sha256': ids_hash, 'sample_sha256': broad.sha(args.sample), 'cache_report_sha256': broad.sha(args.cache/'report.json'),
                'features': broad.NAMES, 'feature_code_sha256': broad.sha(broad.__file__), 'code_sha256': broad.sha(__file__),
                'source1_index_manifest_sha256': broad.sha(args.source1_index.with_suffix('.sqlite.complete.json')),
                'source_queries_raw_normalization_verified': len(references), 'added_targets_raw_normalization_verified': raw_parity,
                'original_candidate_pairs': len(frames['baseline']), 'extended_candidate_pairs': len(frames['upstreamquota']),
                'original_pairs_preserved': True, 'all_final_pairs_frozen_model_scored': True, 'baseline_scores_exact': True,
                'added_pairs': len(additions), 'maximum_added_per_source': int(additions.groupby('source1_entity_id').size().max()),
                'routed_pairs': routed, 'feature_seconds': feature_seconds, 'total_prepare_seconds': time.monotonic()-started,
                'retrieval_existing_measured_worker_seconds': seal['variants']['upstreamquota']['extension_worker_seconds'],
                'candidate_oracle_delta_only': seal['candidate_oracle_delta'], 'oracle_is_matching_gain': False,
                'files': {p.name: broad.sha(p) for p in args.output.glob('*.parquet')}}
    write_json(args.output/'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


def score(args):
    import lightgbm as lgb
    manifest = json.loads((args.features/'manifest.json').read_text())
    names = manifest['features']
    if names[:60] != broad.NAMES or len(names) not in (60, 77) or manifest['feature_code_sha256'] != broad.sha(broad.__file__):
        raise ValueError('Feature schema/code changed')
    for filename, digest in manifest['files'].items():
        if broad.sha(args.features/filename) != digest:
            raise ValueError('Prepared artifact changed')
    model = lgb.Booster(model_file=str(args.adapter))
    if model.feature_name() != names:
        raise ValueError('Adapter features differ')
    args.output.mkdir(parents=True, exist_ok=False)
    sample = json.loads(args.sample.read_text())
    evaluation = json.loads(args.evaluation_subset.read_text())
    evaluation_ids = evaluation['evaluation_reference_ids']
    if (not evaluation.get('no_training_owners_in_evaluation')
            or len(evaluation_ids) != len(set(evaluation_ids))
            or not set(evaluation_ids) <= set(sample['countries'])
            or hashlib.sha256('\n'.join(sorted(evaluation_ids)).encode()).hexdigest() != evaluation['evaluation_ids_sha256']):
        raise ValueError('A sealed model-fit-independent evaluation subset is required')
    config = json.loads(args.frozen.read_text())
    other = pd.read_parquet(args.sample.with_name('other45k_strong_claims.parquet'))
    frames, scoring_seconds = {}, {}
    for mode in ('baseline', 'upstreamquota'):
        frame = pd.read_parquet(args.features/(mode+'_pairs.parquet'))
        features = pd.read_parquet(args.features/(mode+'_features.parquet'))
        indices = pd.MultiIndex.from_frame(frame[KEYS]).get_indexer(pd.MultiIndex.from_frame(features[KEYS]))
        if (indices < 0).any() or not np.array_equal(frame.probability.to_numpy()[indices], features.p2):
            raise ValueError('Rescoring anchor differs')
        started = time.monotonic()
        correction = model.predict(features[names].to_numpy(dtype=np.float32), raw_score=True, num_threads=args.workers)
        frame['probability_original'] = frame.probability
        frame.loc[indices, 'probability'] = broad.reverse.corrected_probability(features.p2, args.weight*correction)
        scoring_seconds[mode] = time.monotonic()-started
        frame.to_parquet(args.output/(mode+'_pairs.parquet'), index=False)
        frames[mode] = frame
    predictions, decisions = {}, {}
    for mode, frame in frames.items():
        claims = pd.concat([other, frame[[*KEYS, 'probability']]], ignore_index=True)
        chosen, lost = broad.decide_control(frame, claims, config)
        selected = frame.source1_entity_id.isin(evaluation_ids).to_numpy()
        predictions[mode] = broad.predictions_for(frame.loc[selected], chosen[selected], evaluation_ids)
        decisions[mode] = {'ownership_rejected_pairs': int(lost.sum()), 'accepted_pairs': int(chosen.sum())}
    report = broad.paired_evaluate({e:sample['truth'][e] for e in evaluation_ids}, predictions['baseline'],
                                  predictions['upstreamquota'], {e:sample['countries'][e] for e in evaluation_ids}, role='development')
    report.update(scope='clean_exposed_subset broad40_vs_broad44 combination diagnostic',
                  evaluated_references=len(evaluation_ids), clean_subset_sha256=broad.sha(args.evaluation_subset),
                  adapter_sha256=broad.sha(args.adapter), adapter_weight=args.weight,
                  feature_manifest_sha256=broad.sha(args.features/'manifest.json'),
                  all_final_pairs_frozen_model_scored=True, all_routed_pairs_new_adapter_scored=True,
                  scoring_seconds=scoring_seconds, decisions=decisions,
                  fresh_or_audit_labels_read=False, promotion_eligible=False,
                  limitation='Changed5k lists plus unchanged frozen45k rival claims; not a uniformly recomputed50k or fullSource1 graph. Metrics use only the supplied sealed clean owner subset; caller must include every fitted model owner in its exclusion seal.')
    write_json(args.output/'report.json', report)
    print(json.dumps(report), flush=True)


def neural_evaluate(args):
    """Replay native37 scores with all rival claims; evaluate clean owners only."""
    sys.path.insert(0, str(ROOT/'research/sprint_6h'))
    import neural_adapter as neural
    prepared = args.features
    subset = json.loads(args.evaluation_subset.read_text())
    sample = json.loads(args.sample.read_text())
    model_manifest = args.adapter/'manifest.json'
    head_manifest = ROOT/'models/sprint_6h/neural/frozen_head120k_v1/manifest.json'
    mm = json.loads(model_manifest.read_text()); hm = json.loads(head_manifest.read_text())
    ids = subset['evaluation_reference_ids']
    fitting = set(mm['training_owner_ids']) | set(hm['training_owners'])
    expected = set(sample['countries']) - fitting
    if len(ids) != len(set(ids)) or set(ids) != expected or not set(ids) <= set(sample['truth']):
        raise ValueError('Evaluation must exclude every fitted owner and only fitted owners')
    if (subset['adapter_fitting_manifest_sha256'] != broad.sha(model_manifest)
            or subset['head_fitting_manifest_sha256'] != broad.sha(head_manifest)
            or subset['evaluation_ids_sha256'] != hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()):
        raise ValueError('Clean evaluation subset provenance changed')
    config = json.loads(args.frozen.read_text())
    other_path = args.sample.with_name('other45k_strong_claims.parquet')
    other = pd.read_parquet(other_path)
    frames, predictions, decisions = {}, {}, {}
    for mode in ('baseline','upstreamquota'):
        directory = prepared/('scored_neural_'+mode)
        marker = json.loads((directory/'manifest.json').read_text())
        if (marker['adapter_model_sha256'] != mm['model_sha256']
                or broad.sha(args.adapter/'adapter.txt') != mm['model_sha256']
                or broad.sha(directory/'pairs.parquet') != marker['pairs_sha256']):
            raise ValueError('Native neural scored graph provenance changed')
        features, fm = neural.load_features(prepared/('combined_neural_'+mode))
        if fm['neural_head_manifest_sha256'] != broad.sha(head_manifest):
            raise ValueError('Neural head changed')
        frame = pd.read_parquet(directory/'pairs.parquet')
        original = pd.read_parquet(prepared/'input_graphs'/mode/'pairs.parquet')
        if not frame[KEYS].equals(original[KEYS]) or not np.array_equal(frame.probability_original, original.probability):
            raise ValueError('Native rescoring changed graph or p2 anchor')
        if set(frame.source1_entity_id) != set(sample['countries']):
            raise ValueError('Global decision graph must retain all 5000 owners')
        frames[mode] = frame
        claims = pd.concat([other,frame[[*KEYS,'probability']]],ignore_index=True)
        chosen,lost = broad.decide_control(frame,claims,config)
        selected = frame.source1_entity_id.isin(ids).to_numpy()
        predictions[mode] = broad.predictions_for(frame.loc[selected],chosen[selected],ids)
        decisions[mode] = {'all5000_accepted_pairs':int(chosen.sum()),
                          'all5000_ownership_rejected_pairs':int(lost.sum()),
                          'clean2952_accepted_pairs':int(chosen[selected].sum()),
                          'routed_neural_scored_pairs':len(features)}
    original_lists = frames['baseline'].groupby('source1_entity_id',sort=False).candidate_entity_id.apply(list).to_dict()
    added = validate_extension(frames['baseline'],frames['upstreamquota'],original_lists)
    truth = {e:sample['truth'][e] for e in ids}
    countries = {e:sample['countries'][e] for e in ids}
    report = broad.paired_evaluate(truth,predictions['baseline'],predictions['upstreamquota'],countries,role='development')
    report.update(scope='clean_exposed2952 native neural_v2 baseline40_vs_quota44',
      evaluated_references=len(ids), all_decision_claimants=5000, excluded_fitted_references=5000-len(ids),
      decisions=decisions, added_candidates=len(added), all_original40_candidates_preserved=True,
      every_final_pair_scored=True, route_recomputed_independently_each_complete5k_graph=True,
      historical_global50k_neural_baseline_parity_claimed=False,
      labels_scope='Previously exposed fixed5k; only the model-independent clean subset enters metrics',
      adapter_manifest_sha256=broad.sha(model_manifest), neural_head_manifest_sha256=broad.sha(head_manifest),
      clean_subset_sha256=broad.sha(args.evaluation_subset), other45k_claims_sha256=broad.sha(other_path),
      fresh_or_audit_labels_read=False, promotion_eligible=False,
      limitation='Decisions use complete5k plus preserved frozen45k rival claims; not a uniformly rescored50k or fullSource1 graph.')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'report.json',report)
    print(json.dumps(report),flush=True)


def alignment_job(keys):
    from src.features.token_alignment import build_alignment_features
    return [[e, c, *build_alignment_features(ALIGNMENT_RECORDS[e], ALIGNMENT_RECORDS[c], ALIGNMENT_STATS).values()]
            for e, c in keys]


def align(args):
    """Match the root's declared77-feature variant without changing any scores."""
    global ALIGNMENT_RECORDS, ALIGNMENT_STATS
    if args.alignment_module is not None:
        spec = importlib.util.spec_from_file_location('src.features.token_alignment', args.alignment_module)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    from src.features.token_alignment import AlignmentStatistics, FEATURE_NAMES
    started = time.monotonic()
    manifest = json.loads((args.features/'manifest.json').read_text())
    if manifest['features'] != broad.NAMES or manifest['feature_code_sha256'] != broad.sha(broad.__file__):
        raise ValueError('Alignment needs sealed broad60 features')
    for filename, digest in manifest['files'].items():
        if broad.sha(args.features/filename) != digest:
            raise ValueError('Prepared artifact changed')
    if broad.sha(args.statistics) != 'ed0712841507c3e0b17f2116c9a2db8423126d894d360f7d74510efe20682d82':
        raise ValueError('Statistics differ from root-sealed fullSource1 artifact')
    ALIGNMENT_STATS = AlignmentStatistics.from_dict(json.loads(args.statistics.read_text()))
    if ALIGNMENT_STATS.record_count != 2206821:
        raise ValueError('Wrong fullSource1 alignment statistics')
    with sqlite3.connect(args.source1_index.resolve().as_uri()+'?mode=ro', uri=True) as conn:
        meta = json.loads(conn.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
    if meta['fingerprint']['source_sha256'] not in set(ALIGNMENT_STATS.source_sha256.values()):
        raise ValueError('Alignment statistics came from another supplied corpus')
    sample = json.loads(args.sample.read_text())
    if hashlib.sha256('\n'.join(sorted(r['entity_id'] for r in sample['references'])).encode()).hexdigest() != SAMPLE_HASH:
        raise ValueError('Only the previously exposed fixed5k is allowed')
    frames = {m: pd.read_parquet(args.features/(m+'_pairs.parquet')) for m in ('baseline', 'upstreamquota')}
    targets, _ = load_cached_targets(args.cache, frames)
    ALIGNMENT_RECORDS = {e: SimpleNamespace(name=name, address=address, country=country)
                         for e, (name, address, source, country) in targets.items()}
    for row in sample['references']:
        record = normalize_record(row)
        ALIGNMENT_RECORDS[record.entity_id] = SimpleNamespace(name=record.name, address=record.address, country=record.country)
    args.output.mkdir(parents=True, exist_ok=False)
    for path in args.features.glob('*.parquet'):
        shutil.copy2(path, args.output/path.name)
    aligned_counts = {}
    for mode in frames:
        frame = pd.read_parquet(args.features/(mode+'_features.parquet'))
        selected = frame.loc[(frame.p2 >= .003) | (frame.rank_p1 <= 4), KEYS]
        keys = list(selected.itertuples(index=False, name=None))
        values = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rows in pool.map(alignment_job, [keys[i:i+2000] for i in range(0, len(keys), 2000)]):
                values.extend(rows)
        added = pd.DataFrame(values, columns=[*KEYS, *FEATURE_NAMES])
        pair_set(added)
        added['alignment_scored'] = 1.
        combined = frame.merge(added, on=KEYS, how='left', sort=False, validate='one_to_one')
        extra = [*FEATURE_NAMES, 'alignment_scored']
        combined[extra] = combined[extra].fillna(0)
        if not frame[KEYS].equals(combined[KEYS]) or not np.isfinite(combined[extra]).all().all():
            raise ValueError('Alignment changed keys or emitted invalid values')
        combined.to_parquet(args.output/(mode+'_features.parquet'), index=False)
        aligned_counts[mode] = len(added)
    result = {**manifest, 'features': [*broad.NAMES, *FEATURE_NAMES, 'alignment_scored'],
              'parent_manifest_sha256': broad.sha(args.features/'manifest.json'),
              'statistics_sha256': broad.sha(args.statistics),
              'alignment_code_sha256': broad.sha(args.alignment_module or ROOT/'code/business_entity_resolution/src/features/token_alignment.py'),
              'alignment_version': ALIGNMENT_STATS.version,
              'alignment_route': 'p2>=.003 or first_stage_rank<=4', 'alignment_pairs': aligned_counts,
              'alignment_prepare_seconds': time.monotonic()-started, 'code_sha256': broad.sha(__file__),
              'files': {p.name: broad.sha(p) for p in args.output.glob('*.parquet')}}
    write_json(args.output/'manifest.json', result)
    print(json.dumps(result), flush=True)


def benchmark_init(config):
    """Read-only cached inventory; the retrieval rule and ranker stay unchanged."""
    global BENCH_INDEXES, BENCH_EMPTY, BENCH_EMPTY_BITMAP, BENCH_INIT_SECONDS
    import run_frozen_pipeline as runner
    started = time.monotonic()
    runner.init(config, ROOT/config['aliases']['path'], str(ROOT/'models/index_train'),
                ROOT/'models/frozen_v4_checkpoint_repair/replay/universe_counts_train.sqlite')
    BENCH_INDEXES = runner.WORKER[0]
    BENCH_EMPTY = []
    BENCH_EMPTY_BITMAP = []
    for index in BENCH_INDEXES:
        records = {}
        sql = "SELECT rowid,id,name,core,address,country,name_digits,address_digits,postcodes,folded FROM records WHERE address=''"
        for row in index.connection.execute(sql):
            records[int(row[0])] = index.record_values(row[1:])
        BENCH_EMPTY.append(records)
        bitmap = np.zeros(index.connection.execute('SELECT MAX(rowid) FROM records').fetchone()[0]+1, dtype=bool)
        bitmap[list(records)] = True
        bitmap.flags.writeable = False
        BENCH_EMPTY_BITMAP.append(bitmap)
    BENCH_INIT_SECONDS = time.monotonic()-started


def benchmark_chunk(rows):
    from src.blocking.cheap_ranker import rank_features
    added, timings = [], []
    for raw, current_ids in rows:
        reference = normalize_record(raw)
        query = replace(reference, address='', address_tokens=frozenset(),
                        address_digits=frozenset(), postcode_candidates=frozenset())
        current = set(current_ids)
        t0, c0 = time.perf_counter(), time.process_time()
        posting_seconds = filter_seconds = rank_seconds = 0.
        for index, empty, bitmap in zip(BENCH_INDEXES, BENCH_EMPTY, BENCH_EMPTY_BITMAP):
            step = time.perf_counter()
            terms = index.rare_terms('name', reference.name_tokens,
                                     index.postings_config.max_postings, index.postings_config.tokens_per_field)
            pieces = [index.postings('name', term)[0] for term in terms]
            for core in index.aliases.variants(reference.core) if index.aliases else [reference.core]:
                if core:
                    pieces.append(index.postings('core', core)[0])
            ids = np.unique(np.concatenate(pieces)) if pieces else np.empty(0, dtype=np.int32)
            posting_seconds += time.perf_counter()-step
            step = time.perf_counter()
            eligible = []
            for rowid in ids[bitmap[ids]]:
                target = empty[int(rowid)]
                if target.entity_id in current or not reference.core or not target.core:
                    continue
                if fuzz.token_set_ratio(reference.core, target.core) >= 80:
                    eligible.append(target)
            filter_seconds += time.perf_counter()-step
            step = time.perf_counter()
            index.prefetch_frequencies([query, *eligible])
            probabilities = index.ranking_model.predict(
                np.asarray([rank_features(query, t, index) for t in eligible], dtype=np.float32),
                num_threads=1) if eligible else []
            chosen = sorted(zip(eligible, probabilities), key=lambda item: (-item[1], item[0].entity_id))[:2]
            rank_seconds += time.perf_counter()-step
            for offset, (target, probability) in enumerate(chosen, 1):
                candidate = Candidate(reference.entity_id, target.entity_id, index.source,
                    index.rerank(reference, target, ('name_only_missing_address',)),
                    ('name_only_missing_address',), 20+offset)
                added.append((candidate.source1_entity_id, candidate.candidate_entity_id))
        timings.append({'source1_entity_id': reference.entity_id,
                        'extension_wall_seconds': time.perf_counter()-t0,
                        'extension_cpu_seconds': time.process_time()-c0,
                        'posting_seconds': posting_seconds, 'filter_seconds': filter_seconds,
                        'rank_seconds': rank_seconds})
    return added, timings, {'pid': __import__('os').getpid(), 'init_seconds': BENCH_INIT_SECONDS,
                             'empty_records': [len(r) for r in BENCH_EMPTY]}


def benchmark(args):
    sample = json.loads(args.sample.read_text())
    if hashlib.sha256('\n'.join(sorted(r['entity_id'] for r in sample['references'])).encode()).hexdigest() != SAMPLE_HASH:
        raise ValueError('Only previously exposed fixed5k is allowed')
    base, extended = (read_scored(args.cache, mode) for mode in ('baseline', 'upstreamquota'))
    expected = validate_extension(base, extended, sample['baseline_candidates'])
    rows = [(r, sample['baseline_candidates'][r['entity_id']]) for r in sample['references']]
    chunks = [rows[i:i+100] for i in range(0, len(rows), 100)]
    config = json.loads(args.frozen.read_text())
    added, timings, workers = [], [], {}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=benchmark_init, initargs=(config,)) as pool:
        for pairs, timing, worker in pool.map(benchmark_chunk, chunks):
            added.extend(pairs)
            timings.extend(timing)
            workers[worker['pid']] = worker
    wall = time.monotonic()-started
    actual = pd.DataFrame(added, columns=KEYS)
    if pair_set(actual) != pair_set(expected):
        raise ValueError('Cached-metadata retrieval changed sealed candidate additions')
    args.output.mkdir(parents=True, exist_ok=False)
    actual.to_parquet(args.output/'added_pairs.parquet', index=False)
    pd.DataFrame(timings).to_parquet(args.output/'timings.parquet', index=False)
    report = {'status': 'exact_candidate_parity', 'scope': 'same_previously_exposed_fixed5k',
              'candidate_rule_changed': False, 'models_or_indices_modified': False, 'labels_used': False,
              'workers': args.workers, 'references': len(rows), 'exact_added_pair_parity': True,
              'added_pairs': len(actual), 'all_final_candidates_have_sealed_frozen_scores': True,
              'total_wall_seconds_including_worker_model_and_empty_metadata_init': wall,
              'extension_worker_wall_seconds': sum(t['extension_wall_seconds'] for t in timings),
              'extension_worker_cpu_seconds': sum(t['extension_cpu_seconds'] for t in timings),
              'mean_extension_ms': 1000*np.mean([t['extension_wall_seconds'] for t in timings]),
              'p95_extension_ms': 1000*np.quantile([t['extension_wall_seconds'] for t in timings], .95),
              'stage_worker_wall_seconds': {k: sum(t[k] for t in timings) for k in ['posting_seconds','filter_seconds','rank_seconds']},
              'worker_initialization': list(workers.values()), 'code_sha256': broad.sha(__file__),
              'limitation': 'Sample benchmark uses saved baseline40 and measures extension only. Initial metadata inventory/model loading included in totalwall; no baseline retrieval, scoring, fulltest throughput, or release ETA measured.'}
    write_json(args.output/'report.json', report)
    print(json.dumps(report), flush=True)


def reverse_builder_task(args):
    import reverse_competition as builder
    pairs, manifest, index, prefix, output, route = args
    return builder.generate_features(pairs, manifest, index, prefix, output, route_dir=route)


def neural_prepare(args):
    """Standard36 reverse inputs plus supplied raw text, ready for neural37 join."""
    sys.path.insert(0, str(ROOT/'research/sprint_6h'))
    import reverse_competition as builder
    import prepare_neural_inputs as inputs
    from dataclasses import asdict
    started = time.monotonic()
    sample = json.loads(args.sample.read_text())
    ids = [r['entity_id'] for r in sample['references']]
    if len(ids) != 5000 or hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest() != SAMPLE_HASH:
        raise ValueError('Only the previously exposed fixed5k is allowed')
    frames = {m: pd.read_parquet(args.features/(m+'_pairs.parquet')).reset_index(drop=True)
              for m in ('baseline', 'upstreamquota')}
    validate_extension(frames['baseline'], frames['upstreamquota'], sample['baseline_candidates'])
    if set(ids) != set(frames['baseline'].source1_entity_id) or set(ids) != set(frames['upstreamquota'].source1_entity_id):
        raise ValueError('Incomplete5k graph')
    args.output.mkdir(parents=True, exist_ok=False)
    markers, routes, route_dirs, jobs, part_jobs = {}, {}, {}, [], {}
    for mode, frame in frames.items():
        directory = args.output/'input_graphs'/mode
        directory.mkdir(parents=True)
        frame.to_parquet(directory/'pairs.parquet', index=False)
        markers[mode] = {'status': 'complete', 'verified_current_pipeline': True,
                         'split': 'development', 'role': 'exposed_retrieval5k',
                         'reference_ids': ids, 'entities': len(ids), 'pairs': len(frame),
                         'pairs_sha256': broad.sha(directory/'pairs.parquet'),
                         'pair_order_sha256': broad.sibling.pair_order_hash(frame),
                         'frozen_sha256': broad.sha(args.frozen), 'labels_read': False,
                         'candidate_variant': mode, 'all_pairs_frozen_model_scored': True,
                         'feature_parent_manifest_sha256': broad.sha(args.features/'manifest.json')}
        write_json(directory/'manifest.json', markers[mode])
        route = broad.sibling.select_route(frame, universe_ids=ids)
        routes[mode] = route
        route_dir = args.output/('reverse_'+mode)/'global_route'
        route_dir.mkdir(parents=True)
        route_dirs[mode] = route_dir
        edges = broad.sibling.route_edges(frame, route)
        edges.to_parquet(route_dir/'route.parquet', index=False)
        write_json(route_dir/'manifest.json', {'status': 'complete', 'route': asdict(broad.sibling.RouteConfig()),
                   'global_references': len(ids), 'global_pair_order_sha256': markers[mode]['pair_order_sha256'],
                   'route_sha256': broad.sha(route_dir/'route.parquet'), 'frozen_sha256': broad.sha(args.frozen),
                   'labels_read': False})
        mode_jobs = []
        # Preserve complete, variable-sized owner groups; only execution is split.
        for number, owner_part in enumerate(np.array_split(np.asarray(ids), max(1, args.workers//2))):
            part = args.output/('reverse_'+mode)/'parts'/str(number)
            part.mkdir(parents=True)
            sub = frame[frame.source1_entity_id.isin(owner_part)].reset_index(drop=True)
            sub.to_parquet(part/'pairs.parquet', index=False)
            marker = {**markers[mode], 'reference_ids': owner_part.tolist(), 'entities': len(owner_part),
                      'pairs': len(sub), 'pairs_sha256': broad.sha(part/'pairs.parquet'),
                      'pair_order_sha256': broad.sibling.pair_order_hash(sub),
                      'routing_universe_pair_order_sha256': markers[mode]['pair_order_sha256'],
                      'parent_manifest_sha256': broad.sha(directory/'manifest.json')}
            write_json(part/'manifest.json', marker)
            job = (part/'pairs.parquet', part/'manifest.json', args.source1_index,
                   ROOT/'models/index_train', part/'features', route_dir)
            mode_jobs.append(job)
            jobs.append(job)
        part_jobs[mode] = mode_jobs
    route_pairs = {m: pair_set(frames[m].loc[list(routes[m]), KEYS]) for m in frames}
    retained = frames['baseline'].merge(frames['upstreamquota'], on=KEYS, suffixes=('_40', '_44'), validate='one_to_one')
    route_report = {'independent_cohort_route_config': asdict(broad.sibling.RouteConfig()),
                    'cohort_references': len(ids), 'original_pairs': len(retained),
                    'first_stage_changed_original_pairs': int((retained.first_stage_40 != retained.first_stage_44).sum()),
                    'rebuilt_context_second_stage_changed_original_pairs': int((retained.probability_40 != retained.probability_44).sum()),
                    'common_routed_pairs': len(route_pairs['baseline'] & route_pairs['upstreamquota']),
                    'baseline_only_routed_pairs': len(route_pairs['baseline']-route_pairs['upstreamquota']),
                    'quota_only_routed_pairs': len(route_pairs['upstreamquota']-route_pairs['baseline']),
                    'quota_routed_new_candidates': len(route_pairs['upstreamquota']-pair_set(frames['baseline'])),
                    'routed_pairs': {m: len(route_pairs[m]) for m in frames},
                    'routed_references': {m: len({e for e,c in route_pairs[m]}) for m in frames},
                    'historical_global50k_neural_route_parity_claimed': False,
                    'fairness_note': 'Fair paired5k pipeline comparison uses independently recomputed routes on both complete graphs. It is not an unchanged-route addition-only ablation or a replay of the historical globally routed50k neural baseline. Retained p2 changes are legitimate rebuilt candidate-context effects.'}
    write_json(args.output/'route_comparison.json', route_report)
    print(json.dumps({'stage': 'routes_pinned', **route_report}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(reverse_builder_task, jobs))
    combined = {}
    for mode in frames:
        output = args.output/('reverse_'+mode)
        chunks = part_jobs[mode]
        data = pd.concat([pd.read_parquet(j[4]/'features.parquet') for j in chunks], ignore_index=True)
        if pair_set(data) != route_pairs[mode]:
            raise ValueError('Reverse builder did not exactly cover pinned global route')
        data.to_parquet(output/'features.parquet', index=False)
        with (output/'diagnostics.jsonl').open('x') as stream:
            for job in chunks:
                with (job[4]/'diagnostics.jsonl').open() as source:
                    shutil.copyfileobj(source, stream)
        part_markers = [json.loads((j[4]/'manifest.json').read_text()) for j in chunks]
        first = part_markers[0]
        for marker in part_markers[1:]:
            for field in ('index_manifest_sha256','query_config','source_tsv_sha256','feature_code_sha256','normalization_sha256','phonetic_code_sha256'):
                if marker[field] != first[field]:
                    raise ValueError('Reverse part provenance differs')
        rm = {**first, 'features_sha256': broad.sha(output/'features.parquet'),
              'pair_order_sha256': broad.sibling.pair_order_hash(data),
              'source_pairs_sha256': markers[mode]['pairs_sha256'],
              'source_pair_order_sha256': markers[mode]['pair_order_sha256'],
              'source_manifest_sha256': broad.sha(args.output/'input_graphs'/mode/'manifest.json'),
              'route_manifest': str(route_dirs[mode]/'manifest.json'),
              'route_manifest_sha256': broad.sha(route_dirs[mode]/'manifest.json'),
              'diagnostics_sha256': broad.sha(output/'diagnostics.jsonl'), 'rows': len(data),
              'routed_references': data.source1_entity_id.nunique(),
              'parallel_wrapper_sha256': broad.sha(__file__), 'workers': len(chunks)}
        write_json(output/'manifest.json', rm)
        index = broad.reverse.validate_reverse_manifest(data, output/'features.parquet', rm, markers[mode], route_dirs[mode])
        joined = broad.reverse.join_reverse(frames[mode], data, routes[mode])
        positions = pd.MultiIndex.from_frame(frames[mode][KEYS]).get_indexer(pd.MultiIndex.from_frame(joined[KEYS]))
        if not np.array_equal(joined.first_stage, frames[mode].first_stage.to_numpy()[positions]) or not np.array_equal(joined.probability, frames[mode].probability.to_numpy()[positions]) or not np.array_equal(joined.candidate_rank, broad.reverse.frozen_p2_rank(frames[mode])[positions]):
            raise ValueError('Reverse36 join changed score anchors or full-graph ranks')
        dest = args.output/('combined_reverse_'+mode)
        dest.mkdir()
        joined.to_parquet(dest/'features.parquet', index=False)
        write_json(dest/'manifest.json', {'status': 'complete', 'version': broad.reverse.VERSION,
                   'features': broad.reverse.FEATURES, 'features_sha256': broad.sha(dest/'features.parquet'),
                   'feature_pair_order_sha256': broad.sibling.pair_order_hash(joined),
                   'input_pairs_sha256': markers[mode]['pairs_sha256'],
                   'input_pair_order_sha256': markers[mode]['pair_order_sha256'],
                   'input_manifest_sha256': broad.sha(args.output/'input_graphs'/mode/'manifest.json'),
                   'reverse_manifest_sha256': broad.sha(output/'manifest.json'),
                   'reverse_features_sha256': broad.sha(output/'features.parquet'),
                   'index_manifest_sha256': rm['index_manifest_sha256'], 'index_split': 'train',
                   'route_plan_sha256': broad.sha(route_dirs[mode]/'manifest.json'),
                   'reference_ids': ids, 'role': 'exposed_retrieval5k', 'split': 'development',
                   'labels_read': False, 'routed_rows': len(joined),
                   'source_sha256': broad.sha(broad.reverse.__file__)})
        loaded, _ = broad.reverse.load_features(dest)
        pd.testing.assert_frame_equal(joined, loaded)
        combined[mode] = joined
    wanted = set().union(*(set(f.source1_entity_id)|set(f.candidate_entity_id) for f in combined.values()))
    records, raw_hashes = {}, {}
    for source in (1, 2, 3):
        path = args.raw_directory/('train_source'+str(source)+'.tsv')
        raw_hashes['S'+str(source)] = broad.sha(path)
        for batch in pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, chunksize=100000):
            for raw in batch.loc[batch.entity_id.isin(wanted)].to_dict('records'):
                if raw['entity_id'] in records:
                    raise ValueError('Duplicate supplied raw record')
                records[raw['entity_id']] = inputs.serialize(raw)
    if set(records) != wanted:
        raise ValueError('Missing supplied raw neural records')
    for mode, frame in combined.items():
        directory = args.output/'neural_inputs'/mode
        directory.mkdir(parents=True)
        with (directory/'pairs.jsonl').open('x') as stream:
            for e,c in frame[KEYS].itertuples(index=False, name=None):
                stream.write(json.dumps({'source1_entity_id': e, 'candidate_entity_id': c,
                             'text_left': records[e], 'text_right': records[c]}, ensure_ascii=False)+'\n')
        feature_dir = args.output/('combined_reverse_'+mode)
        write_json(directory/'manifest.json', {'status': 'complete', 'rows': len(frame), 'labels_read': False,
                   'input_sha256': broad.sha(directory/'pairs.jsonl'),
                   'feature_manifest_sha256': broad.sha(feature_dir/'manifest.json'),
                   'feature_file_sha256': broad.sha(feature_dir/'features.parquet'),
                   'source_tsv_sha256': raw_hashes, 'serializer_code_sha256': broad.sha(inputs.__file__),
                   'code_sha256': broad.sha(__file__), 'role': 'exposed_retrieval5k', 'source_split': 'development'})
    write_json(args.output/'manifest.json', {'status': 'ready_for_mac_neural_scoring_and_standard37_join',
               'scope': 'same_previously_exposed_fixed5k', 'features_before_neural': broad.reverse.FEATURES,
               'features_after_neural': [*broad.reverse.FEATURES, 'neural_probability'],
               'index_corpus': 'train', 'route_comparison': route_report,
               'score_and_rank_anchors_exact': True, 'source_tsv_sha256': raw_hashes,
               'models_fitted': False, 'fresh_or_audit_labels_read': False, 'labels_used': False,
               'neural_join_layout': 'Place Mac score outputs in baseline/ and upstreamquota/ under this root; neural_adapter join then discovers neural_inputs/MODE/manifest.json automatically.',
               'standard_join_compatible': True, 'seconds': time.monotonic()-started,
               'code_sha256': broad.sha(__file__)})
    print(json.dumps({'stage': 'complete', 'seconds': time.monotonic()-started,
                      'routed_pairs': {m: len(f) for m,f in combined.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'score', 'benchmark', 'align', 'neural-prepare', 'neural-evaluate'])
    parser.add_argument('--sample', type=Path, default=ROOT/'research/sprint_6h/retrieval_probe/sample.json')
    parser.add_argument('--cache', type=Path, default=ROOT/'research/sprint_6h/retrieval_probe/result_upstreamquota')
    parser.add_argument('--source1-index', type=Path, default=ROOT/'research/sprint_6h/reverse_competition/train_s1.sqlite')
    parser.add_argument('--raw-directory', type=Path, default=ROOT/'dataset/train')
    parser.add_argument('--frozen', type=Path, default=ROOT/'research/sprint_6h/coordination/baseline_freeze/frozen.json')
    parser.add_argument('--features', type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--evaluation-subset', type=Path)
    parser.add_argument('--statistics', type=Path)
    parser.add_argument('--alignment-module', type=Path, help='Approved module snapshot when remote release lacks alignment code; core files stay untouched')
    parser.add_argument('--weight', type=float, default=1.)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.output.exists() or not 0 < args.weight <= 1:
        parser.error('Use unused output,1..8 workers, and weight in(0,1]')
    if args.command == 'score' and (args.features is None or args.adapter is None or args.evaluation_subset is None):
        parser.error('Scoring needs --features, --adapter, and sealed --evaluation-subset')
    if args.command == 'align' and (args.features is None or args.statistics is None):
        parser.error('Alignment needs --features and --statistics')
    if args.command == 'neural-prepare' and args.features is None:
        parser.error('Neural preparation needs --features')
    if args.command == 'neural-evaluate' and any(v is None for v in (args.features,args.adapter,args.evaluation_subset)):
        parser.error('Neural evaluation needs --features, --adapter model directory, --evaluation-subset')
    {'prepare': prepare, 'score': score, 'benchmark': benchmark, 'align': align, 'neural-prepare': neural_prepare,
     'neural-evaluate':neural_evaluate}[args.command](args)


if __name__ == '__main__':
    main()
