"""Label-free comparison on accepted duplicate targets, using sealed floor cache."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow as pa

KEYS = ['source1_entity_id', 'candidate_entity_id']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    for arg in ['repo', 'cache', 'accepted', 'output']: p.add_argument('--'+arg, required=True, type=Path)
    p.add_argument('--comparator', type=Path)
    a = p.parse_args(); started = time.monotonic()
    if a.output.exists(): raise FileExistsError(a.output)
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    proof = json.loads((a.cache/'baseline_parity.json').read_text())
    if proof['status'] != 'baseline_parity_verified' or not proof['exact_matching_byte_reconstruction']:
        raise ValueError('Exact complete baseline reconstruction required')
    for name, key in [('eligible_claims.parquet','eligible_claims_sha256'), ('baseline_masks.npz','baseline_masks_sha256')]:
        if sha(a.cache/name) != proof[key]: raise ValueError('Sealed cache hash mismatch')
    accepted = pd.read_parquet(a.accepted)
    targets = set(accepted.candidate_entity_id); duplicate_owners = set(accepted.source1_entity_id)
    frame = pd.read_parquet(a.cache/'eligible_claims.parquet', columns=KEYS+['probability','candidate_order'])
    baseline = np.load(a.cache/'baseline_masks.npz')['chosen']
    if len(frame) != proof['retained_pairs'] or len(frame) != len(baseline): raise ValueError('Cache coverage changed')
    # Include every claimant of a reported target, and its complete above-floor
    # owner graph, so original rank-one eligibility is exact for those targets.
    target_rows = frame.candidate_entity_id.isin(targets).to_numpy()
    external = set(frame.loc[target_rows,'source1_entity_id']) - duplicate_owners
    owners = duplicate_owners | external
    take = frame.source1_entity_id.isin(owners).to_numpy()
    work = frame.loc[take].reset_index(drop=True)
    original = baseline[take]
    selected_targets = work.candidate_entity_id.isin(targets).to_numpy()
    reported_original = work.loc[original & selected_targets, KEYS]
    if set(map(tuple, reported_original.to_numpy())) != set(map(tuple, accepted[KEYS].to_numpy())):
        raise ValueError('Accepted collision census differs from verified baseline masks')
    module_path = a.comparator or a.repo/'research/final_2h/cursor_evaluation.py'
    spec = importlib.util.spec_from_file_location('ownership_comparator', module_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    frozen = a.repo/'research/sprint_6h/coordination/baseline_freeze/frozen.json'
    if sha(frozen) != proof['frozen_sha256']: raise ValueError('Frozen decision configuration changed')
    config = json.loads(frozen.read_text())
    chosen, lost = module.admissible_lex(work, config)
    probability = work.probability.to_numpy()
    adjusted = np.where(np.load(a.cache/'baseline_masks.npz')['lost'][take], 0., probability)
    codes, _ = pd.factorize(work.source1_entity_id)
    order = np.lexsort((-probability, codes))
    first = np.zeros(len(work), dtype=bool)
    first[order[np.r_[True, codes[order][1:] != codes[order][:-1]]]] = True
    proposed = (probability >= config['t_rest']) | (first & (probability >= config['t_first']))
    needed_ids = set(work.loc[selected_targets,'source1_entity_id'])
    countries = {}
    source = a.repo/'dataset/test/test_source1.tsv'
    if sha(source) != '3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5':
        raise ValueError('Provided official Source1 hash changed')
    with source.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            if row['entity_id'] in needed_ids: countries[row['entity_id']] = row['country'].strip().casefold()
    if set(countries) != needed_ids: raise ValueError('Country coverage incomplete')
    details = work.loc[selected_targets].copy()
    details['original_accepted'] = original[selected_targets]
    details['original_rank_one'] = first[selected_targets]
    details['admissible_proposed'] = proposed[selected_targets]
    details['admissible_accepted'] = chosen[selected_targets]
    details['country'] = details.source1_entity_id.map(countries)
    details['original_adjusted_probability'] = adjusted[selected_targets]
    removed = original & ~chosen & selected_targets
    added = chosen & ~original & selected_targets
    counts = work.loc[chosen & selected_targets].candidate_entity_id.value_counts()
    accepted_types = {}
    for country, group in details.loc[details.original_accepted].groupby('country'):
        low = group.loc[group.probability < config['threshold']]
        accepted_types[country] = {'accepted_duplicate_claims':len(group), 'accepted_low_band':len(low),
          'low_band_original_rank_one':int(low.original_rank_one.sum()),
          'low_band_promoted_after_high_competition_loss':int((~low.original_rank_one).sum())}
    winners = work.loc[chosen & selected_targets].set_index('candidate_entity_id').probability
    high_removed = work.loc[removed & (probability >= config['threshold'])]
    high_ties = bool(np.array_equal(high_removed.probability.to_numpy(), high_removed.candidate_entity_id.map(winners).to_numpy()))
    report = {
        'status':'complete', 'scope':'Label-free; exact comparator only for all 1198 originally duplicated accepted target IDs',
        'original_duplicate_targets':len(targets), 'original_duplicate_claimants':len(duplicate_owners),
        'additional_external_claimants':len(external), 'complete_above_floor_owner_pairs':len(work),
        'reported_target_claims':int(selected_targets.sum()),
        'original_accepted_claims':int((original & selected_targets).sum()),
        'original_accepted_high_claims':int((original & selected_targets & (probability >= config['threshold'])).sum()),
        'original_accepted_low_band_claims':int((original & selected_targets & (probability < config['threshold'])).sum()),
        'original_accepted_unadjusted_rank_one_claims':int((original & selected_targets & first).sum()),
        'admissible_accepted_claims':int((chosen & selected_targets).sum()),
        'admissible_duplicate_targets':int((counts > 1).sum()),
        'removed_claims':int(removed.sum()), 'added_claims':int(added.sum()),
        'removed_by_country':dict(Counter(work.loc[removed,'source1_entity_id'].map(countries))),
        'added_by_country':dict(Counter(work.loc[added,'source1_entity_id'].map(countries))),
        'removed_low_band_claims':int((removed & (probability < config['threshold'])).sum()),
        'removed_high_claims':int((removed & (probability >= config['threshold'])).sum()),
        'removed_high_claims_are_exact_score_ties':high_ties,
        'accepted_duplicate_claim_type_by_country':accepted_types,
        'original_accepted_low_band_promoted_after_high_competition_loss':int((original & selected_targets & (probability < config['threshold']) & ~first).sum()),
        'config':{k:config[k] for k in ('threshold','t_first','t_rest')},
        'labels_read':False, 'output_TSVs_modified':False, 'release_policies_modified':False,
        'input_sha256':{'cache_proof':sha(a.cache/'baseline_parity.json'), 'eligible_claims':proof['eligible_claims_sha256'],
          'baseline_masks':proof['baseline_masks_sha256'], 'comparator':sha(module_path), 'frozen':sha(frozen), 'accepted_duplicate_pairs':sha(a.accepted)},
        'baseline_matching_sha256':proof['baseline_matching_sha256'],
        'completeness':'All claims for reported target IDs and all above-floor candidate rows for every such owner are included. Below-floor rows cannot be admissible or outrank retained rows. Results for other target IDs are deliberately not evaluated.',
        'seconds':time.monotonic()-started,
    }
    a.output.mkdir(parents=True)
    work.to_parquet(a.output/'needed_owner_pairs.parquet', index=False)
    details.to_parquet(a.output/'duplicate_target_claims.parquet',index=False)
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)

if __name__ == '__main__': main()
