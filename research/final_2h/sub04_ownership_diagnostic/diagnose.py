"""Read-only accepted-link ownership census; no labels or release output writes."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MATCHES = ROOT/'output/submission_04/matching_results.tsv'
SOURCE1 = ROOT/'dataset/test/test_source1.tsv'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = time.monotonic()
    matching_hash = sha(MATCHES)
    frequencies = Counter()
    owners = set()
    links = empty = 0
    with MATCHES.open(newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if reader.fieldnames != ['source1_entity_id', 'matched_entity_ids']:
            raise ValueError('Unexpected accepted-link format')
        for row in reader:
            owner = row['source1_entity_id']
            if owner in owners:
                raise ValueError('Duplicate Source1 output row')
            owners.add(owner)
            targets = row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
            if len(targets) != len(set(targets)):
                raise ValueError('Duplicate target within one owner')
            frequencies.update(targets)
            links += len(targets)
            empty += not bool(targets)
    duplicate_targets = {c for c,n in frequencies.items() if n > 1}
    print(json.dumps({'stage': 'accepted_census', 'owners': len(owners), 'accepted_links': links,
                      'unique_targets': len(frequencies), 'duplicate_targets': len(duplicate_targets),
                      'seconds': time.monotonic()-started}), flush=True)
    duplicate_links = []
    if duplicate_targets:
        with MATCHES.open(newline='') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                for target in row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []:
                    if target in duplicate_targets:
                        duplicate_links.append({'source1_entity_id': row['source1_entity_id'], 'candidate_entity_id': target})
    wanted = {r['source1_entity_id'] for r in duplicate_links}
    countries = {}
    country_counts = Counter()
    source_rows = 0
    with SOURCE1.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            owner = row['entity_id']
            if owner not in owners:
                raise ValueError('Source1 reference missing from accepted-link TSV')
            country = row['country'].strip().casefold()
            country_counts[country] += 1
            source_rows += 1
            if owner in wanted:
                countries[owner] = country
    if source_rows != len(owners) or source_rows != 1732544 or set(countries) != wanted:
        raise ValueError('Incomplete official Source1 / accepted output universe')
    by_country = {c: Counter() for c in country_counts}
    target_countries = defaultdict(set)
    for row in duplicate_links:
        country = countries[row['source1_entity_id']]
        row['country'] = country
        by_country[country][row['candidate_entity_id']] += 1
        target_countries[row['candidate_entity_id']].add(country)
    if sha(MATCHES) != matching_hash:
        raise ValueError('Accepted-link TSV changed during census')
    report = {'status': 'complete', 'scope': 'complete_official_test_submission04_accepted_links_label_free',
              'reference_rows': len(owners), 'country_reference_counts': dict(country_counts),
              'accepted_links': links, 'empty_reference_rows': empty, 'unique_accepted_targets': len(frequencies),
              'targets_owned_by_multiple_Source1': len(duplicate_targets),
              'duplicate_target_accepted_links': len(duplicate_links),
              'excess_accepted_owner_claims': sum(frequencies[c]-1 for c in duplicate_targets),
              'maximum_owners_for_one_target': max(frequencies.values(), default=0),
              'same_country_duplicate_targets': {c: sum(n>1 for n in counts.values()) for c,counts in by_country.items()},
              'global_duplicate_targets_touching_country': {c: len(counts) for c,counts in by_country.items()},
              'cross_country_duplicate_targets': sum(len(v)>1 for v in target_countries.values()),
              'matching_results_sha256': matching_hash, 'supplied_source1_sha256': sha(SOURCE1),
              'code_sha256': sha(Path(__file__)), 'labels_read': False,
              'output_TSVs_modified': False, 'release_policies_modified': False,
              'seconds': time.monotonic()-started}
    (OUT/'accepted_ownership_census.json').write_text(json.dumps(report, indent=2)+'\n')
    if duplicate_links:
        import pandas as pd
        pd.DataFrame(duplicate_links).to_parquet(OUT/'duplicate_accepted_links.parquet', index=False)
        (OUT/'duplicate_claimant_ids.json').write_text(json.dumps(sorted(wanted))+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
