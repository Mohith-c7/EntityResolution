"""Read-only, full-file audit of downloaded challenge TSVs."""

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = Path.home() / 'Downloads'
OUTPUT = PROJECT_ROOT / 'reports/eda/local_download_audit.json'
SOURCE_COLUMNS = ['entity_id', 'business_name', 'business_address', 'country']
GT_COLUMNS = ['source1_entity_id', 'matched_entity_ids']
csv.field_size_limit(10_000_000)


def profile_source(name, prefix, retain=False, compare_ids=None):
    path = BASE / name
    before = path.stat()
    start = time.perf_counter()
    seen = set()
    countries = Counter()
    blanks = Counter()
    null_literals = Counter()
    non_ascii = Counter()
    duplicates = invalid_ids = malformed = overlap = total = 0
    errors = []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, delimiter='\t', strict=True)
        header = next(reader)
        if header != SOURCE_COLUMNS:
            raise ValueError(f'{name}: unexpected header {header!r}')
        for row in reader:
            total += 1
            if len(row) != 4:
                malformed += 1
                if len(errors) < 5:
                    errors.append({'line': reader.line_num, 'field_count': len(row)})
                continue
            eid, business_name, address, country = row
            if eid in seen:
                duplicates += 1
            seen.add(eid)
            if not eid.startswith(prefix) or len(eid) <= len(prefix) or any(c.isspace() or c == ',' for c in eid):
                invalid_ids += 1
            if compare_ids is not None and eid in compare_ids:
                overlap += 1
            countries[country.strip() or '<MISSING>'] += 1
            for column, value in zip(SOURCE_COLUMNS, row):
                if not value.strip():
                    blanks[column] += 1
                if value in ('null', 'NULL', 'None'):
                    null_literals[column] += 1
                if not value.isascii():
                    non_ascii[column] += 1
    after = path.stat()
    result = {
        'file': str(path), 'file_size_bytes': after.st_size, 'rows': total,
        'columns': header, 'unique_ids': len(seen), 'duplicate_ids': duplicates,
        'invalid_ids': invalid_ids, 'malformed_rows': malformed,
        'blank_or_whitespace_counts': {c: blanks[c] for c in SOURCE_COLUMNS},
        'null_literal_counts': {c: null_literals[c] for c in SOURCE_COLUMNS},
        'non_ascii_counts': {c: non_ascii[c] for c in SOURCE_COLUMNS},
        'countries': dict(countries), 'example_errors': errors,
        'unchanged_during_scan': (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        'seconds': time.perf_counter() - start,
    }
    if compare_ids is not None:
        result['same_source_train_test_id_overlap'] = overlap
    print(json.dumps({'file': name, 'rows': total, 'duplicates': duplicates,
                      'invalid_ids': invalid_ids, 'malformed_rows': malformed,
                      'countries': dict(countries), 'seconds': round(result['seconds'], 2)}), flush=True)
    if not retain:
        del seen
        gc.collect()
        return result, None
    return result, seen


def profile_truth(s1_ids, s2_ids, s3_ids):
    path = BASE / 'train_ground_truth.tsv'
    before = path.stat()
    start = time.perf_counter()
    seen = set()
    distribution = Counter()
    counts = Counter()
    target_owners = {}
    conflicting_targets = set()
    ownership_examples = []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, delimiter='\t', strict=True)
        header = next(reader)
        if header != GT_COLUMNS:
            raise ValueError(f'Unexpected ground-truth header: {header!r}')
        for row in reader:
            counts['rows'] += 1
            if len(row) != 2:
                counts['malformed_rows'] += 1
                continue
            sid, raw_matches = row
            if sid in seen:
                counts['duplicate_source1_rows'] += 1
            seen.add(sid)
            if not sid.startswith('S1-') or len(sid) <= 3 or any(c.isspace() or c == ',' for c in sid):
                counts['invalid_source1_ids'] += 1
            if sid not in s1_ids:
                counts['source1_ids_absent_from_training_source1'] += 1
            matches = raw_matches.split(',') if raw_matches else []
            if any(not m or m != m.strip() for m in matches):
                counts['rows_with_malformed_match_lists'] += 1
            distribution[len(matches)] += 1
            if len(matches) != len(set(matches)):
                counts['rows_with_duplicate_match_ids'] += 1
            for mid in matches:
                if mid.startswith('S2-'):
                    counts['source2_positive_links'] += 1
                    if mid not in s2_ids:
                        counts['source2_targets_absent_from_training_source2'] += 1
                elif mid.startswith('S3-'):
                    counts['source3_positive_links'] += 1
                    if mid not in s3_ids:
                        counts['source3_targets_absent_from_training_source3'] += 1
                else:
                    counts['invalid_match_prefixes'] += 1
                previous_owner = target_owners.setdefault(mid, sid)
                if previous_owner != sid:
                    counts['links_to_targets_owned_by_another_source1'] += 1
                    conflicting_targets.add(mid)
                    if len(ownership_examples) < 5:
                        ownership_examples.append({'target_id': mid, 'first_source1_id': previous_owner, 'other_source1_id': sid})
            if counts['rows'] % 500000 == 0:
                print(f'ground truth: {counts["rows"]:,} rows checked', flush=True)
    after = path.stat()
    n = counts['rows']
    links = sum(k * v for k, v in distribution.items())
    cumulative = 0
    lower_middle = (n - 1) // 2
    upper_middle = n // 2
    middle = []
    for k, frequency in sorted(distribution.items()):
        for rank in (lower_middle, upper_middle):
            if cumulative <= rank < cumulative + frequency:
                middle.append(k)
        cumulative += frequency
    result = {
        'file': str(path), 'file_size_bytes': after.st_size, 'columns': header,
        **{key: counts[key] for key in (
            'rows', 'malformed_rows', 'duplicate_source1_rows', 'invalid_source1_ids',
            'source1_ids_absent_from_training_source1', 'rows_with_duplicate_match_ids',
            'source2_positive_links', 'source3_positive_links',
            'source2_targets_absent_from_training_source2',
            'source3_targets_absent_from_training_source3',
            'invalid_match_prefixes', 'rows_with_malformed_match_lists',
            'links_to_targets_owned_by_another_source1')},
        'training_source1_ids_without_ground_truth': len(s1_ids - seen),
        'unique_source1_ids': len(seen), 'positive_links': links,
        'singleton_count': distribution[0], 'singleton_rate': distribution[0] / n if n else None,
        'mean_matches_per_source1': links / n if n else None,
        'median_matches_per_source1': sum(middle) / len(middle) if middle else None,
        'max_matches_per_source1': max(distribution, default=0),
        'match_count_distribution': dict(sorted(distribution.items())),
        'unique_matched_target_ids': len(target_owners),
        'targets_linked_to_multiple_source1_entities': len(conflicting_targets),
        'target_ownership_conflict_examples': ownership_examples,
        'source3_target_existence_check': 'completed full-file check',
        'cross_reference_target_ownership_check': 'completed full-file check',
        'unchanged_during_scan': (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        'seconds': time.perf_counter() - start,
    }
    print(json.dumps({'file': path.name, 'rows': n, 'positive_links': links,
                      'singleton_rate': result['singleton_rate'], 'seconds': round(result['seconds'], 2)}), flush=True)
    return result


def main():
    global BASE, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=BASE, help='Folder containing all seven challenge TSV files.')
    parser.add_argument('--output', type=Path, default=OUTPUT, help='JSON audit report path.')
    args = parser.parse_args()
    BASE, OUTPUT = args.input_dir.resolve(), args.output.resolve()
    required = ['train_ground_truth.tsv'] + [f'{split}_source{number}.tsv' for split in ('train', 'test') for number in (1, 2, 3)]
    missing = [name for name in required if not (BASE / name).is_file()]
    if missing:
        parser.error(f'Missing required files: {missing}')
    start = time.perf_counter()
    reports = {}
    reports['train_source1.tsv'], s1_ids = profile_source('train_source1.tsv', 'S1-', retain=True)
    reports['train_source2.tsv'], s2_ids = profile_source('train_source2.tsv', 'S2-', retain=True)
    reports['train_source3.tsv'], s3_ids = profile_source('train_source3.tsv', 'S3-', retain=True)
    truth = profile_truth(s1_ids, s2_ids, s3_ids)
    reports['test_source1.tsv'], _ = profile_source('test_source1.tsv', 'S1-', compare_ids=s1_ids)
    del s1_ids
    gc.collect()
    reports['test_source2.tsv'], _ = profile_source('test_source2.tsv', 'S2-', compare_ids=s2_ids)
    del s2_ids
    gc.collect()
    reports['test_source3.tsv'], _ = profile_source('test_source3.tsv', 'S3-', compare_ids=s3_ids)
    del s3_ids
    gc.collect()
    expected_path = PROJECT_ROOT / 'reports/eda/eda_profile_results.json'
    comparisons = {}
    if expected_path.exists():
        previous = json.loads(expected_path.read_text())
        for value in previous.get('datasets', {}).values():
            name = value['filename']
            if name in reports:
                comparisons[name] = {
                    'row_count_matches_committed_report': reports[name]['rows'] == value['total_rows'],
                    'byte_size_matches_committed_report': reports[name]['file_size_bytes'] == value['file_size_bytes'],
                }
    result = {
        'audit_type': 'full-file read-only scan of all seven required challenge TSV files',
        'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sources': reports, 'ground_truth': truth, 'previous_report_comparison': comparisons,
        'required_data_files_not_yet_audited': [],
        'elapsed_seconds': time.perf_counter() - start,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {OUTPUT}; total {result["elapsed_seconds"]:.2f}s', flush=True)
    source_errors = any(report[key] for report in reports.values() for key in ('duplicate_ids', 'invalid_ids', 'malformed_rows'))
    truth_errors = any(truth[key] for key in (
        'malformed_rows', 'duplicate_source1_rows', 'invalid_source1_ids',
        'source1_ids_absent_from_training_source1', 'rows_with_duplicate_match_ids',
        'source2_targets_absent_from_training_source2', 'source3_targets_absent_from_training_source3',
        'invalid_match_prefixes', 'rows_with_malformed_match_lists', 'training_source1_ids_without_ground_truth',
    ))
    changed = not all(report['unchanged_during_scan'] for report in [*reports.values(), truth])
    if source_errors or truth_errors or changed:
        raise SystemExit(1)
    if truth['targets_linked_to_multiple_source1_entities']:
        print('WARNING: ambiguous target ownership; inspect before imposing exclusivity.', flush=True)


if __name__ == '__main__':
    main()
