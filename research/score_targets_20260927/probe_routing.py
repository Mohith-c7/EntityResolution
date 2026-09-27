"""Local development-only routing ceilings; never reads test or audit labels.

An oracle here substitutes ground-truth decisions on routed pairs. It is an
optimistic feasibility bound, not a trained-model result or a shipping rule.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa


def scores(chosen, label, group, expected):
    counts = np.bincount(group[chosen], minlength=len(expected))
    tp = np.bincount(group[chosen & label], minlength=len(expected))
    values = np.zeros(len(expected))
    denom = counts + .25 * expected
    np.divide(1.25 * tp, denom, out=values, where=denom > 0)
    values[(expected == 0) & (counts == 0)] = 1.
    return values, counts, tp


def main():
    os.nice(10)
    pa.set_cpu_count(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True,
                        help='Development parquet with IDs, raw_probability and adjusted_probability joined by pair ID at export')
    parser.add_argument('--truth', type=Path, required=True,
                        help='Development truth JSON, including references with no candidates')
    parser.add_argument('--report', type=Path, required=True,
                        help='Decision report for this exact development export')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    truth_path = args.truth
    truth = {e: set(v) for e, v in json.loads(truth_path.read_text()).items()}
    frame = pd.read_parquet(args.pairs, columns=[
        'source1_entity_id', 'candidate_entity_id', 'raw_probability', 'adjusted_probability'])
    if frame.duplicated(['source1_entity_id', 'candidate_entity_id']).any():
        raise ValueError('Duplicate pairs')
    if not set(frame.source1_entity_id).issubset(truth):
        raise ValueError('Pairs reference entities outside development truth')
    ids = sorted(truth)
    group = frame.source1_entity_id.map({e: i for i, e in enumerate(ids)}).to_numpy(dtype=np.int32)
    expected = np.array([len(truth[e]) for e in ids])
    label = np.array([c in truth[e] for e, c in frame[
        ['source1_entity_id', 'candidate_entity_id']].itertuples(index=False, name=None)], dtype=bool)
    adjusted = frame.adjusted_probability.to_numpy()
    raw = frame.raw_probability.to_numpy()
    if len(frame) == 0 or not np.isfinite(adjusted).all() or not np.isfinite(raw).all():
        raise ValueError('Empty pairs or nonfinite probabilities')
    if ((raw < 0) | (raw > 1) | (adjusted < 0) | (adjusted > 1)).any():
        raise ValueError('Probabilities outside [0, 1]')
    order = np.lexsort((-adjusted, group))
    first = np.zeros(len(frame), dtype=bool)
    first[order[np.r_[True, group[order][1:] != group[order][:-1]]]] = True
    report_path = args.report
    reference = json.loads(report_path.read_text())['rank_one_rule']
    chosen = (adjusted >= reference['t_rest']) | (first & (adjusted >= reference['t_first']))
    baseline, count, tp = scores(chosen, label, group, expected)
    if not np.isclose(baseline.mean(), reference['macro_f05'], atol=1e-12, rtol=0):
        raise ValueError(f'Cannot reproduce development baseline: {baseline.mean()}')
    # Raw model rank is available before any labels are read for evaluation.
    order = np.lexsort((-raw, group))
    starts = np.r_[0, np.flatnonzero(group[order][1:] != group[order][:-1]) + 1]
    rank = np.empty(len(frame), dtype=np.int32)
    rank[order] = np.arange(len(frame)) - np.repeat(starts, np.diff(np.r_[starts, len(frame)])) + 1
    empty = count[group] == 0
    gates = {
        'probability_045_095': (raw >= .45) & (raw <= .95),
        'probability_020_099': (raw >= .20) & (raw <= .99),
        'probability_020_099_or_empty_top3': ((raw >= .20) & (raw <= .99)) | (empty & (rank <= 3)),
        'probability_010_0995_or_empty_top5': ((raw >= .10) & (raw <= .995)) | (empty & (rank <= 5)),
        'all_top5': rank <= 5,
        'all_candidates': np.ones(len(frame), dtype=bool),
    }
    rows = []
    for name, gate in gates.items():
        oracle = np.where(gate, label, chosen)
        values, _, _ = scores(oracle, label, group, expected)
        rows.append({'gate': name, 'pairs': int(gate.sum()), 'fraction_pairs': float(gate.mean()),
                     'references': int(len(np.unique(group[gate]))),
                     'oracle_macro_f05': float(values.mean()), 'oracle_gain': float((values-baseline).mean()),
                     'false_negatives_in_gate': int((gate & label & ~chosen).sum()),
                     'false_positives_in_gate': int((gate & ~label & chosen).sum())})
    retrieved = np.bincount(group[label], minlength=len(ids))
    fp = count - tp
    losses = dict(singleton_false_positive=0., empty_with_true_candidate=0., empty_without_true_candidate=0.,
                  false_links_in_nonempty=0., rejected_retrieved_in_nonempty=0., unretrieved_in_nonempty=0.)
    for i, n in enumerate(expected):
        if n == 0:
            losses['singleton_false_positive'] += float(count[i] > 0)
        elif count[i] == 0:
            losses['empty_with_true_candidate' if retrieved[i] else 'empty_without_true_candidate'] += 1.
        else:
            clean = 1.25 * tp[i] / (tp[i] + .25 * n)
            fill = 1.25 * retrieved[i] / (retrieved[i] + .25 * n)
            losses['false_links_in_nonempty'] += clean - baseline[i]
            losses['rejected_retrieved_in_nonempty'] += fill - clean
            losses['unretrieved_in_nonempty'] += 1. - fill
    losses = {k: float(v / len(ids)) for k, v in losses.items()}
    assert np.isclose(sum(losses.values()), 1-baseline.mean())
    result = {'status': 'development_oracle_diagnostic_not_model_accuracy', 'references': len(ids), 'pairs': len(frame),
              'baseline_macro_f05': float(baseline.mean()), 'singleton_false_positives': int(((expected==0)&(count>0)).sum()),
              'empty_non_singletons': int(((expected>0)&(count==0)).sum()),
              'losses': losses, 'gates': rows,
              'assumption': 'Perfect replacement of decisions on routed pairs; other final decisions fixed. Joint ownership constraints are relaxed. Not achievable performance evidence.',
              'input_hashes': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [truth_path, args.pairs, report_path]},
              'audit_labels_read': False, 'test_data_read': False, 'vm_modified': False}
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'input_hashes'}, indent=2))


if __name__ == '__main__':
    main()
