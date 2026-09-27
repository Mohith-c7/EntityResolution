"""Development-only exact loss ledger for frozen four-layer decisions.

No model fitting or new cohorts. Empty non-singletons form a separate stratum;
on other owners Shapley allocation over three oracle error repairs makes the
loss categories additive despite F0.5 interactions. Oracle gains are diagnostic
upper bounds, not evidence that a learnable correction can recover them.
"""
import argparse
import csv
from collections import defaultdict
import itertools
import json
from pathlib import Path
import sys
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts')]
from evaluate_sprint import decide_control, predictions_for, entity_f05
from export_sprint_workbench import sha

CATEGORIES = ['missing_candidates', 'retrieved_rejected_true_links', 'incorrect_positive_links', 'empty_non_singletons']


def allocation(truth, prediction, retrieved):
    truth, prediction, retrieved = map(set, (truth, prediction, retrieved))
    missing = truth - retrieved
    rejected = (truth & retrieved) - prediction
    false = prediction - truth
    loss = 1 - entity_f05(truth, prediction)
    if truth and not prediction:
        return [0., 0., 0., loss], missing, rejected, false
    repairs = [missing, rejected, false]
    result = [0., 0., 0., 0.]
    for order in itertools.permutations(range(3)):
        current = prediction.copy()
        previous = entity_f05(truth, current)
        for i in order:
            current = current - repairs[i] if i == 2 else current | repairs[i]
            now = entity_f05(truth, current)
            result[i] += (now - previous) / 6
            previous = now
    if not np.isclose(sum(result), loss, atol=1e-12):
        raise ValueError('Oracle allocation does not reconcile with exact loss')
    return result, missing, rejected, false


def text_classes(record):
    name, address = record.get('business_name', '') or '', record.get('business_address', '') or ''
    letters = [c for c in name if c.isalpha()]
    script = 'non_latin' if any('LATIN' not in unicodedata.name(c, '') for c in letters) else 'latin'
    return {'script': script if letters else 'no_letters',
        'name': 'empty' if not name.strip() else ('short_1word' if len(name.split()) <= 1 else ('long_5plus_words' if len(name.split()) >= 5 else '2_to_4_words')),
        'address': 'empty' if not address.strip() else ('has_digits' if any(c.isdigit() for c in address) else 'no_digits')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('research/final_2h/current4_remaining_loss_v1'))
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    graph = Path('research/final_2h/neural_last4_fine_v1/pairs.parquet')
    base = Path('research/sprint_6h/sibling/learned30k/pairs.parquet')
    wb = Path('research/sprint_6h/sibling/workbenches50k/selection')
    route_file = Path('research/sprint_6h/sibling/neural_v1_features30k/features.parquet')
    records_file = Path('dataset/train/train_source1.tsv')
    truth = {e:set(v) for e,v in json.loads((wb/'truth.json').read_text()).items()}
    countries = json.loads((wb/'countries.json').read_text())
    if len(truth) != 20000 or set(countries) != set(truth):
        raise ValueError('Only authorized existing selection20k is allowed')
    frame = pd.read_parquet(graph)
    original = pd.read_parquet(base, columns=['source1_entity_id', 'candidate_entity_id', 'candidate_order', 'probability'])
    keys = ['source1_entity_id','candidate_entity_id','candidate_order']
    if not frame[keys].equals(original[keys]) or not np.array_equal(frame.probability_original, original.probability):
        raise ValueError('Frozen graph/anchor changed')
    route = pd.read_parquet(route_file, columns=keys[:2])
    routed_owners = set(route.source1_entity_id)
    route_keys = set(zip(route.source1_entity_id, route.candidate_entity_id))
    config_file = Path('research/sprint_6h/coordination/baseline_freeze/frozen.json')
    config = json.loads(config_file.read_text())
    config = {k:config[k] for k in ['threshold','t_first','t_rest']}
    chosen, lost = decide_control(frame, frame, config)
    mask = frame.source1_entity_id.isin(truth).to_numpy()
    selection = frame.loc[mask]
    predictions = predictions_for(selection, chosen[mask], truth)
    retrieved = defaultdict(set)
    ownership_lost_true = defaultdict(set)
    for e,c,l in zip(selection.source1_entity_id, selection.candidate_entity_id, lost[mask]):
        retrieved[e].add(c)
        if l and c in truth[e]: ownership_lost_true[e].add(c)
    records = {}
    if records_file.exists():
        # Read Source1 text only; retain only the already exposed selection IDs.
        with records_file.open(newline='') as stream:
            for r in csv.DictReader(stream, delimiter='\t'):
                if r['entity_id'] in truth:
                    records[r['entity_id']] = r
    rows, grouped = [], defaultdict(lambda: {'owners':0,'error_owners':0,'loss_sum':0.,'category_loss_sum':[0.]*4,'missing_links':0,'rejected_links':0,'incorrect_links':0,'empty_non_singletons':0})
    counts = defaultdict(int)
    standalone = np.zeros(3)
    route_oracle = defaultdict(float)
    for e,t in truth.items():
        pred, cand = predictions[e], retrieved[e]
        values, missing, rejected, false = allocation(t,pred,cand)
        score = entity_f05(t,pred)
        empty = bool(t and not pred)
        for i,newpred in enumerate([pred|missing, pred|rejected, pred-false]):
            standalone[i] += entity_f05(t,newpred)-score
        is_route = e in routed_owners
        stratum = 'currently_routed_owners' if is_route else 'excluded_owners'
        route_oracle[stratum + '_existing_candidate_perfect_gain'] += entity_f05(t,t & cand)-score
        if is_route:
            corrected = {c for c in pred if (e,c) not in route_keys} | {c for c in t if (e,c) in route_keys}
            route_oracle['current_exact_scored_pair_perfect_gain'] += entity_f05(t,corrected)-score
        counts['missing_links'] += len(missing); counts['retrieved_rejected_true_links'] += len(rejected)
        counts['incorrect_positive_links'] += len(false); counts['empty_non_singletons'] += empty
        counts['missing_links_routed_owners'] += len(missing) if is_route else 0
        counts['missing_error_routed_owners'] += bool(missing and is_route)
        counts['missing_error_unrouted_owners'] += bool(missing and not is_route)
        counts['rejected_true_links_in_scored_route'] += sum((e,c) in route_keys for c in rejected)
        counts['incorrect_links_in_scored_route'] += sum((e,c) in route_keys for c in false)
        counts['rejected_true_links_lost_to_global_rival'] += len(rejected & ownership_lost_true[e])
        classes = text_classes(records[e]) if e in records else {'script':'unavailable','name':'unavailable','address':'unavailable'}
        labels = [('country', countries[e]), ('neural_route', 'routed' if is_route else 'unrouted')] + list(classes.items())
        for field,label in labels:
            g = grouped[field+'='+label]; g['owners']+=1; g['error_owners']+=score<1; g['loss_sum']+=1-score
            g['category_loss_sum'] = [x+y for x,y in zip(g['category_loss_sum'],values)]
            for key,n in [('missing_links',len(missing)),('rejected_links',len(rejected)),('incorrect_links',len(false)),('empty_non_singletons',empty)]: g[key]+=n
        if score<1:
            rows.append({'entity_id':e,'country':countries[e], 'routed_owner':is_route,'classes':classes,'entity_f05':score,
                'additive_loss':dict(zip(CATEGORIES,values)), 'missing_targets':sorted(missing),'rejected_targets':sorted(rejected),
                'incorrect_targets':sorted(false),'empty_non_singleton':empty,'ownership_lost_true_targets':sorted(rejected & ownership_lost_true[e])})
    loss = sum(1-r['entity_f05'] for r in rows)/len(truth)
    additive = {c:sum(r['additive_loss'][c] for r in rows)/len(truth) for c in CATEGORIES}
    if not np.isclose(sum(additive.values()),loss,atol=1e-12): raise ValueError('Ledger mismatch')
    for g in grouped.values():
        g['macro_loss_contribution'] = g['loss_sum']/len(truth)
        g['within_group_macro_loss'] = g['loss_sum']/g['owners']
        g['additive_macro_loss'] = dict(zip(CATEGORIES,[v/len(truth) for v in g.pop('category_loss_sum')]))
    replay_macro = float(np.mean([entity_f05(truth[e],predictions[e]) for e in sorted(truth)]))
    if not np.isclose(replay_macro,.9802609267254342,atol=1e-14,rtol=0):
        raise ValueError('Frozen selected four-layer selection replay mismatch')
    a.output.mkdir(parents=True)
    report = {'scope':'Exposed development selection20k only; global decisions computed on complete30k graph',
        'status':'development_loss_ledger_complete','source_sha256':sha(__file__), 'entities':len(truth),'error_entities':len(rows),
        'macro_f05':replay_macro,'exact_total_macro_loss':loss,'additive_oracle_recoverable_loss':additive,
        'claimant_graph_scope':'Frozen development30k only. Does not establish ownership parity for full331k heldout graph or official test.',
        'selected_selection_replay_verified':True,'expected_selected_macro_f05':.9802609267254342,
        'attribution':'Empty non-singletons are a separate stratum repaired fully. Other owners use equal Shapley allocation over adding missing truths, adding retrieved/rejected truths, and deleting false positives. All four categories sum exactly to 1-macro F0.5.',
        'singleton_convention':'Empty truth scores 1 only with empty prediction; any incorrect positive costs the whole singleton score.',
        'overlapping_standalone_oracle_gains':dict(zip(CATEGORIES[:3],(standalone/len(truth)).tolist())),
        'route_oracle_bounds':{k:v/len(truth) for k,v in route_oracle.items()},
        'route_bound_definition':'Perfect label-informed classification of fixed retrieved candidates on routed/excluded owners; missing candidates remain missing. Exact-scored-pair bound corrects only keys currently scored. These independent oracle prediction sets do not demonstrate globally feasible ownership corrections.',
        'oracle_limit':'Diagnostic label-informed upper bounds, not measured achievable model gain. Missing targets cannot themselves be inside existing pair route; routed-owner counts indicate affected owners only.',
        'counts':dict(counts),'groups':dict(grouped),'source1_records_available':len(records),
        'class_definitions':'Source1 name Unicode letter script; name whitespace word-count; address empty/digits. Groups overlap across dimensions and must not be summed across dimensions.',
        'input_sha256':{str(x):sha(x) for x in [graph,base,wb/'truth.json',wb/'countries.json',route_file,config_file]+([records_file] if records_file.exists() else [])},
        'audit_labels_read':False,'extension_labels_read':False,'production_changes':False}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'error_entities.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['macro_f05','error_entities','additive_oracle_recoverable_loss','counts','source1_records_available']},indent=2))


if __name__=='__main__': main()
