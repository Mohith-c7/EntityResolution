"""Choose native matcher checkpoints on exposed early macro F0.5 only.

Keeps the hybrid graph and fixed-four empty rescue unchanged. No fresh audit,
test data, neural fitting or production artifacts are read or modified.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'research/final_2h')]
import numpy as np
import pandas as pd
import lightgbm as lgb
import neural_peer as peer
from evaluate_sprint import decide_control, predictions_for, paired_evaluate
from post_selection_exclusivity import exclusive_selected
from export_sprint_workbench import sha

KEYS = ['source1_entity_id', 'candidate_entity_id']
CONFIG = {'threshold': .83, 't_rest': .79, 't_first': .6999999999999998}
B = Path('research/sprint_6h/sibling')
OUT = Path('research/final_2h/macro_checkpoint_v1')


class FastDecision:
    """Exact original arbitration, first-row ties, then selected-only ownership."""
    def __init__(self, frame):
        if frame.duplicated(KEYS).any(): raise ValueError('Duplicate pair')
        self.owners, self.owner_ids = pd.factorize(frame.source1_entity_id, sort=True)
        self.targets, self.target_ids = pd.factorize(frame.candidate_entity_id, sort=False)
        self.rows = np.arange(len(frame))

    def choose(self, probability, config=CONFIG):
        p = np.asarray(probability)
        if p.shape != self.rows.shape or not np.isfinite(p).all(): raise ValueError('Invalid probability')
        top = np.full(len(self.target_ids), -np.inf)
        np.maximum.at(top, self.targets, p)
        lost = (p >= config['threshold']) & (top[self.targets] > p)
        adjusted = np.where(lost, 0., p)
        best = np.full(len(self.owner_ids), -np.inf)
        np.maximum.at(best, self.owners, adjusted)
        ties = adjusted == best[self.owners]
        first = np.full(len(self.owner_ids), len(p), dtype=np.int64)
        np.minimum.at(first, self.owners[ties], self.rows[ties])
        chosen = (adjusted >= config['t_rest']) | ((self.rows == first[self.owners]) & (adjusted >= config['t_first']))
        top.fill(-np.inf)
        np.maximum.at(top, self.targets[chosen], p[chosen])
        tied = chosen & (p == top[self.targets])
        winner = np.full(len(self.target_ids), len(self.owner_ids), dtype=np.int64)
        np.minimum.at(winner, self.targets[tied], self.owners[tied])
        return tied & (self.owners == winner[self.targets])

    def metric_inputs(self, frame, truth):
        lookup = {e: set(values) for e, values in truth.items()}
        truth_size = np.array([len(lookup.get(e, ())) for e in self.owner_ids])
        mask = np.array([e in lookup for e in self.owner_ids])
        y = np.fromiter((c in lookup.get(e, ()) for e, c in frame[KEYS].itertuples(index=False, name=None)), bool, len(frame))
        return truth_size, mask, y

    def macro(self, chosen, prepared):
        size, mask, y = prepared
        count = np.bincount(self.owners[chosen], minlength=len(size))
        tp = np.bincount(self.owners[chosen & y], minlength=len(size))
        denominator = count + .25 * size
        value = np.divide(1.25 * tp, denominator, out=np.ones(len(size)), where=denominator > 0)
        return float(value[mask].mean())


def load_views():
    head = Path('models/final_2h/neural_last8_continue_v1/manifest.json')
    result = []
    for features, scope, raw in [('neural_v1_features_fit', 'residual20k', 'residual20k'),
                                 ('neural_v1_features30k', 'learned30k', 'learned30k')]:
        frame, marker = peer.nn.load_features(B / features)
        frame = peer.add_fine(frame, marker, Path('models/sprint_6h/neural') / ('last8_' + scope),
                              Path('models/sprint_6h/neural/neural_inputs') / raw / 'manifest.json', head)
        result.append(frame)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    checkpoints = [80, 160, 320, 600]
    shifts = [-.25, 0., .25, .5]
    recipes = ['anchored', 'direct', 'anchored_missing2x']
    protocol = {'status': 'declared_before_training_and_early_selection',
        'declared_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
        'checkpoints': checkpoints, 'shifts': shifts, 'recipes': recipes,
        'selection': 'Maximum early10k macro; fewer trees, smaller absolute shift, recipe order break ties. Evaluate winner once on exposed selection20k.',
        'decision_config': CONFIG, 'post_selection_exclusivity': True,
        'baseline': 'hybrid8_empty4_calibrated_exclusive_v1',
        'fresh_audit_labels_read': False, 'test_records_read': False}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    train, dev = load_views()
    names = [*peer.nn.FEATURES, 'neural_fine_probability']
    truth = json.loads((B / 'workbenches50k/residual_train/truth.json').read_text())
    early = json.loads((B / 'workbenches50k/early_stop/truth.json').read_text())
    if len(truth) != 20000 or len(early) != 10000 or set(train.source1_entity_id) & set(dev.source1_entity_id):
        raise ValueError('Training/development roles differ')
    y = np.array([int(c in truth[e]) for e, c in train[KEYS].itertuples(index=False, name=None)])
    frame = pd.read_parquet('research/final_2h/fixed_hybrid_v1/pairs.parquet')
    if len(frame) != 1200000 or frame.source1_entity_id.nunique() != 30000:
        raise ValueError('Full30k graph required')
    positions = pd.MultiIndex.from_frame(frame[KEYS]).get_indexer(pd.MultiIndex.from_frame(dev[KEYS]))
    if (positions < 0).any(): raise ValueError('Foreign neural pair')
    engine = FastDecision(frame)
    original_mask, _ = decide_control(frame, frame, CONFIG)
    original_mask, _ = exclusive_selected(frame, original_mask)
    if not np.array_equal(engine.choose(frame.probability.to_numpy()), original_mask):
        raise ValueError('Actual30k fast-decision parity failed')
    prepared = engine.metric_inputs(frame, early)
    original = frame.probability.to_numpy().copy()
    base_logit = peer.nn.reverse.base_logit(dev.probability)
    weights = 1 / train.groupby('source1_entity_id').source1_entity_id.transform('size').to_numpy()
    params = dict(objective='binary', metric='None', num_leaves=31, min_data_in_leaf=40,
        lambda_l2=1., learning_rate=.05, max_bin=127, num_threads=12,
        deterministic=True, force_col_wise=True, seed=42, verbosity=-1)
    results = []; best = None
    for recipe in recipes:
        anchored = recipe != 'direct'
        weight = weights * (np.where(train.reverse_target_address_missing.to_numpy() > 0, 2., 1.) if recipe.endswith('2x') else 1.)
        data = lgb.Dataset(train[names].to_numpy(dtype=np.float32), label=y, weight=weight,
                           init_score=peer.nn.reverse.base_logit(train.probability) if anchored else None,
                           feature_name=names)
        model = lgb.train({**params, 'boost_from_average': not anchored}, data, num_boost_round=max(checkpoints))
        model.save_model(str(OUT / (recipe + '_600.txt')))
        for trees in checkpoints:
            raw = model.predict(dev[names].to_numpy(dtype=np.float32), raw_score=True, num_iteration=trees, num_threads=12)
            for shift in shifts:
                p = original.copy()
                logit = raw + shift + (base_logit if anchored else 0.)
                p[positions] = 1 / (1 + np.exp(-np.clip(logit, -50, 50)))
                score = engine.macro(engine.choose(p), prepared)
                row = {'recipe': recipe, 'trees': trees, 'shift': shift, 'early_macro_f05': score}
                results.append(row)
                key = (-score, trees, abs(shift), recipes.index(recipe), shift)
                if best is None or key < best[0]: best = (key, row, p.copy())
        print(json.dumps({'recipe': recipe, 'best_early': best[1]}), flush=True)
    _, chosen, probability = best
    (OUT / 'chosen.json').write_text(json.dumps(chosen, indent=2) + '\n')
    selection = json.loads((B / 'workbenches50k/selection/truth.json').read_text())
    countries = json.loads((B / 'workbenches50k/selection/countries.json').read_text())
    if len(selection) != 20000 or set(selection) & (set(early) | set(truth)): raise ValueError('Selection roles overlap')
    keep = engine.choose(probability)
    mask = frame.source1_entity_id.isin(selection).to_numpy()
    report = paired_evaluate(selection, predictions_for(frame[mask], original_mask[mask], selection),
        predictions_for(frame[mask], keep[mask], selection), countries, role='development')
    report['legacy_guard_diagnostics'] = report.pop('acceptance')
    changed = frame.copy(deep=False); changed['probability'] = probability
    changed.to_parquet(OUT / 'pairs.parquet', index=False)
    model = lgb.Booster(model_file=str(OUT / (chosen['recipe'] + '_600.txt')))
    model.save_model(str(OUT / 'adapter.txt'), num_iteration=chosen['trees'])
    report.update(status='exposed_development_only', chosen=chosen, early_grid=results, features=names,
        model_sha256=sha(OUT / 'adapter.txt'), source_sha256=sha(__file__),
        baseline_early_macro_f05=engine.macro(original_mask, prepared),
        development_gain_confirmed=report['paired_macro_f05_delta'] >= .0003 and report['paired_delta_95pct_ci'][0] > 0 and min(report['country_deltas'].values()) >= -.0005,
        fresh_audit_labels_read=False, test_records_read=False, full_graph_decision_parity=True)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['chosen', 'paired_macro_f05_delta', 'paired_delta_95pct_ci', 'development_gain_confirmed']}, indent=2))


if __name__ == '__main__': main()
