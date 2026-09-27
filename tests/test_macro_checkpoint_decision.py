import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/final_2h'))
from macro_checkpoint_probe import FastDecision, CONFIG
from evaluate_sprint import decide_control, predictions_for, entity_f05
from post_selection_exclusivity import exclusive_selected


def test_fast_decisions_match_competition_rescue_and_ties():
    rng = np.random.default_rng(123)
    for _ in range(25):
        rows = [(f'S1-{e:03d}', f'S2-{c:03d}', rng.choice([0., .5, .7, .79, .8, .83, .9, 1.]))
                for e in range(12) for c in rng.choice(30, 8, replace=False)]
        rng.shuffle(rows)
        frame = pd.DataFrame(rows, columns=['source1_entity_id', 'candidate_entity_id', 'probability'])
        expected, _ = decide_control(frame, frame, CONFIG)
        expected, _ = exclusive_selected(frame, expected)
        engine = FastDecision(frame)
        actual = engine.choose(frame.probability.to_numpy())
        assert np.array_equal(actual, expected)
        truth = {f'S1-{i:03d}': ([] if i % 3 == 0 else [f'S2-{i:03d}', f'S2-{i+1:03d}']) for i in range(6)}
        mask = frame.source1_entity_id.isin(truth).to_numpy()
        predictions = predictions_for(frame[mask], expected[mask], truth)
        reference = np.mean([entity_f05(truth[e], predictions[e]) for e in truth])
        assert abs(engine.macro(actual, engine.metric_inputs(frame, truth)) - reference) < 1e-12
