import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('hybrid_decision_probe', Path(__file__).resolve().parents[1]/'research/final_2h/hybrid_decision_probe.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cached_thresholds_preserve_rivals_rescue_and_order_ties():
    frame = pd.DataFrame({'source1_entity_id': ['S1-a','S1-a','S1-a','S1-b','S1-b','S1-c','S1-c'],
        'candidate_entity_id': ['S2-x','S2-y','S3-a','S2-x','S3-b','S2-c','S3-c'],
        'probability': [.89,.69,.69,.95,.64,.72,.72]})
    baseline = {'threshold': .83, 't_rest': .83, 't_first': .70}
    adjusted, first, keep = module.cached_decision_state(frame, baseline)
    assert not keep[0]  # Stronger competing owner wins x.
    assert keep[5] and not keep[6]  # Equal-probability rescue retains original order.
    for rest in [.79,.83,.87]:
        for rescue in [.60,.70,.80]:
            config = {**baseline, 't_rest': rest, 't_first': rescue}
            expected, _ = module.decide_control(frame, frame, config)
            assert np.array_equal(module.select(adjusted, first, config), expected)
