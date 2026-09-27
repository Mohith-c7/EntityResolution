import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

path = Path(__file__).resolve().parents[1] / 'research/final_2h/post_selection_exclusivity.py'
spec = importlib.util.spec_from_file_location('post_selection_exclusivity', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_unselected_higher_probability_cannot_steal_target_and_fallback_is_preserved():
    # The first owner's .79 rescue is selected; its losing .90 candidate remains
    # unselected. The second owner's lower .78 claim creates the sole collision.
    frame = pd.DataFrame([
        ('S1-a', 'S2-lost', .90), ('S1-a', 'S2-shared', .79),
        ('S1-b', 'S2-shared', .78), ('S1-c', 'S2-shared', .82),
        ('S1-c', 'S3-owner-c-rank-one', .95),
        ('S1-d', 'S2-unrelated-fallback', .71),
    ], columns=['source1_entity_id','candidate_entity_id','probability'])
    chosen = np.array([False, True, True, False, True, True])
    final, removed = module.exclusive_selected(frame, chosen)
    assert final.tolist() == [False, True, False, False, True, True]
    assert removed.tolist() == [False, False, True, False, False, False]
    assert chosen.tolist() == [False, True, True, False, True, True]
    assert frame.loc[final].candidate_entity_id.is_unique


def test_exact_ties_use_source_id_and_singleton_selection_stays_selected():
    frame = pd.DataFrame([
        ('S1-z','S2-shared',.99), ('S1-a','S2-shared',.99), ('S1-z','S3-alone',.72),
    ], columns=['source1_entity_id','candidate_entity_id','probability'])
    final, removed = module.exclusive_selected(frame, [True,True,True])
    assert final.tolist() == [False,True,True]
    assert removed.tolist() == [True,False,False]


def test_wrapper_preserves_unchanged_control_fallback_after_competition():
    frame = pd.DataFrame([
        ('S1-a','S2-rival',.90), ('S1-a','S2-fallback',.75),
        ('S1-b','S2-rival',.99), ('S1-c','S2-fallback',.74),
        ('S1-d','S2-alone',.71),
    ], columns=['source1_entity_id','candidate_entity_id','probability'])
    final, control_lost, exclusivity_lost = module.decide_post_selection(
        frame, frame, {'threshold':.83,'t_first':.7,'t_rest':.83})
    assert control_lost.tolist() == [True,False,False,False,False]
    assert final.tolist() == [False,True,True,False,True]
    assert exclusivity_lost.tolist() == [False,False,False,True,False]
