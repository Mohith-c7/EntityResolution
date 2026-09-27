import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1] / "research/final_2h/hybrid_audit/prepare_empty_route.py"
spec = importlib.util.spec_from_file_location("hybrid_empty_route", P)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_top4_then_exclude_does_not_refill_or_change_nonempty_owner():
    rows = [{"source1_entity_id": owner, "candidate_entity_id": f"S2-{i:03d}",
             "probability": .9 - .01 * i, "candidate_order": i}
            for owner in ("fixture-A", "fixture-B") for i in range(40)]
    frame = pd.DataFrame(rows)
    selected = np.zeros(len(frame), dtype=bool); selected[40] = True
    main = frame.iloc[[0]][module.KEYS]
    all40, extra, empty_count, overlap = module.select_disjoint_top4(frame, selected, main)
    assert empty_count == 1 and overlap == 1
    assert len(all40) == 40 and set(all40.source1_entity_id) == {"fixture-A"}
    assert extra.candidate_entity_id.tolist() == ["S2-001", "S2-002", "S2-003"]
    assert "S2-004" not in set(extra.candidate_entity_id)


def test_equal_probability_uses_target_id_not_candidate_order():
    frame = pd.DataFrame([{"source1_entity_id": "fixture-A", "candidate_entity_id": f"S2-{i:03d}",
                           "probability": .8, "candidate_order": 39 - i} for i in range(39, -1, -1)])
    main = frame.iloc[0:0][module.KEYS]
    _, extra, _, overlap = module.select_disjoint_top4(frame, np.zeros(40, dtype=bool), main)
    assert overlap == 0
    assert extra.candidate_entity_id.tolist() == ["S2-000", "S2-001", "S2-002", "S2-003"]
