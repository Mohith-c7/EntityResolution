"""Final-audit predictions must retain all competing heldout owners."""
import importlib.util
from pathlib import Path

import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "seal_last4_predictions",
    Path(__file__).resolve().parents[1] / "research/final_2h/seal_last4_predictions.py",
)


def test_external_owner_can_block_reserved_owner():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    frame = pd.DataFrame({
        "source1_entity_id": ["reserved", "other"],
        "candidate_entity_id": ["X", "X"],
        "probability": [.9, .95],
    })
    config = {"threshold": .83, "t_first": .7, "t_rest": .83}
    _, trial = module.replay_and_slice(frame, frame, config, ["reserved"])
    assert trial == {"reserved": set()}
