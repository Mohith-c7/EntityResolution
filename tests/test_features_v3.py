"""Regression tests for additive evidence and model/schema compatibility."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.contracts import Candidate
from src.blocking.disk_index import normalize_record
from src.features.evidence import numeric_relation, romanize, phonetic
from src.features.pairwise_features import build_v3_evidence, build_versioned_pair_features
from src.features.registry import FEATURE_VERSION_V2, FEATURE_VERSION_V3, FEATURE_NAMES_V2, FEATURE_NAMES_V3


class Index:
    aliases = None
    def df(self, col, token): return 3
    def idf(self, col, token): return 7.0
    def resolved_name(self, record): return record.core, 0, 0
    def family_count(self, core): return 2
    def address_containment(self, left, right): return 0
    def weighted_overlap(self, left, right, column): return 0


def rec(eid, name="Alpha Limited", address="1812 Elm Street"):
    return normalize_record(dict(entity_id=eid, business_name=name, business_address=address, country="New Country"))


def test_numeric_noise_types_and_no_missing_contradiction():
    assert numeric_relation(["1812"], ["812"])[:3] == (1, 0, 0)
    assert numeric_relation(["322"], ["323"])[:3] == (0, 1, 1)
    assert numeric_relation([], ["323"])[:3] == (0, 0, 0)
    f = build_v3_evidence(rec("S1-A", address="gui1len hurtad0 0012"), rec("S2-A", address="12"), Index())
    assert f["num_left_only"] == f["num_right_only"] == 0
    # Primary view is preserved rather than corrupting genuine unit numbers.
    assert rec("S1-A", address="B12 12A").address_digits == {"12"}


def test_romanization_word_boundaries_and_unicode_digits():
    word = "इम्पेक्स"
    assert romanize(word + " " + word) == romanize(word) + " " + romanize(word)
    assert romanize(word).endswith("ks")
    assert romanize("१२३") == "123"
    assert phonetic("gui1len") == phonetic("guillen")


def test_v3_preserves_v2_and_handles_empty_fields():
    candidate = Candidate("S1-A", "S2-A", "S2", .7, ("rare_name",))
    for left, right in [(rec("S1-A"), rec("S2-A")), (rec("S1-A", "", ""), rec("S2-A", "", ""))]:
        old = build_versioned_pair_features(left, right, candidate, Index(), FEATURE_VERSION_V2)
        new = build_versioned_pair_features(left, right, candidate, Index(), FEATURE_VERSION_V3)
        assert tuple(old) == FEATURE_NAMES_V2
        assert tuple(new) == FEATURE_NAMES_V3
        assert {k: new[k] for k in old} == old
        assert np.isfinite(list(new.values())).all()
    assert new["addr_left_unmatched_idf"] == -1
    assert new["concat_ratio"] == 0
    with pytest.raises(ValueError):
        build_versioned_pair_features(left, right, candidate, Index(), "unknown")


def test_saved_v3_model_scores_exact_retained_candidates(tmp_path):
    import lightgbm as lgb
    import pandas as pd
    from src.blocking.disk_index import DiskSearchConfig, DiskSourceIndex, build_disk_index
    from src.pipeline import inference
    paths = []
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    config = DiskSearchConfig(top_k=2, path_top_k=5, character_mode="off")
    raw = dict(entity_id="S1-A", business_name="Alpha Limited", business_address="1812 Elm Street", country="New Country")
    reference = normalize_record(raw)
    expected_ids, feature_rows = [], []
    for source in ("S2", "S3"):
        source_file = tmp_path / f"{source}.tsv"
        source_file.write_text(header + f"{source}-A\tAlpha Limited\t1812 Elm Street\tNew Country\n")
        path = tmp_path / f"{source}.sqlite"
        build_disk_index(source_file, path, source)
        paths.append(path)
        with_index = DiskSourceIndex(path, config)
        for candidate, target in with_index.query(reference):
            expected_ids.append(candidate.candidate_entity_id)
            feature_rows.append(build_versioned_pair_features(reference, target, candidate, with_index, FEATURE_VERSION_V3))
        with_index.close()
    frame = pd.DataFrame(feature_rows * 10, columns=FEATURE_NAMES_V3)
    negative = frame.copy()
    negative.loc[:, "concat_ratio"] = 0
    frame = pd.concat([frame, negative], ignore_index=True)
    model = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, min_child_samples=1, n_jobs=1, verbosity=-1)
    model.fit(frame, [1] * 20 + [0] * 20)
    path = tmp_path / "model.txt"
    model.booster_.save_model(str(path))
    expected_probs = model.predict_proba(pd.DataFrame(feature_rows, columns=FEATURE_NAMES_V3))[:, 1]
    inference.initialize_worker(path, paths, config, .5, FEATURE_VERSION_V3)
    try:
        result = inference.score_batch([raw])
        assert result == [("S1-A", expected_ids, sorted(cid for cid, p in zip(expected_ids, expected_probs) if p >= .5))]
    finally:
        for index in inference._WORKER[1]: index.close()
    with pytest.raises(ValueError, match="feature count"):
        inference.initialize_worker(path, paths, config, .5, FEATURE_VERSION_V2)
