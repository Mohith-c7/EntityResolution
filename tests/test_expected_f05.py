import itertools
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code/business_entity_resolution"))
from src.model.expected_f05 import choose_expected_f05
from src.evaluation.metrics import entity_f05


def test_expected_utilities_equal_enumeration_including_singletons():
    probabilities=np.array([[.8,.3,.6],[0.,1.,.1],[0.,0.,0.]])
    selected,utilities=choose_expected_f05(probabilities)
    for row,p in enumerate(probabilities):
        order=np.argsort(-p,kind="stable")
        exact=[]
        for k in range(len(p)+1):
            value=0.
            for labels in itertools.product((0,1),repeat=len(p)):
                mass=np.prod([p[i] if labels[i] else 1-p[i] for i in range(len(p))])
                value+=mass*entity_f05([i for i in range(len(p)) if labels[i]],order[:k])
            exact.append(value)
        np.testing.assert_allclose(utilities[row],exact,atol=1e-12)
        assert selected[row].sum()==np.argmax(exact)
    assert not selected[2].any()


def test_invalid_probability_rejected():
    with pytest.raises(ValueError):choose_expected_f05([[1.1]])
    with pytest.raises(ValueError):choose_expected_f05([[.5]],missing_rate=float("nan"))
