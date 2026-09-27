"""Streaming may partition execution, but must not change global selection."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stream_route',ROOT/'research/sprint_6h/stream_route.py')
stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)


def test_dense_chunks_ties_and_empty_owners_match_complete_route():
    ids=[f'S1-{i:05}' for i in range(602)]
    rng=np.random.default_rng(17)
    rows=[]
    for i,owner in enumerate(ids[:600]):
        # High-uncertainty owners are concentrated in early chunks, making a
        # naive local20% truncation lose globally eligible references.
        for j in range(6):
            rows.append([owner,f'S{2+j%2}-{i:05}-{j}',.99 if j<2 else .5,
                .50 if i<100 and j==2 else float(rng.uniform(.1,.99))])
    frame=pd.DataFrame(rows,columns=['source1_entity_id','candidate_entity_id','first_stage','probability'])
    expected=stream.sibling.route_edges(frame,stream.sibling.select_route(frame,universe_ids=ids))
    priorities={};parts=[]
    for start in range(0,len(frame),300):
        current,edges=stream.collect(frame.iloc[start:start+300].reset_index(drop=True),ids)
        assert not set(current)&set(priorities)
        priorities.update(current);parts.append(edges)
    chosen=set(stream.select(priorities,ids));actual=pd.concat(parts,ignore_index=True)
    actual=actual[actual.source1_entity_id.isin(chosen)]
    assert set(map(tuple,actual.to_numpy()))==set(map(tuple,expected.to_numpy()))
    assert len(chosen)==120


def test_refuses_chunks_that_would_truncate_local_proposals():
    frame=pd.DataFrame({'source1_entity_id':['S1-a','S1-b']})
    with pytest.raises(ValueError,match='Chunk too large'):stream.collect(frame,['S1-a','S1-b'])
