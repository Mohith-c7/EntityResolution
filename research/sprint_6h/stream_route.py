"""Collect exact global route proposals without holding all base pairs in RAM."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import math
import train_sibling_matcher as sibling


def collect(frame, global_ids, config=sibling.RouteConfig()):
    owners=list(frame.source1_entity_id.unique())
    # select_route uses its universe only for membership and the final cap.
    # Genuine auxiliary IDs make this local cap >= every local proposal; the
    # only semantic cap is subsequently applied to the complete global pool.
    if len(global_ids)<5*len(owners):
        raise ValueError('Chunk too large for untruncated proposal collection')
    local_universe=[*owners,*global_ids[:5*len(owners)]]
    route=sibling.select_route(frame,config=config,universe_ids=local_universe)
    edges=sibling.route_edges(frame,route)
    priorities={}
    for position in route:
        owner=frame.source1_entity_id.iat[position]
        probability=float(frame.probability.iat[position])
        priorities[owner]=max(priorities.get(owner,0.),min(probability,1-probability))
    return priorities,edges


def select(priorities,global_ids,config=sibling.RouteConfig()):
    if not set(priorities)<=set(global_ids):raise ValueError('Foreign proposal owner')
    return [e for e in sorted(priorities,key=lambda e:(-priorities[e],e))[:math.floor(config.fraction*len(global_ids))]]
