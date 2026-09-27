"""Seal calibrated hybrid decisions and final accepted-claim exclusivity.

No labels are read. Require exact mask parity on the complete development
graph before adopting the existing deterministic exclusivity helper.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'research/final_2h')]
from evaluate_sprint import decide_control
from export_sprint_workbench import sha
from post_selection_exclusivity import exclusive_selected


def main():
    out = Path('research/final_2h/hybrid_decision_release_v1')
    out.mkdir(parents=True, exist_ok=False)
    probe = Path('research/final_2h/hybrid_decision_probe_v1/report.json')
    report = json.loads(probe.read_text())
    if report['changed_thresholds_accepted'] is not True:
        raise ValueError('Changed thresholds did not pass prospective probe')
    config = report['effective_config']
    data = Path('research/final_2h/fixed_hybrid_v1/pairs.parquet')
    protocol = {'status': 'declared_before_label_free_exclusivity_replay',
        'declared_at': datetime.now(timezone.utc).isoformat(), 'decision_config': config,
        'post_selection_exclusivity': True,
        'policy': 'Apply unchanged decide, then retain highest-probability selected claim per target; source1 ID breaks exact ties. Unselected claims cannot compete.',
        'reason': 'New ordinary acceptance .79 is below inherited pre-arbitration .83; every finally proposed link must participate in final ownership resolution.',
        'development_requirement': 'Exact mask parity on all30000 owners; otherwise abort without release eligibility.',
        'score_probe_sha256': sha(probe), 'pairs_sha256': sha(data),
        'helper_sha256': sha(ROOT/'research/final_2h/post_selection_exclusivity.py'),
        'source_sha256': sha(__file__), 'labels_read': False,
        'extension_audit_labels_read': False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    frame = pd.read_parquet(data)
    if len(frame)!=1200000 or frame.source1_entity_id.nunique()!=30000:
        raise ValueError('Incomplete development graph')
    keep, _ = decide_control(frame, frame, config)
    after, removed = exclusive_selected(frame, keep)
    same = bool(np.array_equal(after, keep))
    evidence = {**protocol, 'status': 'label_free_complete_graph_replay_complete',
        'references': 30000, 'pairs': len(frame), 'selected': int(keep.sum()),
        'removed_by_exclusivity': int(removed.sum()), 'exact_mask_parity': same,
        'eligible_for_fresh_audit': same,
        'development_macro_f05': report['selection']['candidate']['macro_f05'] if same else None}
    (out/'report.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({k: evidence[k] for k in ['eligible_for_fresh_audit','removed_by_exclusivity','development_macro_f05','decision_config']}),flush=True)
    if not same: raise ValueError('Exclusivity changed development predictions; no automatic promotion')


if __name__=='__main__': main()
