"""Single conservative gate-v3 correction over immutable v2 record signals.

Qualified matching house numbers force abstention, never acceptance. Missing or
different numbers do not independently reject a pair. All name/street signals,
fitted text statistics and configured thresholds remain v2.
"""
from france_gate import SIGNAL_VERSION, fit_signals, pair_signals
from france_gate import gate_veto as gate_veto_v2

GATE_VERSION = "conservative-street-v3-house-agreement"


def gate_veto(signals, config=None):
    result = gate_veto_v2(signals, config)
    result = {**result, "gate_version": GATE_VERSION}
    if result["gate_veto"] and signals.get("house_number_match") is True:
        result.update(gate_veto=False, veto=False, gate_reason="qualified_house_agreement", reason="qualified_house_agreement")
    return result
