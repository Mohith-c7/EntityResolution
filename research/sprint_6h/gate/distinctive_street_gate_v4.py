"""One predeclared generic DF/typo street ablation; labels never enter it.

All original v2 tokens/statistics survive. This separate auxiliary view requires
distinctive parsed street tokens on both sides and absence of exact/near support.
The existing v3 house-agreement abstention is mandatory, never acceptance.
"""
from france_gate import SIGNAL_VERSION, _name, fit_signals, pair_signals as pair_signals_v2, reduced_name
from house_agreement_gate_v3 import gate_veto as gate_veto_v3

GATE_VERSION = "conservative-street-v4-distinctive"
VIEW_VERSION = "distinctive-street-df-v1"
DF_FRACTION_MAX = 0.01
MIN_LENGTH = 3
NEAR_MATCH_MAX = 0.25
NEAR_MATCH_MIN_LENGTH = 4
DEFAULT_CONFIG = {"min_name_rivals": 4, "max_street_overlap": 0.0, "min_street_tokens": 2}


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def near_match(left, right):
    if left == right:
        return True
    if min(len(left), len(right)) < NEAR_MATCH_MIN_LENGTH:
        return False
    return edit_distance(left, right) / max(len(left), len(right)) <= NEAR_MATCH_MAX


def pair_signals(s1, target, stats):
    result = pair_signals_v2(s1, target, stats)
    state = stats["countries"][result["country"]]
    def distinctive(address):
        return [token for token in address["street_tokens"]
                if len(token) >= MIN_LENGTH and not token.isdecimal()
                and state["address_df"].get(token, 0) / state["records"] <= DF_FRACTION_MAX]
    left, right = distinctive(result["reference_address"]), distinctive(result["target_address"])
    support = [[a, b] for a in left for b in right if near_match(a, b)]
    return {**result, "distinctive_view_version": VIEW_VERSION,
            "reference_distinctive_street_tokens": left, "target_distinctive_street_tokens": right,
            "distinctive_street_support": support,
            "distinctive_street_contradiction": bool(left and right and not support)}


def gate_veto(signals, config=None):
    config = DEFAULT_CONFIG if config is None else {**DEFAULT_CONFIG, **config}
    result = {**gate_veto_v3(signals, config), "gate_version": GATE_VERSION}
    if signals.get("distinctive_view_version") != VIEW_VERSION:
        raise ValueError("V4 gate requires the separately versioned distinctive-token view")
    if (not signals["name_match"] or signals["name_rivals"] < config["min_name_rivals"]
            or signals["reference_address_missing"] or signals["target_address_missing"]
            or signals["country_conflict"]):
        return result
    if signals.get("house_number_match") is True:
        reason = "qualified_house_agreement"
    elif not signals["street_reliable"]:
        reason = "uncertain_street"
    elif not signals["reference_distinctive_street_tokens"] or not signals["target_distinctive_street_tokens"]:
        reason = "distinctive_street_missing"
    elif signals["distinctive_street_support"]:
        reason = "distinctive_street_support"
    else:
        reason = "distinctive_street_contradiction"
    veto = reason == "distinctive_street_contradiction"
    return {**result, "gate_veto": veto, "veto": veto, "gate_reason": reason, "reason": reason}
