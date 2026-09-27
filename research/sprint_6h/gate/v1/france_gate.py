"""Records-only auxiliary name/street evidence; the frozen matcher is untouched.

``fit_signals`` accepts a country -> Source 1 record iterable mapping. It never
reads truth fields. The resulting JSON-compatible state counts distinct Source
1 record IDs as possible owners, not target rows or supervised identity labels.
``pair_signals`` requires its reference in that exact owner universe. Empty
names are missing keys (-1), rather than a large shared-name bucket.

Affix discovery is diagnostic: only the explicit legal-form vocabulary below
can change the reduced view. No learned/high-frequency token is auto-stripped.
Street parsing is deliberately partial. A missing/uncertain parse, house-number
disagreement, or postcode disagreement cannot independently veto a link.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.preprocessing.normalize import (  # noqa: E402
    LEGAL_SUFFIXES, accent_fold, normalize_address, normalize_country, normalize_name,
)

SIGNAL_VERSION = "aux-name-street-v1"
GATE_VERSION = "conservative-street-v1"
LEGAL_EDGE_TOKENS = LEGAL_SUFFIXES
# A token becomes a template hypothesis only through supplied-record syntax:
# repeated numeric-leading clause edges with several different inner phrases.
# These are structural candidates, never a country-specific street vocabulary.
TEMPLATE_MIN_SUPPORT = 3
TEMPLATE_MIN_CONTEXTS = 3
PREFIX_MIN_FOLLOWING_WORDS = 8
DEFAULT_CONFIG = {"min_name_rivals": 2, "max_street_overlap": 0.0, "min_street_tokens": 2}
_HOUSE = re.compile(r"^(\d{1,4})([a-z]{0,3})$")
_UK_POSTCODE = re.compile(r"\b[a-z]{1,2}\d[a-z\d]?\s+\d[a-z]{2}\b", re.I)


def _field(record, *keys):
    for key in keys:
        value = record.get(key) if isinstance(record, Mapping) else getattr(record, key, None)
        if value is not None:
            return value
    return ""


def _text(value):
    if value is None:
        return ""
    try:
        if bool(value != value):
            return ""
    except (ValueError, TypeError):
        return ""
    return str(value)


def _country(record, fallback=""):
    return normalize_country(_field(record, "country")) or normalize_country(fallback)


def _name(record):
    return accent_fold(normalize_name(_field(record, "business_name", "name")))


def reduced_name(name, edge_tokens=LEGAL_EDGE_TOKENS):
    """Remove explicit edge legal tokens, preserving informative interior tokens."""
    tokens = name.split()
    left, right = 0, len(tokens)
    while left < right and tokens[left] in edge_tokens:
        left += 1
    while right > left and tokens[right - 1] in edge_tokens:
        right -= 1
    return " ".join(tokens[left:right])


def address_view(record, templates=None):
    """Preserve raw text and parse only supplied-record-supported street syntax."""
    raw = _text(_field(record, "business_address", "address"))
    folded = accent_fold(normalize_address(raw))
    tokens = folded.split()
    postcodes = sorted({t for t in tokens if t.isascii() and t.isdecimal() and 5 <= len(t) <= 6}
                       | {normalize_address(m.group()) for m in _UK_POSTCODE.finditer(raw)})
    templates = templates or {"prefix": [], "suffix": []}
    spans, houses, syntax_edges = [], [], []
    for raw_clause in re.split(r"[,;\n]", raw):
        clause = accent_fold(normalize_address(raw_clause)).split()
        if not clause:
            continue
        match = _HOUSE.fullmatch(clause[0])
        # Long numbers might be postcodes. They cannot anchor a house/street.
        if clause[0].isdecimal() and match is None:
            continue
        body = clause[1:] if match else clause
        suffix = match.group(2) if match else None
        if match and not suffix and body and len(body[0]) == 1 and body[0].isalpha():
            suffix, body = body[0], body[1:]
        words = []
        for token in body:
            if any(ch.isdigit() for ch in token):
                break
            words.append(token)
        if match and 2 <= len(words) <= 8:
            syntax_edges.extend([("prefix", words[0], " ".join(words[1:])),
                                 ("suffix", words[-1], " ".join(words[:-1]))])
        parsed = []
        if words and words[0] in templates["prefix"]:
            parsed.append(frozenset(words[1:]))
        for i, token in enumerate(words):
            if token in templates["suffix"] and i:
                parsed.append(frozenset(words[:i]))
        parsed = [span for span in parsed if span]
        if parsed:
            spans.extend(parsed)
            if match:
                houses.append((str(int(match.group(1))), suffix))
    spans = sorted(set(spans), key=lambda span: (-len(span), sorted(span)))
    reliable = bool(spans) and all(span <= spans[0] for span in spans)
    street = sorted(spans[0]) if reliable else []
    unique_houses = set(houses)
    house, suffix = next(iter(unique_houses)) if len(unique_houses) == 1 else (None, None)
    return {"raw": raw, "folded": folded, "tokens": sorted(set(tokens)),
            "street_tokens": street, "street_reliable": reliable,
            "house_number": house, "house_suffix": suffix, "postcodes": postcodes,
            "syntax_edges": syntax_edges}


def fit_signals(records_by_country):
    """Fit deterministic text statistics on the supplied Source 1 universe only.

    Extra fields (including any label field) are ignored. The membership map
    makes self exclusion verifiable and survives JSON serialization. Duplicate
    IDs fail rather than silently inflating possible-owner counts.
    """
    owners, countries = {}, {}
    for country_label, records in records_by_country.items():
        country = normalize_country(country_label)
        state = countries.setdefault(country, {"records": 0, "original_name_counts": Counter(),
            "reduced_name_counts": Counter(), "address_df": Counter(), "edge_counts": Counter(),
            "name_token_counts": Counter(), "names": set(), "empty_reduced": 0,
            "template_edges": {"prefix": Counter(), "suffix": Counter()},
            "template_contexts": {"prefix": {}, "suffix": {}}})
        for record in records:
            actual_country = _country(record, country)
            if actual_country != country:
                raise ValueError("Country group does not match its record country")
            owner = _text(_field(record, "entity_id", "source1_entity_id"))
            if not owner or owner in owners:
                raise ValueError("Source 1 owner IDs must be nonempty and unique")
            name = _name(record)
            reduced = reduced_name(name)
            owners[owner] = [country, name, reduced]
            state["records"] += 1
            if name:
                state["original_name_counts"][name] += 1
                state["names"].add(name)
                words = name.split()
                state["edge_counts"].update(set((words[0], words[-1])))
                state["name_token_counts"].update(set(words))
            if reduced:
                state["reduced_name_counts"][reduced] += 1
            else:
                state["empty_reduced"] += 1
            address = address_view(record)
            state["address_df"].update(address["tokens"])
            for orientation, token, context in set(address["syntax_edges"]):
                state["template_edges"][orientation][token] += 1
                state["template_contexts"][orientation].setdefault(token, set()).add(context)
    for state in countries.values():
        state["street_template_candidates"] = {orientation: {token: {"support": n,
            "distinct_contexts": len(state["template_contexts"][orientation][token]),
            "distinct_following_words": len({context.split()[0] for context in
                state["template_contexts"][orientation][token] if context}),
            "edge_fraction": n / state["address_df"][token]}
            for token, n in sorted(state["template_edges"][orientation].items()) if n >= TEMPLATE_MIN_SUPPORT}
            for orientation in ("prefix", "suffix")}
        state["street_templates"] = {orientation: [token for token, info in candidates.items()
            if info["distinct_contexts"] >= TEMPLATE_MIN_CONTEXTS and info["edge_fraction"] >= 0.75
            and (orientation != "prefix" or info["distinct_following_words"] >= PREFIX_MIN_FOLLOWING_WORDS)]
            for orientation, candidates in state["street_template_candidates"].items()}
        state.pop("template_edges")
        state.pop("template_contexts")
        names = state.pop("names")
        variant_support = Counter()
        variant_bases = {}
        for name in names:
            words = name.split()
            if len(words) > 1:
                if " ".join(words[1:]) in names:
                    variant_support[words[0]] += 1
                    base = " ".join(words[1:])
                    if state["original_name_counts"][base] >= 2:
                        variant_bases.setdefault(words[0], set()).add(base)
                if " ".join(words[:-1]) in names:
                    variant_support[words[-1]] += 1
                    base = " ".join(words[:-1])
                    if state["original_name_counts"][base] >= 2:
                        variant_bases.setdefault(words[-1], set()).add(base)
        state["affix_candidates"] = {token: {"edge_count": n,
            "token_count": state["name_token_counts"][token],
            "variant_support": variant_support[token], "applied": token in LEGAL_EDGE_TOKENS}
            for token, n in sorted(state["edge_counts"].items()) if n >= 2 or variant_support[token]}
        # A separate ablation view, never evidence that a token is semantically
        # a legal form and never enabled in the default gate. Require three
        # observed base relationships, repeated base records and edge dominance.
        state["learned_edge_tokens"] = sorted(token for token, bases in variant_bases.items()
            if len(bases) >= 3 and len(token) >= 2 and not token.isdecimal()
            and state["edge_counts"][token] / state["name_token_counts"][token] >= 0.9)
        learned_tokens = LEGAL_EDGE_TOKENS | frozenset(state["learned_edge_tokens"])
        learned_counts = Counter()
        for name, count in state["original_name_counts"].items():
            core = reduced_name(name, learned_tokens)
            if core:
                learned_counts[core] += count
        state["learned_core_counts"] = dict(sorted(learned_counts.items()))
        state.pop("edge_counts")
        state.pop("name_token_counts")
        for key in ("original_name_counts", "reduced_name_counts", "address_df"):
            state[key] = dict(sorted(state[key].items()))
    return {"version": SIGNAL_VERSION, "owner_unit": "source1_record_id", "countries": countries,
            "owner_keys": owners, "legal_edge_tokens": sorted(LEGAL_EDGE_TOKENS),
            "affix_discovery_applied": False, "learned_core_version": "strict-edge-hypothesis-v1",
            "learned_core_used_for_veto": False}


def _weighted_overlap(left, right, state):
    left, right = set(left), set(right)
    if not left or not right:
        return None
    n, df = state["records"], state["address_df"]
    def weight(token):
        return 1.0 + math.log((1.0 + n) / (1.0 + df.get(token, 0)))
    numerator = sum(weight(t) for t in left & right)
    # Containment tolerates dropped tokens; exact missing fields stay None.
    return numerator / min(sum(weight(t) for t in left), sum(weight(t) for t in right))


def pair_signals(s1, target, stats):
    """Compute evidence without scores/labels and exclude exactly the current S1."""
    if stats.get("version") != SIGNAL_VERSION:
        raise ValueError("Unsupported gate signal version")
    owner = _text(_field(s1, "entity_id", "source1_entity_id"))
    country = _country(s1)
    rname, tname = _name(s1), _name(target)
    rreduced, treduced = reduced_name(rname), reduced_name(tname)
    owner_key = stats["owner_keys"].get(owner)
    if owner_key is None or owner_key != [country, rname, rreduced]:
        raise ValueError("Reference must belong unchanged to the fitted Source 1 universe")
    state = stats["countries"][country]
    learned_tokens = LEGAL_EDGE_TOKENS | frozenset(state["learned_edge_tokens"])
    rlearned, tlearned = reduced_name(rname, learned_tokens), reduced_name(tname, learned_tokens)
    def rivals(kind, key, reference_key):
        if not key:
            return -1
        return state[kind].get(key, 0) - int(key == reference_key)
    raddress, taddress = address_view(s1, state["street_templates"]), address_view(target, state["street_templates"])
    overlap = _weighted_overlap(raddress["street_tokens"], taddress["street_tokens"], state)
    rn = rivals("original_name_counts", rname, rname)
    tn = rivals("original_name_counts", tname, rname)
    rr = rivals("reduced_name_counts", rreduced, rreduced)
    tr = rivals("reduced_name_counts", treduced, rreduced)
    name_match = bool(rname and rname == tname) or bool(rreduced and rreduced == treduced)
    house_known = raddress["house_number"] is not None and taddress["house_number"] is not None
    postcode_known = bool(raddress["postcodes"] and taddress["postcodes"])
    return {"version": SIGNAL_VERSION, "country": country,
        "reference_name_raw": _text(_field(s1, "business_name", "name")),
        "target_name_raw": _text(_field(target, "business_name", "name")),
        "reference_name": rname, "target_name": tname,
        "reference_reduced_name": rreduced, "target_reduced_name": treduced,
        "reference_learned_core": rlearned, "target_learned_core": tlearned,
        "reference_learned_core_rivals": rivals("learned_core_counts", rlearned, rlearned),
        "target_learned_core_rivals": rivals("learned_core_counts", tlearned, rlearned),
        "reference_original_name_rivals": rn, "target_original_name_rivals": tn,
        "reference_reduced_name_rivals": rr, "target_reduced_name_rivals": tr,
        "name_rivals": max(rn, tn, rr, tr), "name_match": name_match,
        "reference_name_missing": not rname, "target_name_missing": not tname,
        "reference_reduced_empty": not rreduced, "target_reduced_empty": not treduced,
        "reference_address_missing": not raddress["folded"], "target_address_missing": not taddress["folded"],
        "reference_address": raddress, "target_address": taddress,
        "address_overlap": _weighted_overlap(raddress["tokens"], taddress["tokens"], state),
        "street_overlap": overlap,
        "street_reliable": raddress["street_reliable"] and taddress["street_reliable"],
        "reference_street_token_count": len(raddress["street_tokens"]),
        "target_street_token_count": len(taddress["street_tokens"]),
        "strong_street_contradiction": bool(raddress["street_reliable"] and taddress["street_reliable"]
            and len(raddress["street_tokens"]) >= 2 and len(taddress["street_tokens"]) >= 2 and overlap == 0.0),
        "house_number_match": raddress["house_number"] == taddress["house_number"] if house_known else None,
        "house_number_conflict": house_known and raddress["house_number"] != taddress["house_number"],
        "house_suffix_match": raddress["house_suffix"] == taddress["house_suffix"] if house_known else None,
        "postcode_match": bool(set(raddress["postcodes"]) & set(taddress["postcodes"])) if postcode_known else None,
        "country_conflict": bool(_country(target) and country != _country(target)),
        # An overlap token remains visible: downweighting is an auxiliary name
        # hypothesis, never deletion from the address/street or original name.
        "reference_name_token_weights": {t: (0.5 if t in raddress["tokens"] else 1.0) for t in rreduced.split()},
        "target_name_token_weights": {t: (0.5 if t in taddress["tokens"] else 1.0) for t in treduced.split()}}


def gate_veto(signals, config=None):
    """Pure conservative veto; returns separate decision fields, never a score."""
    config = DEFAULT_CONFIG if config is None else {**DEFAULT_CONFIG, **config}
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError("Unknown gate configuration key")
    if (not isinstance(config["min_name_rivals"], int) or config["min_name_rivals"] < 1
        or not isinstance(config["min_street_tokens"], int) or config["min_street_tokens"] < 1
        or not 0 <= config["max_street_overlap"] < 1):
        raise ValueError("Invalid conservative gate configuration")
    if signals.get("version") != SIGNAL_VERSION:
        raise ValueError("Unsupported gate signal version")
    reason = "street_contradiction"
    if not signals["name_match"]:
        reason = "name_not_equal"
    elif signals["name_rivals"] < config["min_name_rivals"]:
        reason = "name_not_ambiguous"
    elif signals["reference_address_missing"] or signals["target_address_missing"]:
        reason = "missing_address"
    elif signals["country_conflict"]:
        reason = "country_conflict_requires_existing_policy"
    elif not signals["street_reliable"] or min(signals["reference_street_token_count"],
        signals["target_street_token_count"]) < config["min_street_tokens"]:
        reason = "uncertain_street"
    elif signals["street_overlap"] is None or signals["street_overlap"] > config["max_street_overlap"]:
        reason = "street_support"
    veto = reason == "street_contradiction"
    return {"gate_version": GATE_VERSION, "gate_veto": veto, "gate_reason": reason,
            "veto": veto, "reason": reason}
