"""Label-free Source 1 universe evidence: could another reference own this target?

Counts come only from the record text of every Source 1 row in one split. Keys
are normalized text, so count tables are identical across processes and runs.
The current reference is always removed from its own counts. A missing key is
-1; a present key that no other reference shares is 0.
"""
from collections import Counter

from ..preprocessing.normalize import accent_fold
from .evidence import phonetic

KEY_KINDS = ("core", "core_set", "phonetic_set", "core_number", "address_set")
COMPETITION_NAMES = (
    "cmp_target_core_others", "cmp_target_core_set_others", "cmp_target_phonetic_others",
    "cmp_target_core_number_others", "cmp_target_address_others",
    "cmp_reference_core_others", "cmp_reference_address_others",
)


def record_keys(name, core, address):
    """Keys from the normalized name, name core and address stored in the indexes."""
    core = accent_fold(core or name)
    folded_address = accent_fold(address)
    core_set = " ".join(sorted(set(core.split())))
    phonetic_set = " ".join(sorted(set(phonetic(core).split()))) if core else ""
    address_tokens = [t.lstrip("0") or "0" if t.isdecimal() else t for t in folded_address.split()]
    numbers = [t for t in address_tokens if t.isdecimal()]
    address_set = " ".join(sorted(set(address_tokens)))
    return {
        "core": core or None,
        "core_set": core_set or None,
        "phonetic_set": phonetic_set or None,
        "core_number": f"{core_set}|{numbers[0]}" if core_set and numbers else None,
        "address_set": address_set or None,
    }


def count_universe(keys_iterable, wanted):
    """Count only keys of interest; ``wanted`` maps each kind to a set of keys."""
    counts = {kind: Counter() for kind in KEY_KINDS}
    for keys in keys_iterable:
        for kind in KEY_KINDS:
            key = keys[kind]
            if key is not None and key in wanted[kind]:
                counts[kind][key] += 1
    return counts


def _others(counts, kind, key, reference_key):
    if key is None:
        return -1
    total = counts[kind][key]
    own = int(key == reference_key)
    if total < own:
        raise ValueError("The reference must belong to the counted Source 1 universe")
    return total - own


def competition_features(reference_keys, target_keys, counts):
    r, t = reference_keys, target_keys
    return [
        _others(counts, "core", t["core"], r["core"]),
        _others(counts, "core_set", t["core_set"], r["core_set"]),
        _others(counts, "phonetic_set", t["phonetic_set"], r["phonetic_set"]),
        _others(counts, "core_number", t["core_number"], r["core_number"]),
        _others(counts, "address_set", t["address_set"], r["address_set"]),
        _others(counts, "core", r["core"], r["core"]),
        _others(counts, "address_set", r["address_set"], r["address_set"]),
    ]
