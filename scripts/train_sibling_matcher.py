"""Bounded, owner-excluded target-to-target evidence for the frozen matcher.

No retrieval, first-stage fitting, test pseudo-labels, or array-by-length joins.
Commands are intentionally separate: audit the current route before preparing
training examples; fit compatibility models; export pair-keyed support features.
"""
import argparse
import hashlib
import itertools
import json
import math
import sqlite3
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.model.crossfit_aliases import inner_fold
from src.preprocessing.normalize import accent_fold
from src.preprocessing.normalize import extract_digits
from src.features.evidence import phonetic, grams, TLD, numeric_relation

KEYS = ["source1_entity_id", "candidate_entity_id"]
FEATURE_NAMES = ["name_ratio", "name_sort", "name_set", "name_jaccard", "name_length_ratio",
    "address_ratio", "address_sort", "address_set", "address_jaccard", "address_length_ratio",
    "independent_name_sort", "independent_name_set", "numeric_jaccard", "numeric_disjoint",
    "numeric_both_present", "name_both_present", "address_both_present", "name_one_missing",
    "address_one_missing", "name_script_jaccard", "address_script_jaccard", "same_source",
    "name_unmatched_fraction", "address_unmatched_fraction", "cross_field_max",
    "numeric_containment", "numeric_single_substitution", "numeric_relation_disjoint",
    "numeric_unmatched_min", "numeric_unmatched_max", "phonetic_name_set", "phonetic_name_sort",
    "phonetic_name_ratio", "phonetic_char_jaccard", "collapsed_phonetic_ratio", "collapsed_phonetic_partial"]
FEATURE_VERSION = "sibling-tt-v2-36"
LEARNED_WEIGHTS = (.50, .75, 1.)
SUPPORT_NAMES = ["sibling_count", "sibling_max", "sibling_mean", "sibling_min",
    "sibling_weighted_max", "sibling_contradiction", "sibling_opposite_max"]
RESIDUAL_NAMES = ["probability", "first_stage", *SUPPORT_NAMES]
PINNED_ROUTE_CACHE = {}


@dataclass(frozen=True)
class RouteConfig:
    fraction: float = .20
    candidates: int = 4
    seeds: int = 2
    seed_min: float = .90
    candidate_min: float = .03
    candidate_max: float = .995
    threshold: float = .83
    seed_column: str = "first_stage"

    def validate(self):
        if not 0 <= self.fraction <= .20 or not 1 <= self.candidates <= 4 or not 1 <= self.seeds <= 2:
            raise ValueError("Route exceeds sprint's 20% / four candidates / two seeds budget")
        if not 0 <= self.candidate_min < self.candidate_max <= 1 or not 0 <= self.seed_min <= 1:
            raise ValueError("Invalid route probability bounds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def pair_order_hash(frame):
    h = hashlib.sha256()
    for s, c in frame[KEYS].itertuples(index=False, name=None):
        h.update((str(s) + "\t" + str(c) + "\n").encode())
    return h.hexdigest()


def load_pairs(path, manifest_path, training=False):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("status") != "complete" or not manifest.get("verified_current_pipeline"):
        raise ValueError("Need a complete, verified current-pipeline pair table")
    if manifest.get("split") not in (("train",) if training else ("development", "confirmation", "audit", "cohort", "test")):
        raise ValueError("Wrong split; audit labels and test pseudo-labels are forbidden")
    if digest(path) != manifest.get("pairs_sha256"):
        raise ValueError("Pair table hash mismatch")
    frame = pd.read_parquet(path).reset_index(drop=True)
    required = [*KEYS, "probability", "first_stage"]
    if not set(required) <= set(frame): raise ValueError(f"Missing pair columns: {required}")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate pair keys")
    if pair_order_hash(frame) != manifest.get("pair_order_sha256"):
        raise ValueError("Pair order hash mismatch")
    for name in ("probability", "first_stage"):
        if not np.isfinite(frame[name]).all() or not frame[name].between(0, 1).all():
            raise ValueError(f"Invalid {name}")
    if training:
        if not manifest.get("first_stage_owner_excluded_oof") or "oof_excluded_fold" not in frame:
            raise ValueError("Training seeds require owner-excluded OOF scores")
        folds = frame.source1_entity_id.map(inner_fold)
        if not folds.equals(frame.oof_excluded_fold.astype(folds.dtype)):
            raise ValueError("A training row was scored by a model containing its owner's fold")
    return frame, manifest


def select_route(frame, config=RouteConfig(), universe_ids=None):
    """Labels are never accessed. Returns original row positions and peer positions.

    Supply all reference IDs, including references with no candidates, for the cap.
    Within each reference the two highest OOF first-stage seeds are kept even
    when they are wrong; neither seed correctness nor target owners are read.
    """
    config.validate()
    if not frame.index.equals(pd.RangeIndex(len(frame))): raise ValueError("Reset pair-table index before routing")
    if config.seed_column not in frame: raise ValueError("Missing declared seed probability")
    ids = set(universe_ids) if universe_ids is not None else set(frame.source1_entity_id)
    if not set(frame.source1_entity_id) <= ids: raise ValueError("Pair reference is outside declared universe")
    probabilities = frame.probability.to_numpy()
    seed_probabilities = frame[config.seed_column].to_numpy()
    candidate_ids = frame.candidate_entity_id.to_numpy()
    proposals = []
    for owner, positions in frame.groupby("source1_entity_id", sort=False).indices.items():
        positions = list(positions)
        seeds = sorted((i for i in positions if seed_probabilities[i] >= config.seed_min),
            key=lambda i: (-seed_probabilities[i], candidate_ids[i]))[:config.seeds]
        difficult = [i for i in positions if config.candidate_min <= probabilities[i] <= config.candidate_max
            and any(candidate_ids[i] != candidate_ids[j] for j in seeds)]
        difficult.sort(key=lambda i: (abs(probabilities[i] - config.threshold), candidate_ids[i]))
        difficult = difficult[:config.candidates]
        if not difficult: continue
        # Uncertainty is monotonic around 0.5, and supplements proximity to the
        # incumbent decision boundary. It does not require labels or features.
        priority = max(min(probabilities[i], 1 - probabilities[i]) for i in difficult)
        proposals.append((-priority, str(owner), difficult, seeds))
    proposals.sort(key=lambda x: (x[0], x[1]))
    chosen = proposals[:math.floor(config.fraction * len(ids))]
    route = {}
    for _, _, difficult, seeds in chosen:
        for i in difficult:
            route[i] = [j for j in seeds if candidate_ids[j] != candidate_ids[i]]
    return route


def route_edges(frame, route):
    return pd.DataFrame([{ "source1_entity_id": frame.source1_entity_id.iat[i],
        "candidate_entity_id": frame.candidate_entity_id.iat[i],
        "peer_entity_id": frame.candidate_entity_id.iat[j] } for i, peers in route.items() for j in peers],
        columns=[*KEYS, "peer_entity_id"])


def load_pinned_route(frame, source_manifest, directory):
    """Use one globally selected route in subsets/chunks without reranking.

    The whole plan is verified once per process and cached for a streaming runner.
    Chunk manifests must carry routing_universe_pair_order_sha256 from the
    complete source; a full-source manifest can use its existing pair-order hash.
    """
    directory = Path(directory)
    marker = json.loads((directory / "manifest.json").read_text())
    if marker.get("status") != "complete" or marker.get("route") != asdict(RouteConfig()):
        raise ValueError("Pinned route is incomplete or has a different policy")
    universe_hash = source_manifest.get("routing_universe_pair_order_sha256", source_manifest.get("pair_order_sha256"))
    if universe_hash != marker.get("global_pair_order_sha256"):
        raise ValueError("Chunk/subset is not from the pinned route's global candidate universe")
    if marker.get("frozen_sha256") and source_manifest.get("frozen_sha256") != marker["frozen_sha256"]:
        raise ValueError("Pinned route model/rule lineage changed")
    path = directory / "route.parquet"
    cache_key = (str(path.resolve()), marker["route_sha256"])
    if cache_key not in PINNED_ROUTE_CACHE:
        if digest(path) != marker["route_sha256"]: raise ValueError("Pinned route edge hash mismatch")
        edges = pd.read_parquet(path)
        if edges.isna().any().any() or edges.duplicated([*KEYS, "peer_entity_id"]).any():
            raise ValueError("Pinned route contains missing or duplicate edges")
        if (edges.candidate_entity_id == edges.peer_entity_id).any(): raise ValueError("Pinned self support")
        if edges.source1_entity_id.nunique() > math.floor(.20*marker["global_references"]):
            raise ValueError("Pinned route exceeds global reference cap")
        if (edges.groupby("source1_entity_id").candidate_entity_id.nunique() > 4).any() or (
                edges.groupby("source1_entity_id").peer_entity_id.nunique() > 2).any():
            raise ValueError("Pinned route exceeds candidate/seed cap")
        grouped = {}
        for owner, candidate, peer in edges.itertuples(index=False, name=None):
            grouped.setdefault(owner, []).append((candidate, peer))
        needed = {owner: {target for pair in pairs for target in pair} for owner, pairs in grouped.items()}
        PINNED_ROUTE_CACHE[cache_key] = grouped, needed
    grouped, needed = PINNED_ROUTE_CACHE[cache_key]
    positions = {}
    for i, (owner, candidate) in enumerate(frame[KEYS].itertuples(index=False, name=None)):
        if owner in needed and candidate in needed[owner]: positions[owner, candidate] = i
    route = {}
    for owner in frame.source1_entity_id.unique():
        for candidate, peer in grouped.get(owner, ()):
            if (owner, candidate) not in positions or (owner, peer) not in positions:
                raise ValueError("Pinned candidate/peer missing: chunks must contain complete reference candidates")
            route.setdefault(positions[owner, candidate], []).append(positions[owner, peer])
    return route, marker


def entity_values(frame, accepted, truth):
    predicted = {e: set() for e in truth}
    for s, c, keep in zip(frame.source1_entity_id, frame.candidate_entity_id, accepted):
        if s not in truth: raise ValueError("Ground truth does not cover all pair references")
        if keep: predicted[s].add(c)
    values = {}
    for e, actual in truth.items():
        actual, selected = set(actual), predicted[e]
        tp = len(actual & selected)
        values[e] = (1. if not actual and not selected else
            1.25 * tp / (1.25 * tp + len(selected - actual) + .25 * len(actual - selected)) if tp else 0.)
    return values


def opportunity_audit(frame, truth, route, countries=None, evaluation_ids=None):
    if "accepted" not in frame or frame.accepted.isna().any() or not frame.accepted.isin([0, 1, False, True]).all():
        raise ValueError("Audit requires incumbent accepted state, including ownership/rescues")
    baseline = frame.accepted.to_numpy(dtype=bool)
    perfect = baseline.copy()
    for i in route: perfect[i] = frame.candidate_entity_id.iat[i] in set(truth[frame.source1_entity_id.iat[i]])
    base, oracle = entity_values(frame, baseline, truth), entity_values(frame, perfect, truth)
    evaluation_ids = list(truth) if evaluation_ids is None else list(evaluation_ids)
    evaluation_set = set(evaluation_ids)
    if not evaluation_set <= set(truth): raise ValueError("Evaluation owners are outside claimant truth")
    gain = float(np.mean([oracle[e] - base[e] for e in evaluation_ids]))
    scored_route = [i for i in route if frame.source1_entity_id.iat[i] in evaluation_set]
    result = {"baseline_macro_f05": float(np.mean([base[e] for e in evaluation_ids])),
        "fixed_other_claims_route_oracle_macro_f05": float(np.mean([oracle[e] for e in evaluation_ids])),
        "route_perfect_correction_gain": gain, "meets_opportunity_floor": gain >= .0025,
        "references": len(evaluation_ids), "claimant_references": len(truth),
        "routed_references": len(set(frame.source1_entity_id.iat[i] for i in route)),
        "routed_evaluation_references": len(set(frame.source1_entity_id.iat[i] for i in scored_route)),
        "routed_candidates": len(route), "comparisons": sum(map(len, route.values())),
        "false_links_correctable": int(sum(baseline[i] and not perfect[i] for i in scored_route)),
        "true_links_recoverable": int(sum(perfect[i] and not baseline[i] for i in scored_route)),
        "caveat": "Perfect row decisions with all other claims fixed; opportunity ceiling, not a learned gain or ownership validation."}
    if countries:
        result["country_gain"] = {country: float(np.mean([oracle[e] - base[e] for e in evaluation_ids if countries[e] == country]))
            for country in sorted({countries[e] for e in evaluation_ids})}
    return result


def feasible_score_diagnostic(frame, truth, route, config, decide, evaluation_ids=None):
    """Ideal residual endpoints under the three predeclared blends.

    Replays the original policy within the supplied cohort only. This is a
    diagnostic, not fitted performance, and not an external-claim oracle.
    """
    baseline, _ = decide(frame, frame, config)
    base_values = entity_values(frame, baseline, truth)
    evaluation_ids = list(truth) if evaluation_ids is None else list(evaluation_ids)
    base_macro = float(np.mean([base_values[e] for e in evaluation_ids]))
    trials = []
    indices = np.asarray(list(route), dtype=np.int64)
    labels = np.asarray([int(frame.candidate_entity_id.iat[i] in set(truth[frame.source1_entity_id.iat[i]])) for i in indices])
    for weight in (.25, .50, .75):
        changed = frame.copy()
        probabilities = frame.probability.to_numpy().copy()
        probabilities[indices] = (1-weight)*probabilities[indices]+weight*labels
        changed["probability"] = probabilities
        chosen, _ = decide(changed, changed, config)
        values = entity_values(changed, chosen, truth)
        macro = float(np.mean([values[e] for e in evaluation_ids]))
        trials.append({"weight": weight, "macro_f05": macro, "gain_vs_same_cohort_policy": macro-base_macro,
            "ordinary_positive_recovery_probability_floor": max(0., (config["t_rest"]-weight)/(1-weight)),
            "rank_one_positive_recovery_probability_floor": max(0., (config["t_first"]-weight)/(1-weight))})
    return {"baseline_same_cohort_policy_macro_f05": base_macro, "trials": trials,
        "scope": "Label-perfect residual endpoints with ordinary ownership/rank-one replay inside supplied cohort; outside claimants omitted. Not fitted performance or a full external-claim oracle."}


def scripts(text):
    result = set()
    for c in text:
        if c.isalpha(): result.add(unicodedata.name(c, "UNKNOWN").split()[0])
    return result


def clean_target(name, address, source):
    name, address = accent_fold(str(name or "")), accent_fold(str(address or ""))
    nt, at = set(name.split()), set(address.split())
    phonetic_name = phonetic(name)
    collapsed = "".join(phonetic(t) for t in name.split() if t not in TLD)
    return {"name": name, "address": address, "nt": nt, "at": at,
        "independent": " ".join(t for t in name.split() if t not in at),
        "digits": {t.lstrip("0") or "0" for t in at if t.isdecimal()},
        "name_scripts": scripts(name), "address_scripts": scripts(address), "source": source,
        "canonical_digits": {t.lstrip("0") or "0" for t in extract_digits(" ".join(t for t in address.split() if t.isdecimal()))},
        "phonetic": phonetic_name, "phonetic_grams": grams(phonetic_name), "collapsed_phonetic": collapsed}


def jaccard(left, right):
    return len(left & right) / len(left | right) if left or right else 0.


def length_ratio(left, right):
    return min(len(left), len(right)) / max(len(left), len(right), 1)


def sibling_features(left, right):
    row = []
    for field, token in (("name", "nt"), ("address", "at")):
        a, b = left[field], right[field]
        row.extend([f(a, b) / 100 if a and b else 0. for f in (fuzz.ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio)])
        row.extend([jaccard(left[token], right[token]), length_ratio(a, b)])
    a, b = left["independent"], right["independent"]
    row.extend([f(a, b) / 100 if a and b else 0. for f in (fuzz.token_sort_ratio, fuzz.token_set_ratio)])
    da, db = left["digits"], right["digits"]
    row.extend([jaccard(da, db), bool(da and db and not da & db), bool(da and db),
        bool(left["name"] and right["name"]), bool(left["address"] and right["address"]),
        bool(left["name"]) != bool(right["name"]), bool(left["address"]) != bool(right["address"]),
        jaccard(left["name_scripts"], right["name_scripts"]),
        jaccard(left["address_scripts"], right["address_scripts"]), left["source"] == right["source"],
        1 - jaccard(left["nt"], right["nt"]), 1 - jaccard(left["at"], right["at"]),
        max(fuzz.token_set_ratio(left["name"], right["address"]) if left["name"] and right["address"] else 0.,
            fuzz.token_set_ratio(right["name"], left["address"]) if right["name"] and left["address"] else 0.) / 100])
    containment, substitution, disjoint, only_left, only_right = numeric_relation(left["canonical_digits"], right["canonical_digits"])
    row.extend([containment, substitution, disjoint, min(only_left, only_right), max(only_left, only_right)])
    pa, pb = left["phonetic"], right["phonetic"]
    row.extend([f(pa, pb)/100 if pa and pb else 0. for f in (fuzz.token_set_ratio, fuzz.token_sort_ratio, fuzz.ratio)])
    row.append(jaccard(left["phonetic_grams"], right["phonetic_grams"]))
    ca, cb = left["collapsed_phonetic"], right["collapsed_phonetic"]
    row.extend([fuzz.ratio(ca, cb)/100 if ca and cb else 0.,
        fuzz.partial_ratio(ca, cb)/100 if ca and cb and min(len(ca),len(cb)) >= 4 else 0.])
    return np.asarray(row, dtype=np.float32)


def load_targets(ids, index_prefix):
    result = {}
    for source in ("S2", "S3"):
        wanted = sorted(c for c in ids if c.startswith(source))
        with sqlite3.connect(Path(f"{index_prefix}_{source}.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as conn:
            for start in range(0, len(wanted), 500):
                part = wanted[start:start+500]
                for cid, name, address in conn.execute(f"SELECT id,name,address FROM records WHERE id IN ({','.join('?' for _ in part)})", part):
                    result[cid] = clean_target(name, address, source)
    if set(ids) - set(result): raise ValueError(f"Missing {len(set(ids)-set(result))} target records")
    return result


def owner_map(truth):
    owners = {}
    for owner, targets in truth.items():
        for target in targets:
            if target in owners and owners[target] != owner: raise ValueError("Target has multiple supervised owners")
            owners[target] = owner
    return owners


def lookup_training_owners(ids, path):
    """Read only queried target owners; other outer splits never yield labels."""
    result = {}
    with sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True) as conn:
        wanted = sorted(set(ids))
        for start in range(0, len(wanted), 500):
            part = wanted[start:start+500]
            for target, owner in conn.execute(
                f"SELECT id,owner FROM owners WHERE outer_fold='train' AND id IN ({','.join('?' for _ in part)})", part):
                if not owner: raise ValueError("Known training target lacks owner")
                result[target] = owner
    return result


def training_examples(frame, route, truth, records, positive_cap=8, external_training_owners=None):
    """Owner partitions are fixed before selecting natural hard negatives.

    Every example retains BOTH target owners, allowing fit_fold to exclude any
    example touching a held-out owner. Unlabeled targets never become negatives.
    """
    owners = owner_map(truth)
    external_training_owners = external_training_owners or {}
    for target, owner in external_training_owners.items():
        if target in owners and owners[target] != owner: raise ValueError("Owner DB conflicts with training truth")
        owners[target] = owner
    pairs = {}
    training_owners = set(frame.source1_entity_id)
    if not training_owners <= set(truth): raise ValueError("Training references lack owner truth")
    for owner in sorted(training_owners):
        # Hash bounded shared-owner positives include same- and cross-source.
        ids = sorted(truth[owner])
        positives = sorted(itertools.combinations(ids, 2),
            key=lambda p: hashlib.sha256((p[0] + "\t" + p[1]).encode()).digest())[:positive_cap]
        for a, b in positives: pairs[(a, b)] = "shared_owner"
    for i, peers in route.items():
        a = frame.candidate_entity_id.iat[i]
        for j in peers:
            b = frame.candidate_entity_id.iat[j]
            if a in owners and b in owners and (owners[a] in training_owners or a in external_training_owners) and (
                    owners[b] in training_owners or b in external_training_owners):
                pairs.setdefault(tuple(sorted((a, b))), "oof_neighborhood")
    rows = []
    for (a, b), kind in sorted(pairs.items()):
        if a == b: raise ValueError("Self supporting pair")
        if a not in records or b not in records: raise ValueError("Training target record missing")
        oa, ob = owners[a], owners[b]
        rows.append({"left_id": a, "right_id": b, "left_owner": oa, "right_owner": ob,
            "left_fold": inner_fold(oa), "right_fold": inner_fold(ob),
            "label": int(oa == ob), "kind": kind, **dict(zip(FEATURE_NAMES, sibling_features(records[a], records[b])))})
    return pd.DataFrame(rows)


def fold_masks(examples, fold):
    fit = (examples.left_fold != fold) & (examples.right_fold != fold)
    validation = (examples.left_fold == fold) & (examples.right_fold == fold)
    fit_owners = set(examples.loc[fit, "left_owner"]) | set(examples.loc[fit, "right_owner"])
    held_owners = set(examples.loc[validation, "left_owner"]) | set(examples.loc[validation, "right_owner"])
    if fit_owners & held_owners: raise ValueError("Owner leakage in sibling split")
    return fit, validation


def support_features(frame, route, records, model, threads=1):
    """Batch predictions; output aligns by verified pair keys, never by count."""
    links, features = [], []
    for i, peers in route.items():
        a = frame.candidate_entity_id.iat[i]
        for j in peers:
            b = frame.candidate_entity_id.iat[j]
            if a == b: raise ValueError("Candidate cannot support itself")
            links.append((i, j)); features.append(sibling_features(records[a], records[b]))
    matrix = np.asarray(features, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    values = model.predict(matrix, num_threads=threads) if len(matrix) else np.zeros(0)
    grouped = {}
    for (i, j), p in zip(links, values): grouped.setdefault(i, []).append((j, float(p)))
    result = frame[KEYS].copy()
    output = np.zeros((len(frame), len(SUPPORT_NAMES)), dtype=np.float32)
    for i, peers in grouped.items():
        ps = [p for _, p in peers]
        weighted = [frame.first_stage.iat[j] * p for j, p in peers]
        opposite = [p for j, p in peers if frame.candidate_entity_id.iat[j][:2] != frame.candidate_entity_id.iat[i][:2]]
        output[i] = [len(peers), max(ps), np.mean(ps), min(ps), max(weighted),
            max(frame.first_stage.iat[j] * (1-p) for j, p in peers), max(opposite, default=0.)]
    for n, column in enumerate(SUPPORT_NAMES): result[column] = output[:, n]
    return result


def aligned_support(frame, path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if digest(path) != manifest.get("support_sha256") or pair_order_hash(frame) != manifest.get("pair_order_sha256"):
        raise ValueError("Support source hash or pair order changed")
    support = pd.read_parquet(path)
    if not support[KEYS].equals(frame[KEYS]): raise ValueError("Support keys do not align")
    if manifest.get("feature_names") != SUPPORT_NAMES or not np.isfinite(support[SUPPORT_NAMES]).all().all():
        raise ValueError("Invalid support feature schema")
    return pd.concat([frame, support[SUPPORT_NAMES]], axis=1), manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("route", "audit", "prepare", "fit", "support", "fit-residual", "rescore"))
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--countries", type=Path)
    parser.add_argument("--decision-config", type=Path, help="Optional development-only ideal-endpoint diagnostic with original cohort policy")
    parser.add_argument("--reference-ids", type=Path, help="JSON reference-ID list defining a disjoint role and its route cap")
    parser.add_argument("--evaluation-reference-ids", type=Path, help="Audit only: score these owners after routing/deciding complete claimant universe")
    parser.add_argument("--route-plan", type=Path, help="Hash-pinned global route directory; retain membership in chunk/subset inference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-prefix", default="models/index_train")
    parser.add_argument("--owner-db", type=Path, help="Read-only owners(id,owner,outer_fold) SQLite; labels only outer_fold=train")
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-file", type=Path, help="Fixed sibling model; --fold declares its excluded owner fold")
    parser.add_argument("--support", type=Path)
    parser.add_argument("--support-manifest", type=Path)
    parser.add_argument("--weight", type=float, default=.75, help="Predeclared residual blend; no implicit selection")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--training-support", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    if not 1 <= args.threads <= 20: parser.error("threads must be 1..20, subject to coordinator allocation")
    pa.set_cpu_count(args.threads)
    if args.output.exists(): raise FileExistsError(args.output)
    if args.command == "route" and args.reference_ids: parser.error("Global route must be selected before subset slicing")
    if args.command == "prepare" and args.route_plan: parser.error("OOF training mining has its separate p1 policy")
    args.output.mkdir(parents=True)
    config = RouteConfig()
    if args.command == "fit":
        import lightgbm as lgb
        if not args.examples: parser.error("fit requires --examples")
        data = pd.read_parquet(args.examples)
        reports = []
        for fold in ([args.fold] if args.fold is not None else [-1]):
            fit, valid = fold_masks(data, fold) if fold >= 0 else (np.ones(len(data), dtype=bool), np.zeros(len(data), dtype=bool))
            train = data[fit]
            if train.label.nunique() != 2: raise ValueError("Sibling fit needs real positives and hard negatives")
            dataset = lgb.Dataset(train[FEATURE_NAMES].to_numpy(dtype=np.float32), label=train.label, feature_name=FEATURE_NAMES)
            model = lgb.train(dict(objective="binary", metric="binary_logloss", num_leaves=15,
                min_data_in_leaf=60, lambda_l2=5., learning_rate=.04, max_bin=127,
                num_threads=args.threads, deterministic=True, force_col_wise=True, seed=42, verbosity=-1),
                dataset, num_boost_round=min(args.trees, 400))
            path = args.output / (f"fold_{fold}.txt" if fold >= 0 else "model.txt")
            model.save_model(str(path))
            validation = data[valid]
            report = {"excluded_fold": fold if fold >= 0 else None, "training_pairs": len(train),
                "validation_pairs_both_owners_held": len(validation), "model_sha256": digest(path),
                "training_owner_ids": sorted(set(train.left_owner) | set(train.right_owner)),
                "positive_pairs": int(train.label.sum()), "negative_pairs": int((train.label == 0).sum()),
                "same_source_pairs": int(sum(a[:2] == b[:2] for a, b in train[["left_id", "right_id"]].itertuples(index=False, name=None)))}
            if len(validation):
                p = np.clip(model.predict(validation[FEATURE_NAMES].to_numpy(dtype=np.float32), num_threads=args.threads), 1e-6, 1-1e-6)
                y = validation.label.to_numpy()
                report["owner_disjoint_logloss"] = float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))
            reports.append(report)
        write_json(args.output / "manifest.json", {"status": "complete", "features": FEATURE_NAMES,
            "feature_version": FEATURE_VERSION,
            "examples_sha256": digest(args.examples), "source_sha256": digest(__file__),
            "models": reports, "seconds": time.perf_counter()-started})
        return
    if not args.pairs or not args.manifest: parser.error("Need --pairs and --manifest")
    training = args.command == "prepare" or args.training_support
    frame, manifest = load_pairs(args.pairs, args.manifest, training=training)
    if args.truth and (manifest["split"] in ("audit", "cohort", "test") or args.command in ("route", "support", "rescore")):
        raise ValueError("Audit/test inputs permit label-free inference only")
    if args.command in ("audit", "fit-residual") and manifest["split"] != "development":
        raise ValueError("Development labels only for opportunity/residual fitting")
    truth = json.loads(args.truth.read_text()) if args.truth else None
    universe = manifest.get("reference_ids") or (list(truth) if truth is not None else list(frame.source1_entity_id.unique()))
    if args.reference_ids:
        wanted = json.loads(args.reference_ids.read_text())
        if len(wanted) != len(set(wanted)) or not set(wanted) <= set(universe): raise ValueError("Reference role IDs invalid")
        universe = wanted
        frame = frame[frame.source1_entity_id.isin(universe)].reset_index(drop=True)
        if truth is not None: truth = {e: truth[e] for e in universe}
    route_started = time.perf_counter()
    # Current p2 is in-sample on the outer-training population. Use only OOF p1
    # for hard-negative mining; retain the source table and its actual scores.
    routing_frame = frame.assign(probability=frame.first_stage) if args.command == "prepare" else frame
    route, pinned = load_pinned_route(frame, manifest, args.route_plan) if args.route_plan else (select_route(routing_frame, config, universe), None)
    route_seconds = time.perf_counter() - route_started
    common = {"route": asdict(config), "pairs_sha256": digest(args.pairs), "pair_order_sha256": pair_order_hash(frame),
        "feature_version": FEATURE_VERSION, "learned_comparison_weights": LEARNED_WEIGHTS,
        "reference_ids": sorted(universe),
        "input_manifest_sha256": digest(args.manifest), "route_references": len(set(frame.source1_entity_id.iat[i] for i in route)),
        "comparisons": sum(map(len, route.values())), "route_seconds": route_seconds,
        "route_probability_source": "owner_excluded_oof_first_stage" if args.command == "prepare" else "frozen_final_probability",
        "route_scope": "global_pinned_before_subset" if pinned else "declared_input_reference_universe",
        "route_plan_sha256": digest(args.route_plan / "manifest.json") if args.route_plan else None,
        "route_edges_sha256": pinned["route_sha256"] if pinned else None,
        "labels_used_for_route": False, "test_pseudo_labels": False}
    if args.command == "route":
        route_edges(frame, route).to_parquet(args.output / "route.parquet", index=False)
        write_json(args.output / "manifest.json", {**common, "status": "complete", "global_references": len(universe),
            "global_pairs_sha256": digest(args.pairs), "global_pair_order_sha256": pair_order_hash(frame),
            "frozen_sha256": manifest.get("frozen_sha256"), "route_sha256": digest(args.output / "route.parquet"),
            "seconds": time.perf_counter()-started})
        return
    if args.command in ("fit-residual", "rescore"):
        import lightgbm as lgb
        if not args.support or not args.support_manifest: parser.error("Residual commands need --support and --support-manifest")
        full, support_manifest = aligned_support(frame, args.support, args.support_manifest)
        if args.command == "fit-residual":
            if truth is None: parser.error("Residual fit needs separate --truth")
            if manifest["split"] != "development": raise ValueError("Residual owners must come from permitted development roles")
            if not args.reference_ids and manifest.get("role") != "residual_train":
                raise ValueError("Declare a residual-only role using --reference-ids or manifest role=residual_train")
            sibling_owners = support_manifest.get("fixed_training_owner_ids")
            if sibling_owners is None or set(universe) & set(sibling_owners):
                raise ValueError("Residual owners overlap supervised sibling owners or no owner ledger exists")
            train = full[full.sibling_count > 0].copy()
            labels = np.array([int(c in set(truth[s])) for s, c in train[KEYS].itertuples(index=False, name=None)])
            if len(set(labels)) != 2: raise ValueError("Residual fit needs both real classes on routed held owners")
            group_sizes = train.groupby("source1_entity_id").source1_entity_id.transform("size").to_numpy()
            dataset = lgb.Dataset(train[RESIDUAL_NAMES].to_numpy(dtype=np.float32), label=labels,
                weight=1/group_sizes, feature_name=RESIDUAL_NAMES)
            model = lgb.train(dict(objective="binary", num_leaves=7, min_data_in_leaf=100,
                lambda_l2=10., learning_rate=.04, max_bin=127, num_threads=args.threads,
                deterministic=True, force_col_wise=True, seed=42, verbosity=-1), dataset,
                num_boost_round=min(args.trees, 150))
            model.save_model(str(args.output / "residual.txt"))
            write_json(args.output / "manifest.json", {**common, "status": "complete", "feature_names": RESIDUAL_NAMES,
                "training_references": int(train.source1_entity_id.nunique()), "training_pairs": len(train),
                "residual_training_owner_ids": sorted(set(train.source1_entity_id)),
                "sibling_model_sha256": support_manifest["fixed_model_sha256"],
                "model_sha256": digest(args.output / "residual.txt"), "support_manifest_sha256": digest(args.support_manifest),
                "baseline_probability_is_oof": False,
                "baseline_owner_excluded": manifest.get("second_stage_owner_excluded"),
                "seconds": time.perf_counter()-started})
        else:
            if not args.model_dir or not 0 <= args.weight <= 1: parser.error("Rescore needs --model-dir and weight in [0,1]")
            residual_manifest = json.loads((args.model_dir / "manifest.json").read_text())
            if residual_manifest.get("sibling_model_sha256") != support_manifest.get("fixed_model_sha256"):
                raise ValueError("Inference sibling model differs from residual training transform")
            path = args.model_dir / "residual.txt"
            if digest(path) != residual_manifest.get("model_sha256"): raise ValueError("Residual model changed")
            model = lgb.Booster(model_file=str(path))
            if model.feature_name() != RESIDUAL_NAMES: raise ValueError("Residual feature order differs")
            routed = full.sibling_count > 0
            result = frame.copy()
            result["probability_original"] = result.probability
            result["residual_probability"] = result.probability
            if routed.any():
                predictions = model.predict(full.loc[routed, RESIDUAL_NAMES].to_numpy(dtype=np.float32), num_threads=args.threads)
                result.loc[routed, "residual_probability"] = predictions
                result.loc[routed, "probability"] = (1-args.weight)*result.loc[routed, "probability_original"] + args.weight*predictions
            result.to_parquet(args.output / "pairs.parquet", index=False)
            write_json(args.output / "manifest.json", {**common, "status": "scored_requires_global_decision_replay",
                "weight": args.weight, "changed_scores": int(routed.sum()), "pairs_sha256": digest(args.output / "pairs.parquet"),
                "seconds": time.perf_counter()-started})
        return
    if args.command == "audit":
        if manifest["split"] != "development" or truth is None: parser.error("audit requires development and separate --truth")
        countries = json.loads(args.countries.read_text()) if args.countries else None
        evaluation_ids = json.loads(args.evaluation_reference_ids.read_text()) if args.evaluation_reference_ids else None
        if evaluation_ids is not None and hashlib.sha256("\n".join(sorted(evaluation_ids)).encode()).hexdigest() != manifest.get("evaluation_reference_ids_sha256"):
            raise ValueError("Evaluation cohort differs from declared learned-workbench cohort")
        report = opportunity_audit(frame, truth, route, countries, evaluation_ids)
        if args.decision_config:
            from run_frozen_pipeline import decide
            decision_config = json.loads(args.decision_config.read_text())
            report["feasible_blend_diagnostic"] = feasible_score_diagnostic(frame, truth, route, decision_config, decide, evaluation_ids)
        expected = manifest.get("evaluation_expected_macro_f05") if evaluation_ids is not None else manifest.get("expected_macro_f05")
        if expected is None or abs(report["baseline_macro_f05"]-expected) > 1e-6:
            raise ValueError("Incumbent accepted-state baseline does not replay declared metric")
        route_edges(frame, route).to_parquet(args.output / "route.parquet", index=False)
        write_json(args.output / "report.json", {**common, **report, "seconds": time.perf_counter()-started})
    elif args.command == "prepare":
        if truth is None: parser.error("prepare requires outer-training owner truth")
        ids = {c for targets in truth.values() for c in targets}
        compared_ids = {frame.candidate_entity_id.iat[j] for i, peers in route.items() for j in [i, *peers]}
        extra_owners = lookup_training_owners(compared_ids, args.owner_db) if args.owner_db else {}
        ids.update(extra_owners)
        records = load_targets(ids, args.index_prefix)
        examples = training_examples(frame, route, truth, records, external_training_owners=extra_owners)
        examples.to_parquet(args.output / "examples.parquet", index=False)
        write_json(args.output / "manifest.json", {**common, "examples": len(examples),
            "positives": int(examples.label.sum()), "truth_sha256": digest(args.truth),
            "negatives": int((examples.label == 0).sum()), "compared_targets": len(compared_ids),
            "compared_targets_with_known_train_owner": len(compared_ids & (set(owner_map(truth)) | set(extra_owners))),
            "owner_db_sha256": digest(args.owner_db) if args.owner_db else None,
            "all_supervised_owner_ids": sorted(set(examples.left_owner) | set(examples.right_owner)),
            "examples_sha256": digest(args.output / "examples.parquet"), "seconds": time.perf_counter()-started})
    else:
        import lightgbm as lgb
        if not args.model_dir and not args.model_file: parser.error("support requires --model-dir or --model-file")
        ids = {frame.candidate_entity_id.iat[j] for i, peers in route.items() for j in [i, *peers]}
        lookup_started = time.perf_counter()
        records = load_targets(ids, args.index_prefix)
        lookup_seconds = time.perf_counter()-lookup_started
        result = frame[KEYS].copy()
        for n in SUPPORT_NAMES: result[n] = np.float32(0)
        folds = [args.fold if args.fold is not None else -1] if args.model_file else list(range(5)) if training else [-1]
        fixed_sha = None
        fixed_owners = None
        predict_started = time.perf_counter()
        for fold in folds:
            local_route = {i: peers for i, peers in route.items() if fold < 0 or inner_fold(frame.source1_entity_id.iat[i]) == fold}
            # At inference, use the SAME fixed transform as residual training on
            # all references, not only references hashing to its excluded fold.
            if args.model_file and not training: local_route = route
            path = args.model_file or args.model_dir / (f"fold_{fold}.txt" if fold >= 0 else "model.txt")
            if args.model_file:
                fixed_sha = digest(path)
                model_manifest = json.loads((path.parent / "manifest.json").read_text())
                matches = [m for m in model_manifest["models"] if m["model_sha256"] == fixed_sha and m["excluded_fold"] == args.fold]
                if len(matches) != 1:
                    raise ValueError("Fixed model excluded-fold declaration does not match fitted manifest")
                fixed_owners = matches[0]["training_owner_ids"]
                if training and set(frame.source1_entity_id.iat[i] for i in local_route) & set(fixed_owners):
                    raise ValueError("Training support's owners occur in fitted sibling model")
            model = lgb.Booster(model_file=str(path))
            if model.feature_name() != FEATURE_NAMES: raise ValueError("Sibling model feature order changed")
            features = support_features(frame, local_route, records, model, args.threads)
            indices = list(local_route)
            result.loc[indices, SUPPORT_NAMES] = features.loc[indices, SUPPORT_NAMES]
        feature_prediction_seconds = time.perf_counter()-predict_started
        result.to_parquet(args.output / "support.parquet", index=False)
        elapsed = time.perf_counter()-started
        write_json(args.output / "manifest.json", {**common, "feature_names": SUPPORT_NAMES,
            "owner_excluded": not bool(set(universe) & set(fixed_owners)) if fixed_owners is not None else training,
            "support_sha256": digest(args.output / "support.parquet"),
            "fixed_excluded_fold": args.fold if args.model_file else None, "fixed_model_sha256": fixed_sha,
            "fixed_training_owner_ids": fixed_owners,
            "lookup_seconds": lookup_seconds, "lookup_records": len(records),
            "feature_and_prediction_seconds": feature_prediction_seconds,
            "seconds": elapsed, "references_per_second": len(universe)/max(elapsed, 1e-9),
            "runtime_scope": "Sibling addition: pair/support I/O, hashing, route, record lookup, features and prediction; excludes baseline and global decisions."})


if __name__ == "__main__": main()
