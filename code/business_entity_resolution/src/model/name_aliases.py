"""A train-fold-only name noise channel learned from provided match labels.

This resolves name strings, not entity IDs. Ambiguous aliases are withheld.
No validation counterpart contributes to fitted alias state.
"""

import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..evaluation.validation import entity_fold
from ..preprocessing.normalize import name_core, normalize_name

ALIAS_VERSION = "name-noise-channel-v2"


class NameAliases:
    def __init__(self, entries: dict, metadata: dict, alternatives: dict | None = None):
        self.entries, self.metadata = entries, metadata
        self.alternatives = alternatives or {}
        families = defaultdict(list)
        for alias, (canonical, support, total) in entries.items():
            families[canonical].append((support, alias))
        for alias, possibilities in self.alternatives.items():
            for canonical, support, total in possibilities:
                families[canonical].append((support, alias))
        self.families = {name: [alias for _,alias in sorted(values,key=lambda item:(-item[0],item[1]))[:32]]
                         for name, values in families.items()}

    def resolve(self, core: str):
        entry = self.entries.get(core)
        if not entry:
            return core, 0.0, 0
        canonical, support, total = entry
        return canonical, support / total, support

    def variants(self, canonical: str) -> list[str]:
        return [canonical, *[alias for alias in self.families.get(canonical, ()) if alias != canonical]][:32]

    def evidence(self, alias: str, canonical: str):
        entry = self.entries.get(alias)
        if entry and entry[0] == canonical:
            return entry[1] / entry[2], entry[1]
        for name, support, total in self.alternatives.get(alias, ()):
            if name == canonical:
                return support / total, support
        return 0.0, 0

    @classmethod
    def load(cls, path: Path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value["metadata"]["version"] not in (ALIAS_VERSION,"name-noise-channel-v1"):
            raise ValueError("Unsupported alias model version")
        return cls(value["entries"], value["metadata"], value.get("alternatives"))


def fit_name_aliases(train_dir: Path, output: Path, *, seed=42, min_support=2, min_dominance=.98) -> NameAliases:
    train_dir, output = Path(train_dir), Path(output)
    training_files = {name:{"size":(train_dir/name).stat().st_size,"mtime_ns":(train_dir/name).stat().st_mtime_ns}
        for name in ("train_source1.tsv","train_source2.tsv","train_source3.tsv","train_ground_truth.tsv")}
    if output.exists():
        aliases = NameAliases.load(output)
        if (aliases.metadata["version"] != ALIAS_VERSION or aliases.metadata["seed"] != seed or aliases.metadata.get("training_files") != training_files
                or aliases.metadata["min_support"] != min_support or aliases.metadata["min_dominance"] != min_dominance):
            raise ValueError("Alias input fingerprint or fitting settings differ; use a fresh artifact directory")
        return aliases
    start = time.perf_counter()
    canonical_by_reference = {}
    with (train_dir / "train_source1.tsv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            eid = row["entity_id"]
            if entity_fold(eid, seed) == "train":
                canonical = name_core(normalize_name(row["business_name"]))
                if canonical:
                    canonical_by_reference[eid] = sys.intern(canonical)
    owner_by_target = {}
    with (train_dir / "train_ground_truth.tsv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            canonical = canonical_by_reference.get(row["source1_entity_id"])
            if canonical:
                for mid in row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else ():
                    owner_by_target[mid] = canonical
    fitted_references, fitted_targets = len(canonical_by_reference), len(owner_by_target)
    del canonical_by_reference
    counts = defaultdict(Counter)
    for number in (2, 3):
        processed = 0
        with (train_dir / f"train_source{number}.tsv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                canonical = owner_by_target.get(row["entity_id"])
                if canonical:
                    alias = name_core(normalize_name(row["business_name"]))
                    if alias:
                        counts[alias][canonical] += 1
                        processed += 1
        print(json.dumps({"stage":"alias_fit", "source":f"S{number}", "training_targets":processed,
            "name_strings":len(counts), "seconds":round(time.perf_counter()-start,1)}),flush=True)
    entries, alternatives = {}, {}
    for alias, histogram in counts.items():
        canonical, support = min(histogram.items(), key=lambda item:(-item[1],item[0]))
        total = sum(histogram.values())
        if support >= min_support and support / total >= min_dominance:
            entries[alias] = [canonical, support, total]
        else:
            possibilities = [[name, count, total] for name, count in sorted(histogram.items(),key=lambda item:(-item[1],item[0]))[:3]
                if count >= min_support and count / total >= .10]
            if possibilities:
                alternatives[alias] = possibilities
    after = {name:{"size":(train_dir/name).stat().st_size,"mtime_ns":(train_dir/name).stat().st_mtime_ns} for name in training_files}
    if after != training_files:
        raise ValueError("Training files changed during alias fitting")
    metadata = {"version":ALIAS_VERSION,"seed":seed,"fitted_fold":"train","training_files":training_files,"fitted_references":fitted_references,
        "fitted_targets":fitted_targets,"min_support":min_support,"min_dominance":min_dominance,
        "alias_strings":len(entries),"ambiguous_alias_strings":len(alternatives),"observed_strings":len(counts),"seconds":time.perf_counter()-start,
        "license":"MIT","policy":"Only matched targets owned by training-fold S1 references contribute. No external lookup, validation labels, or entity-ID features."}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({"metadata":metadata,"entries":entries,"alternatives":alternatives},ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"stage":"alias_ready",**metadata}),flush=True)
    return NameAliases(entries,metadata,alternatives)
