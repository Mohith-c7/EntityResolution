import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution"))
from src.blocking.disk_index import build_disk_index
from src.evaluation.validation import entity_fold
from src.model.crossfit_aliases import build_counts, export_aliases, inner_fold
from src.model.name_aliases import NameAliases, fit_name_aliases


def test_alias_retrieval_and_features_exclude_owned_inner_fold(tmp_path):
    def ref(outer, inner=None):
        return next(f"S1-{n}" for n in range(10000) if entity_fold(f"S1-{n}") == outer
                    and (inner is None or inner_fold(f"S1-{n}") == inner))
    a, b, v = ref("train", 0), ref("train", 1), ref("holdout")
    train = tmp_path / "train"; train.mkdir()
    indexes = tmp_path / "indexes"; indexes.mkdir()
    columns = ["entity_id", "business_name", "business_address", "country"]
    data = {1: [(a, "Alpha Ltd", "1 Road", "Novel Country"), (b, "Alpha Ltd", "2 Road", "Novel Country"),
                (v, "Leaky", "3 Road", "Novel Country")],
            2: [("S2-a1", "Secret-A", "1 Road", "Novel Country"), ("S2-a2", "Secret-A", "1 Road", "Novel Country"),
                ("S2-b1", "Shared", "2 Road", "Novel Country"), ("S2-b2", "Shared", "2 Road", "Novel Country")],
            3: [("S3-a1", "Shared", "1 Road", "Novel Country"), ("S3-a2", "Shared", "1 Road", "Novel Country"),
                ("S3-v1", "LeakAlias", "3 Road", "Novel Country"), ("S3-v2", "LeakAlias", "3 Road", "Novel Country")]}
    for number, rows in data.items():
        path = train / f"train_source{number}.tsv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t"); writer.writerow(columns); writer.writerows(rows)
        if number > 1: build_disk_index(path, indexes / f"index_train_S{number}.sqlite", f"S{number}")
    with (train / "train_ground_truth.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t"); writer.writerow(["source1_entity_id", "matched_entity_ids"])
        writer.writerows([(a, "S2-a1,S2-a2,S3-a1,S3-a2"), (b, "S2-b1,S2-b2"), (v, "S3-v1,S3-v2")])
    database = tmp_path / "counts.sqlite"
    metadata = build_counts(train, indexes, database)
    assert metadata["fitted_targets"] == 6
    full, excluded = tmp_path / "full.json", tmp_path / "excluded.json"
    export_aliases(database, full)
    export_aliases(database, excluded, 0)
    full_model, model = NameAliases.load(full), NameAliases.load(excluded)
    assert full_model.evidence("secret a", "alpha")[0] == 1
    assert model.evidence("secret a", "alpha")[0] == 0
    assert "secret a" not in model.variants("alpha")
    assert model.evidence("shared", "alpha") == (1., 2)
    assert full_model.evidence("leakalias", "leaky")[0] == 0
    old = fit_name_aliases(train, tmp_path / "old.json")
    assert old.entries == full_model.entries
    assert old.alternatives == full_model.alternatives
    assert build_counts(train, indexes, database) == metadata
    assert export_aliases(database, excluded, 0)["excluded_inner_fold"] == 0
