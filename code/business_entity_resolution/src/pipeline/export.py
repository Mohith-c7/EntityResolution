"""Standard-library output validator with a disk-backed candidate lookup."""

import csv
import sqlite3
import tempfile
from pathlib import Path


def source_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if "entity_id" not in (reader.fieldnames or []):
            raise ValueError(f"Missing entity_id column: {path}")
        return {row["entity_id"] for row in reader}


def output_rows(path: Path, columns: tuple[str, str]):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if stream.readline().rstrip("\r\n").split("\t") != list(columns):
            raise ValueError(f"Invalid columns in {path.name}; expected {list(columns)}")
        for number, line in enumerate(stream, 2):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path.name} line {number}: expected exactly two tab-separated fields")
            eid, raw = fields
            ids = raw.split(",") if raw else []
            if eid != eid.strip() or not eid:
                raise ValueError(f"{path.name} line {number}: malformed Source 1 ID")
            if any(not cid or cid != cid.strip() for cid in ids):
                raise ValueError(f"{path.name} line {number}: malformed ID list")
            yield eid, ids, raw


def validate_submission(matching_path, candidate_path, test_dir=None, test_s1_ids=None, test_s2_ids=None, test_s3_ids=None) -> list[str]:
    matching_path, candidate_path = Path(matching_path), Path(candidate_path)
    errors = []
    for kind,path in (("Matching",matching_path),("Candidate",candidate_path)):
        if not path.is_file():
            errors.append(f"{kind} file not found: {path}")
    if errors:
        return errors
    try:
        supplied = [test_s1_ids, test_s2_ids, test_s3_ids]
        for position in range(3):
            if supplied[position] is None:
                if test_dir is None:
                    raise ValueError("No test IDs or test_dir provided")
                supplied[position] = source_ids(Path(test_dir) / f"test_source{position+1}.tsv")
        s1, s2, s3 = supplied
        def validate_list(eid, ids, filename, kind):
            if len(ids) != len(set(ids)):
                errors.append(f"Duplicate candidate IDs found in {kind} list for {eid} ({filename})")
            for cid in ids:
                if cid.startswith("S2-"):
                    valid = cid in s2
                elif cid.startswith("S3-"):
                    valid = cid in s3
                else:
                    reason="Forbidden S1 ID" if cid.startswith("S1-") else "Invalid candidate prefix"
                    errors.append(f"{reason}: {cid} in {filename}")
                    continue
                if not valid:
                    errors.append(f"Candidate {cid} does not exist in test Source {cid[1]} ({filename})")
        def validate_coverage(seen,filename):
            if s1-seen:
                errors.append(f"{filename} is missing {len(s1-seen)} test S1 entities")
            if seen-s1:
                errors.append(f"{filename} contains {len(seen-s1)} S1 IDs not in test_source1")
        with tempfile.TemporaryDirectory(prefix="er-validation-") as folder:
            connection = sqlite3.connect(Path(folder) / "candidates.sqlite")
            try:
                connection.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE TABLE candidates(id TEXT PRIMARY KEY, ids TEXT);")
                candidate_seen = set()
                for eid, ids, raw in output_rows(candidate_path, ("source1_entity_id", "candidate_entity_ids")):
                    if eid in candidate_seen:
                        errors.append(f"Found duplicate S1 rows in candidate_pairs.tsv: {eid}")
                        continue
                    candidate_seen.add(eid)
                    validate_list(eid, ids, candidate_path.name,"candidate")
                    connection.execute("INSERT INTO candidates VALUES (?,?)", (eid, raw))
                validate_coverage(candidate_seen,"candidate_pairs.tsv")
                connection.commit()
                matching_seen = set()
                for eid, ids, raw in output_rows(matching_path, ("source1_entity_id", "matched_entity_ids")):
                    if eid in matching_seen:
                        errors.append(f"Found duplicate S1 rows in matching_results.tsv: {eid}")
                    matching_seen.add(eid)
                    validate_list(eid, ids, matching_path.name,"match")
                    row = connection.execute("SELECT ids FROM candidates WHERE id=?", (eid,)).fetchone()
                    candidates = set(row[0].split(",")) if row and row[0] else set()
                    if set(ids) - candidates:
                        errors.append(f"Final match is not in candidate pool for {eid}; violates subset invariant")
                validate_coverage(matching_seen,"matching_results.tsv")
            finally:
                connection.close()
    except (OSError, ValueError, sqlite3.Error, KeyError) as exc:
        errors.append(str(exc))
    return errors
