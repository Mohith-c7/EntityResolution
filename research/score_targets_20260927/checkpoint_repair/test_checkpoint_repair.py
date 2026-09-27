import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import migrate_checkpoint_markers as migration
import prepare_repair_frozen as frozen


class MarkerRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.chunks = self.root / "chunks"
        self.chunks.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, source_rows, candidate_rows, source_ids, *, chunk=0):
        parquet = self.chunks / f"{chunk:07d}.parquet"
        parquet.write_bytes(b"local parquet fixture bytes")
        marker = {
            "chunk": chunk,
            "entities": len(source_ids),
            "pairs": len(source_rows),
            "input_ids_sha256": migration.sha256_ids(sorted({c for c in candidate_rows if c.startswith("S3")})),
            "parquet_sha256": migration.sha256_file(parquet),
        }
        (self.chunks / f"{chunk:07d}.json").write_text(json.dumps(marker))
        return lambda _path: (source_rows, candidate_rows, len(source_rows))

    def test_missing_candidate_source1_singleton_is_allowed_and_still_hashed(self):
        ids = ["S1-a", "S1-singleton", "S1-c"]
        reader = self.fixture(["S1-a", "S1-c"], ["S3-x", "S2-y"], ids)
        plans = migration.inspect_markers(ids, self.chunks, batch_size=3, parquet_reader=reader)
        self.assertEqual(plans[0].state, "legacy_s3_candidate_hash")
        self.assertEqual(plans[0].missing_source1_ids_without_candidates, 1)
        self.assertEqual(plans[0].new_input_ids_sha256, migration.sha256_ids(ids))

    def test_all_candidate_free_singletons_keep_already_correct_hash(self):
        ids = ["S1-a", "S1-b"]
        reader = self.fixture([], [], ids)
        marker_path = self.chunks / "0000000.json"
        marker = json.loads(marker_path.read_text())
        marker["input_ids_sha256"] = migration.sha256_ids(ids)
        marker_path.write_text(json.dumps(marker))
        plans = migration.inspect_markers(ids, self.chunks, batch_size=2, parquet_reader=reader)
        self.assertEqual(plans[0].state, "current_reference_hash")
        self.assertEqual(plans[0].missing_source1_ids_without_candidates, 2)

    def test_parquet_source1_outside_slice_is_rejected(self):
        ids = ["S1-a", "S1-b"]
        reader = self.fixture(["S1-other"], ["S3-x"], ids)
        with self.assertRaisesRegex(migration.RepairError, "outside its current slice"):
            migration.inspect_markers(ids, self.chunks, batch_size=2, parquet_reader=reader)

    def test_apply_backs_up_manifest_then_updates_only_marker_hash(self):
        ids = ["S1-a", "S1-singleton"]
        reader = self.fixture(["S1-a"], ["S3-x"], ids)
        plans = migration.inspect_markers(ids, self.chunks, batch_size=2, parquet_reader=reader)
        before = json.loads((self.chunks / "0000000.json").read_text())
        backup = self.root / "backup"
        changed = migration.apply_repairs(plans, backup, {"markers": [migration.asdict(plans[0])]})
        after = json.loads((self.chunks / "0000000.json").read_text())
        backed_up = json.loads((backup / "markers" / "0000000.json").read_text())
        self.assertEqual(changed, 1)
        self.assertEqual(backed_up, before)
        self.assertEqual(after["parquet_sha256"], before["parquet_sha256"])
        self.assertEqual(after["input_ids_sha256"], migration.sha256_ids(ids))
        self.assertTrue((backup / "manifest.json").exists())

    def test_universe_metadata_blocks_changed_source_order(self):
        source = self.root / "test_source1.tsv"
        source.write_text("entity_id\nS1-a\nS1-b\n")
        database = self.root / "counts.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
            connection.execute(
                "INSERT INTO meta VALUES ('universe', ?)",
                (json.dumps({"records": 2, "sha256": "wrong"}),),
            )
        with self.assertRaisesRegex(migration.RepairError, "differs"):
            migration.validate_universe(source, database, ["S1-a", "S1-b"])


class FrozenDescriptorTests(unittest.TestCase):
    def test_exact_patch_changes_only_expected_metadata_variables(self):
        root = Path(__file__).resolve().parents[3]
        original = (root / "scripts/run_frozen_pipeline.py").read_text()
        patched = frozen.expected_patched_runner(original)
        self.assertIn("reference_ids =", patched)
        self.assertIn("candidate_ids =", patched)
        with self.assertRaises(frozen.FreezeRepairError):
            frozen.expected_patched_runner(patched)

    def test_new_descriptor_pins_patch_and_preserves_audit_provenance(self):
        root = Path(__file__).resolve().parents[3]
        original_runner_bytes = (root / "scripts/run_frozen_pipeline.py").read_bytes()
        original_runner_text = original_runner_bytes.decode()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original_runner = directory / "original_runner.py"
            patched_runner = directory / "patched_runner.py"
            descriptor_path = directory / "frozen.json"
            audit_path = directory / "report.json"
            original_runner.write_bytes(original_runner_bytes)
            patched_runner.write_text(frozen.expected_patched_runner(original_runner_text))
            descriptor = {
                "status": "frozen_before_audit",
                "code_sha256": {
                    "scripts/run_frozen_pipeline.py": frozen.digest_file(original_runner),
                },
            }
            descriptor_path.write_text(json.dumps(descriptor))
            original_descriptor_bytes = descriptor_path.read_bytes()
            audit_path.write_text(
                json.dumps(
                    {
                        "status": "fresh_audit_complete",
                        "frozen_sha256": hashlib.sha256(original_descriptor_bytes).hexdigest(),
                        "ship_decision": {"accepted": True},
                    }
                )
            )
            repaired, provenance = frozen.build_descriptor(
                descriptor_path, original_runner, patched_runner, audit_path
            )
            self.assertEqual(descriptor_path.read_bytes(), original_descriptor_bytes)
            self.assertEqual(
                repaired["code_sha256"]["scripts/run_frozen_pipeline.py"],
                frozen.digest_file(patched_runner),
            )
            self.assertEqual(provenance["score_logic_parity"], "exact_expected_metadata_only_patch")
            self.assertTrue(provenance["original_audit_ship_accepted"])


if __name__ == "__main__":
    unittest.main()
