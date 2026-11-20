"""Parallel collection integrity checks using synthetic files, without physics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from clean import parallel_collect as collect


class ParallelCollectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "results").mkdir()
        (self.root / "data/plans").mkdir(parents=True)
        self.sources = {}
        for name in collect.CRITICAL_SOURCES:
            (self.root / name).write_text("synthetic source")
            self.sources[name] = collect.digest(self.root / name)
        self.settings = [{"episode_id": i} for i in range(10)]
        (self.root / "data/episode_settings.json").write_text(json.dumps(self.settings))
        for i in range(10):
            (self.root / "data/plans" / f"episode_{i:02d}.npz").write_bytes(b"synthetic read-only plan")

    def test_gate_rejects_stale_source_and_changed_pilot_data(self):
        episode = self.root / "data/pilot.npz"
        episode.write_bytes(b"pilot")
        report = {"status": "pass", "static_provenance": {"source_sha256": dict(self.sources)},
                  "episode_sha256": {"pilot.npz": collect.digest(episode)}}
        path = self.root / "results/pilot_audit.json"
        path.write_text(json.dumps(report))
        with mock.patch.object(collect, "sample_settings", side_effect=lambda i: self.settings[i]):
            settings, plans = collect.validate_pilot_gate(self.root)
            self.assertEqual(settings, self.settings)
            self.assertEqual(len(plans), 10)
            episode.write_bytes(b"changed pilot")
            with self.assertRaisesRegex(RuntimeError, "Pilot gate data changed"):
                collect.validate_pilot_gate(self.root)
            episode.write_bytes(b"pilot")
            (self.root / "script.py").write_text("changed generator")
            with self.assertRaisesRegex(RuntimeError, "different generator"):
                collect.validate_pilot_gate(self.root)

    def test_receipt_requires_identity_data_and_generator_integrity(self):
        path = self.root / "data/episode.npz"
        path.write_bytes(b"synthetic data")
        receipt = {"cube_id": 7, "episode_id": 3, "sha256": collect.digest(path),
                   "source_sha256": self.sources}
        path.with_suffix(".json").write_text(json.dumps(receipt))
        self.assertEqual(collect.verify_receipt(path, 7, 3, self.sources), receipt)
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            collect.verify_receipt(path, 8, 3, self.sources)
        changed_sources = dict(self.sources)
        changed_sources["script.py"] = "new script"
        with self.assertRaisesRegex(RuntimeError, "different generator"):
            collect.verify_receipt(path, 7, 3, changed_sources)
        path.write_bytes(b"changed data")
        with self.assertRaisesRegex(RuntimeError, "failed integrity"):
            collect.verify_receipt(path, 7, 3, self.sources)

    def test_atomic_publication_never_overwrites_an_existing_episode(self):
        destination = self.root / "data/episode.npz"
        temporary = self.root / "data/episode.partial"
        destination.write_bytes(b"existing authoritative data")
        temporary.write_bytes(b"new data")
        with self.assertRaises(FileExistsError):
            collect._publish_exclusive(temporary, destination)
        self.assertEqual(destination.read_bytes(), b"existing authoritative data")
        fresh = self.root / "data/fresh.npz"
        collect._publish_exclusive(temporary, fresh)
        self.assertEqual(fresh.read_bytes(), b"new data")
        self.assertFalse(temporary.exists())


if __name__ == "__main__":
    unittest.main()
