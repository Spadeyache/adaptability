"""Guard the scientific checks without loading any real held-out data."""
import tempfile
import unittest
import hashlib
import json
from pathlib import Path

import numpy as np

from clean.audit import (
    ACTION_COLUMNS, COLUMNS, correlation_report, load_training_episodes,
    moving_mask, pearson, r2_per_property, static_provenance, summarize,
    verify_collection_receipts, CRITICAL_COLLECTION_SOURCES, ROOT,
)


class AuditTests(unittest.TestCase):
    def test_constant_prediction_and_undefined_r2(self):
        target = np.array([[1.0, 2.0], [3.0, 2.0]])
        scores = r2_per_property(target, np.array([[2.0, 2.0], [2.0, 2.0]]))
        self.assertEqual(scores[0], 0.0)
        self.assertTrue(np.isnan(scores[1]))
        self.assertIsNone(pearson(np.ones(5), np.arange(5)))

    def test_motion_threshold_catches_rotation_and_translation(self):
        names = [column["name"] for column in COLUMNS]
        state = np.zeros((3, len(COLUMNS)))
        rotation = [names.index(f"cube_rotation_{i}") for i in range(6)]
        state[:, rotation] = [1, 0, 0, 0, 1, 0]
        next_state = state.copy()
        next_state[1, names.index("cube_position_0")] = 0.0002
        angle = 0.002
        next_state[2, rotation] = [np.cos(angle), np.sin(angle), 0, -np.sin(angle), np.cos(angle), 0]
        np.testing.assert_array_equal(moving_mask(state, next_state), [False, True, True])

    def test_suspicious_command_correlation_fails_but_sensor_allowed(self):
        labels = np.column_stack((np.arange(10), np.arange(10)[::-1]))
        summaries = np.zeros((10, 4, len(COLUMNS) + len(ACTION_COLUMNS)))
        summaries[:, :, 0] = np.arange(10)[:, None]
        report = correlation_report(summaries, labels)
        self.assertTrue(report["pass"])
        self.assertTrue(all(flag["classification"] == "physical_sensor" for flag in report["flags_abs_r_gt_0_9"]))
        summaries[:, :, len(COLUMNS)] = np.arange(10)[:, None]
        report = correlation_report(summaries, labels)
        self.assertFalse(report["pass"])

    def test_nontraining_cube_rejected_before_file_read(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "nontraining cube"):
                load_training_episodes(folder, [200])

    def test_label_field_rejected_in_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            train = Path(folder) / "train"
            train.mkdir()
            path = train / "cube_0000_episode_00.npz"
            np.savez(path, state=np.zeros((1, len(COLUMNS))), mass_kg=np.array([0.1]))
            with self.assertRaisesRegex(ValueError, "Unexpected fields"):
                load_training_episodes(folder, [0])

    def test_import_or_property_leak_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "schema.py").write_text("def observe(env):\n    return env.mass\n")
            (root / "script.py").write_text("from robot_adaptation import policy\ndef plan_episode(cube_pose, settings):\n    return []\n")
            report = static_provenance(root)
            self.assertFalse(report["pass"])
            self.assertTrue(any("property read mass" in error for error in report["violations"]))
            self.assertTrue(any("old-project import" in error for error in report["violations"]))

    def test_sample_standard_deviation(self):
        result = summarize([0, 1, 2, 3, 4])
        self.assertEqual(result["mean"], 2)
        self.assertAlmostEqual(result["std"], np.std([0, 1, 2, 3, 4], ddof=1))

    def test_collection_receipt_detects_data_and_source_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            train = Path(folder) / "train"
            train.mkdir()
            sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CRITICAL_COLLECTION_SOURCES}
            for episode in range(10):
                path = train / f"cube_0000_episode_{episode:02d}.npz"
                np.savez(path, state=np.zeros((1, len(COLUMNS))))
                receipt = {"cube_id": 0, "episode_id": episode, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_sha256": sources}
                path.with_suffix(".json").write_text(json.dumps(receipt))
            self.assertEqual(verify_collection_receipts(folder, [0])["verified_episode_receipts"], 10)
            path = train / "cube_0000_episode_00.npz"
            receipt_path = path.with_suffix(".json")
            receipt = json.loads(receipt_path.read_text())
            receipt["source_sha256"]["schema.py"] = "changed"
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                verify_collection_receipts(folder, [0])
            receipt["sha256"] = "changed"
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "data digest mismatch"):
                verify_collection_receipts(folder, [0])


if __name__ == "__main__":
    unittest.main()
