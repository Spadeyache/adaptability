"""Review provenance and coverage must never silently become automatic labels."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot_adaptation.analysis.failure_discovery.visual_review import navigation_indices, report


class VisualReviewTests(unittest.TestCase):
    def test_event_frames_are_chronological_unique_and_include_end(self):
        obj = np.zeros((101, 7))
        obj[:, 2] = .02
        obj[5:10, 2] = .06
        grasp = np.zeros(101, bool)
        grasp[4:9] = True
        d = {"meta__state_time_s": np.arange(101) / 20,
             "input__object_pose_world_m_wxyz": obj,
             "input__tcp_pose_world_m_wxyz": obj.copy(),
             "sim__is_grasped": grasp, "sim__object_contact": grasp.copy()}
        frames = navigation_indices(d)
        self.assertEqual(frames, sorted(set(frames)))
        self.assertEqual(len(frames), 6)
        self.assertIn(100, frames)
        self.assertIn(8, frames)  # immediately before recorded release
        self.assertIn(11, frames)  # after recorded release

    def test_no_annotation_means_unreviewed_even_with_original_failure_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            inventory = [{"review_id": "E001", "episode_id": "original", "source_file": "original.npz",
                          "source_sha256": "saved-source-hash", "success": False, "failure_type": "drop"}]
            (out / "inventory.json").write_text(json.dumps(inventory))
            summary = report(out)
            self.assertEqual(summary["visually_reviewed"], 0)
            self.assertEqual(summary["unreviewed"], 1)
            self.assertEqual(summary["visual_type_counts"]["slip_or_drop"], 0)
            rows = json.loads((out / "review_data.json").read_text())["rows"]
            self.assertEqual(rows[0]["primary_visual_type"], "")
            self.assertEqual(rows[0]["original_failure_type"], "drop")

    def test_duplicate_annotations_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "inventory.json").write_text(json.dumps([{"review_id": "E001"}]))
            annotation = {"review_id": "E001", "primary_visual_type": "uncertain", "confidence": "low",
                          "evidence": "Contact is occluded", "review_basis": "inspected storyboard", "visual_tags": ["view_occluded"]}
            (out / "annotations.json").write_text(json.dumps([annotation, annotation]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                report(out)


if __name__ == "__main__":
    unittest.main()
