"""Causal snippet boundaries, privileged-input isolation and persistent inference."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from robot_adaptation.estimators.interaction import ARCHITECTURE
from robot_adaptation.estimators.interaction.data import (
    FEATURE_NAMES, FEATURE_SPECS, condition_splits, extract_snippets, transition_features,
)
from robot_adaptation.estimators.interaction.model import InteractionEstimator
from robot_adaptation.estimators.interaction.predict import InteractionPredictor


def episode(compression, goal_distance=.1):
    compression = np.asarray(compression, dtype=float)
    if compression.ndim == 1:
        compression = np.repeat(compression[:, None], 2, axis=1)
    n = len(compression)
    data = {name: np.zeros((n + (timeline == "state"), size), dtype=np.float32)
            for name, timeline, size in FEATURE_SPECS}
    for name, timeline, size in FEATURE_SPECS:
        if size == 1:
            data[name] = data[name][:, 0]
    data["input__finger_compression_mean_n"] = compression
    data["input__object_pose_world_m_wxyz"] = np.zeros((n + 1, 7))
    data["input__object_goal_distance_m"] = np.full(n + 1, goal_distance)
    data["meta__state_time_s"] = np.arange(n + 1) * .05
    return data


class SnippetTests(unittest.TestCase):
    def test_first_contact_response_and_release_alignment(self):
        data = episode([0, 2, 2, 0, 0, 0, 0])
        data["input__joint_position_rad_m"][:, 0] = np.arange(8)
        data["input__action_applied"][:, 0] = 20 + np.arange(7)
        clips = extract_snippets(data)
        self.assertEqual(len(clips), 1)
        clip = clips[0]
        self.assertEqual((clip["start"], clip["end"], clip["reason"]), (1, 6, "release"))
        self.assertEqual(clip["x"][0, 0], 1)
        self.assertEqual(clip["x"][0, FEATURE_NAMES.index("input__action_applied[0]")], 21)
        self.assertAlmostEqual(clip["end_time_s"], .3)

    def test_privileged_values_never_change_observable_clips(self):
        data = episode([0, 2, 2, 0, 0, 0])
        before = extract_snippets(data)
        data.update(label__mass_kg=np.asarray(500.), label__object_static_friction=np.asarray(999.),
                    sim__object_contact=np.ones(7), sim__is_grasped=np.ones(7),
                    sim__task_success=np.ones(7), input__reported_mass_kg=np.asarray(777.))
        after = extract_snippets(data)
        np.testing.assert_array_equal(before[0]["x"], after[0]["x"])
        np.testing.assert_array_equal(before[0]["outcome"], after[0]["outcome"])
        self.assertEqual(before[0]["end"], after[0]["end"])

    def test_goal_holding_ends_once_without_repeated_tail(self):
        clips = extract_snippets(episode([2.] * 15, goal_distance=.001))
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]["reason"], "stable_hold")
        self.assertTrue(clips[0]["outcome"][0])
        self.assertLess(clips[0]["end"], 15)

    def test_no_contact_and_terminal_contact_timeout(self):
        self.assertEqual(extract_snippets(episode([0.] * 8)), [])
        clips = extract_snippets(episode([0.] * 6 + [2., 2.]))
        self.assertEqual((clips[0]["start"], clips[0]["end"], clips[0]["reason"]), (6, 8, "timeout"))

    def test_later_outcomes_do_not_enter_earlier_clip(self):
        data = episode([2., 0, 0, 0, 0, 2, 2, 2])
        before = extract_snippets(data)[0]
        data["input__object_goal_distance_m"][6:] = 0
        data["input__action_applied"][6:] = 1
        after = extract_snippets(data)[0]
        np.testing.assert_array_equal(before["x"], after["x"])
        np.testing.assert_array_equal(before["outcome"], after["outcome"])

    def test_split_preserves_all_attempts_of_a_condition(self):
        rows = [dict(condition_id=f"c{i}") for i in range(20) for _ in range(10)]
        split = condition_splits(rows)
        self.assertEqual(len(split), 20)
        self.assertEqual(set(split.values()), {"train", "validation", "test"})
        self.assertEqual(split, condition_splits(rows))

    def test_invalid_response_lengths_rejected(self):
        data = episode([2.] * 5)
        data["input__action_applied"] = np.zeros((4, 5))
        with self.assertRaises(ValueError):
            transition_features(data)


class PredictorTests(unittest.TestCase):
    def test_training_and_inference_average_bouts_identically(self):
        torch.manual_seed(5)
        model = InteractionEstimator(8)
        x = torch.randn(3, 4, len(FEATURE_NAMES))
        lengths = torch.tensor([4, 3, 2])
        outcomes = torch.randn(3, 4)
        owners = torch.tensor([0, 0, 1])
        embedding = model.encode(x, lengths, outcomes)
        first_attempt = embedding[:2].mean(dim=0)
        second_attempt = embedding[2]
        expected = torch.stack([model.head(first_attempt), model.head((first_attempt + second_attempt) / 2)])
        actual = model(x, lengths, outcomes, 2, owners)[0]
        torch.testing.assert_close(actual, expected)

    def test_empty_attempt_retains_prior_and_reset_clears_memory(self):
        model = InteractionEstimator(8)
        norm = dict(mean=[0.] * len(FEATURE_NAMES), std=[1.] * len(FEATURE_NAMES),
                    outcome_mean=[0.] * 4, outcome_std=[1.] * 4,
                    target_log_mean=np.log([.1, .4]).tolist(), target_log_std=[1., 1.], prior=[.1, .4])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.pt"
            torch.save(dict(architecture=ARCHITECTURE, hidden=8, state_dict=model.state_dict(),
                            normalization=norm, alignment="observable"), path)
            predictor = InteractionPredictor(path)
            result = predictor.update_episode(episode([0.] * 10))
            self.assertEqual(result["mass_kg"], .1)
            self.assertEqual(result["informative_attempts"], 0)
            result = predictor.update_episode(episode([2.] * 10))
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(result["informative_attempts"], 1)
            self.assertGreater(result["friction"], 0)
            predictor.reset()
            self.assertEqual(predictor.predict()["attempts"], 0)
            self.assertEqual(predictor.predict()["mass_kg"], .1)


if __name__ == "__main__":
    unittest.main()
