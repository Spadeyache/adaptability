"""Boundaries and matched-budget accounting for the real retry experiment."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.interaction_adaptation.evaluate import (
    estimator_inputs, evaluation_source_paths, make_task, supplied_observation, summarize,
)


class InteractionAdaptationTests(unittest.TestCase):
    def test_task_factory_preserves_vertical_protocol_and_declares_native_dependencies(self):
        settings=dict(grip_force_limits_n=[.25,40],goal_clearance_m=[.1,.3],grip_force_cost=.1)
        with patch("experiments.interaction_adaptation.evaluate.WeightedCollectionEnv") as vertical:
            make_task("vertical", {}, reported_mass_kg=.1, settings=settings)
            self.assertEqual(vertical.call_args.kwargs["goal_clearance_m"], .2)
        paths={p.name for p in evaluation_source_paths("native_carry")}
        self.assertIn("checkpoint_probe.py", paths)
        self.assertIn("checkpoint_metadata.py", paths)
        self.assertNotIn("checkpoint_probe.py", {p.name for p in evaluation_source_paths("vertical")})
        with self.assertRaises(ValueError):
            make_task("unknown", {}, reported_mass_kg=.1, settings=settings)

    def test_estimator_boundary_drops_targets_and_simulator_contact_masks(self):
        source = {"input__finger_width_m": np.asarray([.03, .03]),
                  "meta__state_time_s": np.asarray([0., .05]),
                  "meta__control_frequency_hz": np.asarray(20),
                  "label__mass_kg": np.asarray(.5),
                  "label__condition_id": np.asarray("heldout"),
                  "sim__is_grasped": np.asarray([False, True]),
                  "sim__policy_observation": np.zeros((2, 45))}
        result = estimator_inputs(source)
        self.assertEqual(set(result), {"input__finger_width_m", "meta__state_time_s",
                                       "meta__control_frequency_hz"})

    def test_supplied_friction_does_not_modify_existing_observations(self):
        observation = torch.arange(45, dtype=torch.float32).reshape(1, -1)
        saved = observation.clone()
        conditioned = supplied_observation(observation, .7)
        torch.testing.assert_close(observation, saved)
        torch.testing.assert_close(conditioned[:, :45], saved)
        self.assertEqual(conditioned.shape, (1, 46))
        self.assertAlmostEqual(conditioned[0, 45].item(), .7, places=6)
        for invalid in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                supplied_observation(observation, invalid)

    def test_success_is_counted_per_sequence_after_any_attempt(self):
        rows = [dict(method=method, sequence_key=str(sequence), success=success)
                for method in ("fixed", "adaptive", "true")
                for sequence in range(4)
                for success in ((False, True) if method != "fixed" else (False, False, False))]
        report = summarize(rows, budget=3)
        self.assertEqual(report["methods"]["adaptive"]["sequences"], 4)
        self.assertEqual(report["methods"]["adaptive"]["successes_within_budget"], 4)
        self.assertEqual(report["methods"]["fixed"]["successes_within_budget"], 0)
        difference = report["methods"]["adaptive_vs_fixed"]
        self.assertEqual(difference["success_rate_difference"], 1.)
        self.assertEqual(difference["bootstrap_95_interval"], [1., 1.])
        self.assertTrue(difference["supported_on_this_sample"])

    def test_no_success_improvement_is_not_reported_as_support(self):
        rows = [dict(method=method, sequence_key=str(sequence), success=False)
                for method in ("fixed", "adaptive", "true") for sequence in range(4)]
        difference = summarize(rows, 3)["methods"]["adaptive_vs_fixed"]
        self.assertEqual(difference["success_rate_difference"], 0.)
        self.assertFalse(difference["supported_on_this_sample"])


if __name__ == "__main__":
    unittest.main()
