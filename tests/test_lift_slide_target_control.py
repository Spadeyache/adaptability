"""Paired physical target execution, cache reuse, hidden labels and final-row updates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.lift_slide_belief.evaluation import write_results
from experiments.lift_slide_belief import target_control as control
from experiments.lift_slide_belief.scripted import shove_actions


class TargetControlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.study = self.root / "data", self.root / "study"
        (self.data / "test").mkdir(parents=True)
        self.study.mkdir()
        self.cfg = dict(seed=42, train_cubes=2, test_cubes=2, task_steps=[3, 3, 3],
                        mass_range_kg=[.04, .2], friction_range=[.1, 1.],
                        shove_force_cap_n=4., shove_duration_s=.1, control_hz=10,
                        retract_steps=1, retract_xy_action=.1, retract_z_action=.2)
        (self.data / "dataset.json").write_text(json.dumps(self.cfg))
        self.starts = {}
        for cube in (2, 3):
            episodes = []
            for ep in range(5):
                rng = np.random.default_rng([42, cube, 1, 2, ep])
                reset_seed = int(rng.integers(2**31))
                angle = rng.uniform(-np.pi, np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
                state = np.zeros((3, 43), dtype=np.float32)
                state[:, 19:21] = [cube * .02, ep * .01]
                self.starts[reset_seed] = state[1].copy()
                episodes.append(dict(task=np.full(3, 2), episode_id=np.full(3, ep), step=np.arange(3),
                                     force_start_step=np.ones(3), stop_step=np.full(3, 2),
                                     reset_seed=np.full(3, reset_seed), motion_direction=np.tile(direction, (3, 1)),
                                     state=state))
            np.savez(self.data / "test" / f"cube_{cube:04}.npz",
                     **{key: np.concatenate([episode[key] for episode in episodes]) for key in episodes[0]})
        results = []
        for seed in (0, 1):
            folder = self.study / f"seed_{seed}"
            folder.mkdir()
            (folder / "model.pt").write_bytes(b"mock model checkpoint")
            np.savez(folder / "beliefs.npz", cube_id=[2, 3], donor_cube_id=[3, 2],
                     prior_mean=np.zeros(4), z_lifts_only=np.ones((2, 6, 4)),
                     z_slides_only=np.full((2, 6, 4), 2.),
                     z_both=np.full((2, 11, 4), 3.), z_shove_ceiling=np.full((2, 4), 4.))
            for cube in (2, 3):
                for condition, count in control.FINAL_COUNTS.items():
                    results.append(dict(seed=seed, cube_id=cube, condition=condition, learning_episodes=count))
                results.append(dict(seed=seed, cube_id=cube, condition="both", learning_episodes=5))
        write_results(results, self.study / "results.csv")

    def choose(self, model, initial_state, direction, distance, candidates, actions, *, z, **kwargs):
        self.assertEqual(model, "mock")
        self.assertEqual(initial_state.shape, (43,))
        self.assertEqual(actions.shape, (13, 2, 8))
        self.assertIn(distance, control.TARGET_DISTANCES_M)
        fallback = int(z[0]) == 3
        index = 6 if fallback else (int(z[0]) % 3) * 6
        target = initial_state[19:21] + distance * direction
        return dict(
            selected_candidate_index=index, selected_force_n=float(candidates[index]),
            planner_status="all_nonfinite_midpoint_fallback" if fallback else "finite_prediction",
            target_xy=target.tolist(), predicted_endpoint_xy=None if fallback else target.tolist(),
            predicted_target_error_m=None if fallback else 0.,
            predicted_final_speed_m_s=None if fallback else 0., valid_candidates=0 if fallback else 13,
            candidate_target_errors_m=[None if fallback else .01] * 13,
            candidate_divergence_steps=[1 if fallback else None] * 13,
        )

    def simulate(self, mass, friction, reset_seed, direction, force, cfg):
        self.assertGreater(mass, 0)
        self.assertGreater(friction, 0)
        state = np.tile(self.starts[reset_seed], (3, 1))
        next_state = state.copy()
        next_state[-1, 19:21] += force * .003 * direction
        commands = np.vstack((np.zeros((1, 8)), shove_actions(direction, force, 2, cfg)))
        return dict(state=state, action=commands, next_state=next_state,
                    force_start_step=np.ones(3, dtype=int), stop_step=np.full(3, 2),
                    observed_stopped=np.full(3, force != 1.))

    def test_control_pairing_fallback_scoring_and_cached_resume(self):
        with patch.object(control, "load_world_model", return_value="mock"), \
             patch.object(control, "choose_force", side_effect=self.choose) as planner, \
             patch.object(control, "simulate_shove", side_effect=self.simulate) as simulator:
            rows = control.evaluate_targets(self.data, self.study, workers=1)
            self.assertEqual(planner.call_count, 120)
            self.assertEqual(simulator.call_count, 30)
        with np.load(self.study / "target_control" / "trials.npz") as saved:
            trials = {key: saved[key] for key in saved.files}
        for cube in (2, 3):
            for ep in range(5):
                selected = (trials["cube_id"] == cube) & (trials["episode_id"] == ep)
                # Identical ready state, target and candidate force grid across every seed/condition.
                np.testing.assert_allclose(trials["target_xy"][selected],
                                           np.tile(trials["target_xy"][selected][0], (12, 1)))
        for row in rows:
            if row["condition"] == "both" and int(row["learning_episodes"]) == 5:
                self.assertEqual(row["target_landing_error_m"], "")
                continue
            selected = ((trials["cube_id"] == int(row["cube_id"])) & (trials["seed"] == int(row["seed"]))
                        & (trials["condition"] == row["condition"]))
            np.testing.assert_allclose(row["target_landing_error_m"], trials["actual_target_error_m"][selected].mean())
            expected_success = (trials["actual_stopped"][selected]
                                & (trials["actual_target_error_m"][selected] <= control.TARGET_TOLERANCE_M))
            np.testing.assert_allclose(row["target_success_rate"], expected_success.mean())
            self.assertEqual(row["target_planner_failures"], 5 if row["condition"] in ("both", "other_both") else 0)
        self.assertTrue((trials["actual_target_error_m"][trials["planner_status"] != "finite_prediction"] > 0).all())
        # Completed caches replay without loading a model or creating new simulator trajectories.
        with patch.object(control, "load_world_model", side_effect=AssertionError("unexpected model load")), \
             patch.object(control, "simulate_shove", side_effect=AssertionError("unexpected simulation")):
            repeated = control.evaluate_targets(self.data, self.study, workers=1)
        self.assertEqual(rows, repeated)

    def test_replay_ready_state_mismatch_rejected_before_cache_write(self):
        episodes = control._episodes(self.data, self.cfg)
        episode = episodes[2, 0]
        cache = self.root / "cache"
        cache.mkdir()

        def changed(*args):
            rows = self.simulate(*args)
            rows["state"][1, 0] += .001
            return rows

        with patch.object(control, "simulate_shove", side_effect=changed):
            with self.assertRaisesRegex(AssertionError, "Replay ready state changed"):
                control._simulate_variant((2, 0, 6, episode), cfg=self.cfg, cache_dir=cache)
        self.assertFalse(list(cache.iterdir()))


if __name__ == "__main__":
    unittest.main()
