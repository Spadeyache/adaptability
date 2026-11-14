"""Support/prove isolation, property blinding, equal budgets and stored-belief retrieval."""
from collections import Counter
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.lift_slide_belief.evaluation import evaluate_seed, retrieve, LEARNING_COUNTS


def _rollouts(ids):
    grid = np.meshgrid(ids, [0, 1, 2], np.arange(10), [0, 1], indexing="ij")
    cube, task, episode, step = np.stack(grid).reshape(4, -1)
    # Sentinels in the observation identify every exact transition seen by a spy.
    state = np.column_stack((cube, task, episode, step)).astype(np.float32)
    order = np.random.default_rng(32).permutation(len(cube))
    rows = dict(cube_id=cube, task=task, episode_id=episode, step=step, state=state,
                action=step[:, None], next_state=state + 1., cube_moving=np.ones(len(step), dtype=bool),
                true_mass=cube * .02, true_friction=cube * .04,
                stop_step=np.ones(len(step), dtype=int), observed_stopped=task == 2,
                force_start_step=np.ones(len(step), dtype=int))
    return {key: values[order] for key, values in rows.items()}


class _Spy:
    def __init__(self):
        self.cube_ids = np.arange(5)
        self.train_z = np.eye(5, 4, dtype=np.float32)
        self.prior_mean = self.train_z.mean(0)
        self.prior_sd = self.train_z.std(0)
        self.validation = {"passed": True}
        self.fits = []
        self.proves = []
        self.endpoints = []

    @staticmethod
    def observe(rows, endpoint=False):
        allowed = {"state", "action", "next_state", "cube_moving"}
        if endpoint:
            allowed |= {"step", "force_start_step", "stop_step", "observed_stopped"}
        if set(rows) != allowed:
            raise AssertionError("An unexpected label or bookkeeping column reached the model")
        return {tuple(row) for row in rows["state"].astype(int)}

    def fit_z(self, rows, initial=None, steps=100):
        self.fits.append((self.observe(rows), initial, steps))
        return rows["state"].mean(axis=0)

    def fit_many_z(self, groups, steps=100, initials=None):
        starts = np.tile(self.prior_mean, (len(groups), 1)) if initials is None else initials
        return np.stack([self.fit_z(group, initial=initial, steps=steps) for group, initial in zip(groups, starts)])

    def moving_error(self, rows, z):
        self.proves.append((self.observe(rows), z.copy()))
        return float(np.mean((rows["state"][rows["cube_moving"]] - z) ** 2))

    def stopping_errors(self, episodes, z):
        results = []
        for rows in episodes:
            self.endpoints.append((self.observe(rows, endpoint=True), z.copy()))
            np.testing.assert_array_equal(rows["step"], [0, 1])
            if rows["observed_stopped"].all():
                result = dict(endpoint_error_m=float(np.linalg.norm(rows["next_state"][-1, :2] - z[:2])),
                              endpoint_status="finite", predicted_final_speed_m_s=2.)
            else:
                result = dict(endpoint_error_m=None, endpoint_status="observed_not_stopped",
                              predicted_final_speed_m_s=None)
            results.append(result)
        return results


class BeliefEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.dest = Path(self.root.name)
        self.ids = (20, 21, 22)
        self.model = _Spy()
        self.train = _rollouts(self.model.cube_ids)
        self.test = _rollouts(self.ids)

    def evaluate(self):
        return evaluate_seed(self.model, self.train, self.test, seed=3, output_dir=self.dest, fit_steps=2)

    def test_shared_prove_rows_distinct_support_fair_budgets_and_retrieval(self):
        table_before = self.model.train_z.copy()
        rows = self.evaluate()
        np.testing.assert_array_equal(self.model.train_z, table_before)
        keys = {(r["cube_id"], r["condition"], r["learning_episodes"]) for r in rows}
        self.assertEqual(len(keys), len(rows))
        self.assertEqual(len(keys), 23 * len(self.ids))
        self.assertEqual(len(self.model.fits), 21 * len(self.ids))

        def sentinels(cube, task, episodes):
            return {(cube, task, ep, step) for ep in episodes for step in (0, 1)}

        with np.load(self.dest / "beliefs.npz") as saved:
            beliefs = {condition: dict(zip(saved["cube_id"], saved["z_" + condition]))
                       for condition in LEARNING_COUNTS}
            saved_ceilings = dict(zip(saved["cube_id"], saved["z_shove_ceiling"]))
            saved_donors = dict(zip(saved["cube_id"], saved["donor_cube_id"]))
        expected_supports = []
        for cube in self.ids:
            for condition, count in LEARNING_COUNTS.items():
                np.testing.assert_array_equal(beliefs[condition][cube][0], self.model.prior_mean)
                for n in range(1, count + 1):
                    if condition == "both":
                        expected = sentinels(cube, 0, range((n + 1) // 2)) | sentinels(cube, 1, range(n // 2))
                    else:
                        expected = sentinels(cube, int(condition == "slides_only"), range(n))
                    expected_supports.append(frozenset(expected))
                    np.testing.assert_allclose(beliefs[condition][cube][n], np.mean(list(expected), axis=0))
            expected_supports.append(frozenset(sentinels(cube, 2, range(5, 10))))
            np.testing.assert_array_equal(saved_ceilings[cube], [cube, 2, 7, .5])
        self.assertEqual(Counter(expected_supports), Counter(frozenset(seen) for seen, _, _ in self.model.fits))
        for _, initial, steps in self.model.fits:
            np.testing.assert_array_equal(initial, self.model.prior_mean)
            self.assertEqual(steps, 2)

        donors = {r["cube_id"]: r["donor_cube_id"] for r in rows if r["condition"] == "other_both"}
        self.assertEqual(set(donors.values()), set(self.ids))
        self.assertTrue(all(c != d for c, d in donors.items()))
        self.assertEqual(donors, saved_donors)
        for index, (row, (seen, z)) in enumerate(zip(rows, self.model.proves)):
            cube, condition = row["cube_id"], row["condition"]
            self.assertEqual(seen, {(cube, 2, ep, 1) for ep in range(5)})
            self.assertEqual(row["moving_steps"], 5)
            self.assertEqual(row["valid_endpoints"], 5)
            # Finite predicted motion remains a valid endpoint and gets a diagnostic.
            self.assertEqual(row["predicted_moving_endpoints"], 5)
            for ep, (endpoint_seen, endpoint_z) in enumerate(self.model.endpoints[5*index:5*(index+1)]):
                self.assertEqual(endpoint_seen, sentinels(cube, 2, [ep]))
                np.testing.assert_array_equal(z, endpoint_z)
            if condition == "other_both":
                np.testing.assert_array_equal(z, beliefs["both"][donors[cube]][-1])
            elif condition in LEARNING_COUNTS:
                np.testing.assert_array_equal(z, beliefs[condition][cube][row["learning_episodes"]])
            elif condition == "train_mean":
                np.testing.assert_array_equal(z, self.model.prior_mean)
                self.assertEqual(row["adaptation_steps"], 0)
            else:
                np.testing.assert_array_equal(z, [cube, 2, 7, .5])

        prove_mask = (self.test["task"] == 2) & (self.test["episode_id"] < 5)
        only_prove = {k: v[prove_mask] for k, v in self.test.items()}
        fresh_model = _Spy()
        path = self.dest / "beliefs.npz"
        for condition, n in LEARNING_COUNTS.items():
            scores = retrieve(fresh_model, only_prove, path, condition=condition)
            for row in rows:
                if row["condition"] == condition and row["learning_episodes"] == n:
                    self.assertEqual(scores[row["cube_id"]]["shove_error"], row["shove_error"])
                    self.assertEqual(scores[row["cube_id"]]["endpoint_error_m"], row["endpoint_error_m"])
                    self.assertEqual(row["retrieved_shove_error"], row["shove_error"])
                    self.assertEqual(row["retrieved_endpoint_error_m"], row["endpoint_error_m"])
        self.assertEqual(retrieve(fresh_model, only_prove, path, cube_ids=21), {21: scores[21]})
        subset = {21: scores[21], 22: scores[22]}
        self.assertEqual(retrieve(fresh_model, only_prove, path, cube_ids=[21, 22]), subset)
        subset_rows = {k: v[only_prove["cube_id"] != 20] for k, v in only_prove.items()}
        self.assertEqual(retrieve(fresh_model, subset_rows, path), subset)
        with self.assertRaisesRegex(KeyError, "No stored belief for cube 99"):
            retrieve(fresh_model, only_prove, path, cube_ids=99)
        self.assertFalse(fresh_model.fits)

    def test_stationary_cube_keeps_endpoint_score_without_moving_error(self):
        cube = self.test["cube_id"] == self.ids[0]
        self.test["cube_moving"][cube & (self.test["task"] == 2) & (self.test["step"] == 1)] = False
        rows = self.evaluate()
        for row in rows:
            if row["cube_id"] == self.ids[0]:
                self.assertIsNone(row["shove_error"])
                self.assertEqual(row["moving_steps"], 0)
                self.assertIsNotNone(row["endpoint_error_m"])

    def test_invalid_endpoint_remains_visible_without_dropping_valid_one_step_score(self):
        failed = ((self.test["cube_id"] == 20) & (self.test["task"] == 2)
                  & (self.test["episode_id"] == 3))
        self.test["observed_stopped"][failed] = False
        rows = self.evaluate()
        for row in rows:
            if row["cube_id"] == 20:
                self.assertIsNone(row["endpoint_error_m"])
                self.assertEqual(row["valid_endpoints"], 4)
                self.assertEqual(row["unstopped_endpoints"], 1)
                self.assertEqual(row["endpoint_status"], "observed_not_stopped")
                self.assertTrue(np.isfinite(row["shove_error"]))

    def test_endpoint_plot_omits_incomplete_conditions(self):
        from experiments.lift_slide_belief.plotting import _seed_summary

        rows = [dict(seed=str(seed), condition="both", learning_episodes="10", endpoint_error_m=str(value))
                for seed, value in [(0, 1), (0, 3), (1, 5), (1, 7)]]
        mean, sem, valid, total = _seed_summary(rows, "both", 10, "endpoint_error_m")
        self.assertEqual((mean, sem, valid, total), (4., 2., 4, 4))
        rows[-1]["endpoint_error_m"] = ""
        mean, sem, valid, total = _seed_summary(rows, "both", 10, "endpoint_error_m")
        self.assertTrue(np.isnan(mean) and np.isnan(sem))
        self.assertEqual((valid, total), (3, 4))

    def test_moving_score_uses_identical_active_cubes_across_conditions(self):
        from experiments.lift_slide_belief.plotting import _seed_summary

        rows = [dict(seed="0", cube_id=str(cube), condition=condition, learning_episodes=str(n),
                     shove_error="" if cube == 2 else "3")
                for condition, n in [("train_mean", 0), ("both", 10)] for cube in (1, 2)]
        mean, _, active, total = _seed_summary(rows, "both", 10, "shove_error")
        self.assertEqual((mean, active, total), (3., 1, 2))
        rows[-1]["shove_error"] = "7"
        with self.assertRaisesRegex(ValueError, "availability differs across conditions"):
            _seed_summary(rows, "both", 10, "shove_error")


if __name__ == "__main__":
    unittest.main()
