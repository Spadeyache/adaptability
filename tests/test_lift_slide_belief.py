"""Episode isolation, property blinding, frozen test adaptation and retrieval."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.lift_slide_belief.evaluation import evaluate_seed, retrieve


def _rollouts(ids):
    grid = np.meshgrid(ids, [0, 1], np.arange(10), [0, 1], indexing="ij")
    cube, task, episode, step = np.stack(grid).reshape(4, -1)
    # A sentinel in the observation lets a spy identify every exact transition.
    state = np.column_stack((cube, task, episode, step)).astype(np.float32)
    order = np.random.default_rng(32).permutation(len(cube))
    rows = dict(cube_id=cube, task=task, episode_id=episode, step=step, state=state,
                action=step[:, None], next_state=state + 1., cube_moving=step == 1,
                true_mass=cube * .02, true_friction=cube * .04)
    return {key: values[order] for key, values in rows.items()}


class _Spy:
    def __init__(self):
        self.cube_ids = np.arange(5)
        self.train_z = np.eye(5, 4, dtype=np.float32)
        self.validation = {"passed": True}
        self.fits = []
        self.proves = []

    @staticmethod
    def observe(rows):
        if set(rows) != {"state", "action", "next_state", "cube_moving"}:
            raise AssertionError("A label or bookkeeping column reached the model")
        return {tuple(row) for row in rows["state"].astype(int)}

    def fit_z(self, rows, initial=None, steps=100):
        self.fits.append((self.observe(rows), initial, steps))
        return rows["state"].mean(axis=0)

    def moving_error(self, rows, z):
        self.proves.append((self.observe(rows), z.copy()))
        return float(np.mean((rows["state"][rows["cube_moving"]] - z) ** 2))


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

    def test_shared_prove_rows_distinct_support_and_disk_only_retrieval(self):
        table_before = self.model.train_z.copy()
        rows = self.evaluate()
        np.testing.assert_array_equal(self.model.train_z, table_before)
        keys = {(r["cube_id"], r["condition"], r["number_of_lifts"]) for r in rows}
        self.assertEqual(len(keys), len(rows))
        self.assertEqual(len(keys), 8 * len(self.ids))
        self.assertEqual(len(self.model.fits), 6 * len(self.ids))

        def sentinels(cube, task, episodes):
            return {(cube, task, ep, step) for ep in episodes for step in (0, 1)}

        with np.load(self.dest / "beliefs.npz") as saved:
            beliefs = dict(zip(saved["cube_id"], saved["z_after_lift"]))
        for i, cube in enumerate(self.ids):
            for j in range(5):
                seen, initial, steps = self.model.fits[5*i+j]
                self.assertEqual(seen, sentinels(cube, 0, range(j+1)))
                self.assertEqual(steps, 2)
                if j:
                    np.testing.assert_array_equal(initial, beliefs[cube][j-1])
                else:
                    self.assertIsNone(initial)
                np.testing.assert_array_equal(beliefs[cube][j], [cube, 0, j / 2, .5])
            seen, initial, steps = self.model.fits[5*len(self.ids)+i]
            self.assertEqual(seen, sentinels(cube, 1, range(5, 10)))
            self.assertIsNone(initial)
            self.assertEqual(steps, 10)

        donors = {r["cube_id"]: r["donor_cube_id"] for r in rows if r["condition"] == "other_lifts"}
        self.assertEqual(set(donors.values()), set(self.ids))
        self.assertTrue(all(c != d for c, d in donors.items()))
        for row, (seen, z) in zip(rows, self.model.proves):
            cube = row["cube_id"]
            self.assertEqual(seen, sentinels(cube, 1, range(5)))
            self.assertEqual(row["moving_steps"], 5)
            if row["condition"] == "other_lifts":
                np.testing.assert_array_equal(z, beliefs[donors[cube]][-1])
            elif row["condition"] == "own_lifts":
                np.testing.assert_array_equal(z, beliefs[cube][row["number_of_lifts"] - 1])
            elif row["condition"] == "zero":
                np.testing.assert_array_equal(z, np.zeros(4))
            else:
                np.testing.assert_array_equal(z, [cube, 1, 7, .5])

        prove_mask = (self.test["task"] == 1) & (self.test["episode_id"] < 5)
        only_prove = {k: v[prove_mask] for k, v in self.test.items()}
        fresh_model = _Spy()
        scores = retrieve(fresh_model, only_prove, self.dest / "beliefs.npz")
        self.assertFalse(fresh_model.fits)
        self.assertEqual(len(fresh_model.proves), len(self.ids))
        for row in rows:
            if row["condition"] == "own_lifts" and row["number_of_lifts"] == 5:
                self.assertEqual(scores[row["cube_id"]], row["slide_error"])
                self.assertEqual(scores[row["cube_id"]], row["retrieved_slide_error"])
        path = self.dest / "beliefs.npz"
        self.assertEqual(retrieve(fresh_model, only_prove, path, cube_ids=21), {21: scores[21]})
        subset = {21: scores[21], 22: scores[22]}
        self.assertEqual(retrieve(fresh_model, only_prove, path, cube_ids=[21, 22]), subset)
        subset_rows = {k: v[only_prove["cube_id"] != 20] for k, v in only_prove.items()}
        self.assertEqual(retrieve(fresh_model, subset_rows, path), subset)
        with self.assertRaisesRegex(KeyError, "No stored belief for cube 99"):
            retrieve(fresh_model, only_prove, path, cube_ids=99)
        self.assertFalse(fresh_model.fits)

    def test_rejects_no_moving_prove_steps(self):
        cube = self.test["cube_id"] == self.ids[0]
        self.test["cube_moving"][cube & (self.test["task"] == 1)] = False
        with self.assertRaisesRegex(ValueError, "No moving steps"):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()
