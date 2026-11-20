"""Synthetic scientific guards for final-only property diagnostics."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from clean.recovery import align_latents, episode_zero_sequences, fit_ridge_decoder, ridge_predict


class RecoveryTests(unittest.TestCase):
    def test_first_episode_probe_ignores_later_fitting_motion(self):
        pack = SimpleNamespace(
            cube_ids=torch.tensor([200, 200, 200, 200, 201, 201, 201, 201]),
            episode_ids=torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]),
            step=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
            x=torch.arange(8, dtype=torch.float32)[:, None],
            a=torch.zeros(8, 1),
        )
        ids, sequence, lengths = episode_zero_sequences(pack)
        self.assertEqual(ids.tolist(), [200, 201])
        self.assertEqual(lengths.tolist(), [2, 2])
        pack.x[pack.episode_ids == 1] = 1e9
        _, again, _ = episode_zero_sequences(pack)
        self.assertTrue(torch.equal(sequence, again))
        pack.episode_ids[2] = 5
        with self.assertRaisesRegex(ValueError, "first-five-lift"):
            episode_zero_sequences(pack)

    def test_latent_rows_align_by_cube_identity_and_reject_mismatches(self):
        fitted = {"cube_ids": torch.tensor([201, 200]), "z": torch.tensor([[3., 4.], [1., 2.]])}
        np.testing.assert_array_equal(align_latents(torch.tensor([200, 201]), fitted), [[1, 2], [3, 4]])
        fitted["cube_ids"] = torch.tensor([200, 200])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            align_latents(torch.tensor([200, 201]), fitted)
        fitted["cube_ids"] = torch.tensor([0, 1])
        with self.assertRaisesRegex(ValueError, "match exactly"):
            align_latents(torch.tensor([200, 201]), fitted)

    def test_ridge_keeps_intercept_unpenalized_and_stores_no_labels(self):
        z = np.column_stack((np.linspace(-1, 1, 20), np.linspace(-1, 1, 20) ** 2))
        labels = np.column_stack((2 * z[:, 0] + 8, 3 * z[:, 1] - 4))
        decoder = fit_ridge_decoder(z, labels)
        self.assertFalse(decoder["intercept_penalized"])
        self.assertNotIn("labels", decoder)
        np.testing.assert_allclose(ridge_predict(z, decoder), labels, atol=0.001)
        constant = fit_ridge_decoder(np.zeros((20, 2)), np.full((20, 2), 7.0))
        np.testing.assert_allclose(ridge_predict(np.zeros((3, 2)), constant), np.full((3, 2), 7.0))


if __name__ == "__main__":
    unittest.main()
