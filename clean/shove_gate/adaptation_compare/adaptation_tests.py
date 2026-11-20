"""Synthetic contracts; these tests never read experiment traces or checkpoints."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from . import adaptation as ad


def norm():
    return dict(state_mean=[0.] * 15, state_std=[1.] * 15,
                action_mean=[0.] * 3, action_std=[1.] * 3,
                context_mean=[0.] * 2, context_std=[1.] * 2,
                horizontal_velocity_delta_mse_scale=.005)


def episode(cid=180, eid=0, count=4, shift=.3):
    state = np.zeros((count, 15), dtype=np.float32)
    state[:, 9] = 1.
    action = np.zeros((count, 3), dtype=np.float32)
    action[:2, 0] = 5.
    delta = np.zeros_like(state)
    delta[:, 0] = shift
    delta[:2, 9] = .1
    delta[2:, 9] = -.05
    return dict(cube_id=cid, episode_id=eid, state=state, action=action,
                next_state=state + delta, phase=np.array(['push'] * 2 + ['coast'] * (count - 2)))


class ToyLatent(nn.Module):
    variant = 'z4'
    z_dim = 4

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(180, 4)
        nn.init.zeros_(self.embedding.weight)
        self.shared = nn.Parameter(torch.tensor(1.))

    def forward(self, state, action, z=None, **kwargs):
        output = state.new_zeros((len(state), 15))
        output[:, 0] = z[:, 0] * self.shared
        return output


class ToyPlain(nn.Module):
    variant = 'plain'
    z_dim = 0

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(18, 15)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, state, action, **kwargs):
        return self.linear(torch.cat([state, action], 1))


class ZeroModel(nn.Module):
    variant = 'plain'
    z_dim = 0

    def forward(self, state, action, **kwargs):
        return state.new_zeros((len(state), 15))


class AdaptationContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.device = torch.device('cpu')

    def test_full_state_loss_adapts_nonvelocity_coordinate_and_freezes_weights(self):
        model = ToyLatent()
        before = ad.model_state_digest(model)
        fitted = ad.fit_latents(model, [episode()], norm(), self.device, penalty=.0001)
        self.assertGreater(fitted['z_values'][0, 0], .2)
        self.assertEqual(before, ad.model_state_digest(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(len(fitted['loss_log']), 50)
        self.assertLess(fitted['loss_log'][-1]['mean_cube_data_loss'], fitted['loss_log'][0]['mean_cube_data_loss'])

    def test_prior_is_centered_at_training_mean(self):
        a, b = ToyLatent(), ToyLatent()
        with torch.no_grad():
            a.embedding.weight.fill_(.1)
            b.embedding.weight.fill_(.1)
        loose = ad.fit_latents(a, [episode(shift=.8)], norm(), self.device, penalty=.0001)
        strong = ad.fit_latents(b, [episode(shift=.8)], norm(), self.device, penalty=1.)
        np.testing.assert_allclose(strong['initial_mean_z'], [.1] * 4, atol=1e-7)
        self.assertLess(np.linalg.norm(strong['z_values'] - .1), np.linalg.norm(loose['z_values'] - .1))

    def test_vectorized_latents_are_independent(self):
        support = [episode(180, 0, count=4, shift=.3), episode(181, 0, count=7, shift=.8)]
        joint = ad.fit_latents(ToyLatent(), support, norm(), self.device, penalty=.01, updates=10)
        for i, ep in enumerate(support):
            separate = ad.fit_latents(ToyLatent(), [ep], norm(), self.device, penalty=.01, updates=10)
            np.testing.assert_array_equal(joint['z_values'][i], separate['z_values'][0])

    def test_support_episode_weighting_not_length_weighting(self):
        short, long = episode(180, 0, 4, .2), episode(180, 1, 20, .9)
        _, cubes, _, weights = ad._support_pack([short, long], norm(), self.device)
        self.assertEqual(cubes, [180])
        self.assertAlmostEqual(float(weights[:4].sum()), .5)
        self.assertAlmostEqual(float(weights[4:].sum()), .5)

    def test_query_rejected_by_both_fitters(self):
        with self.assertRaisesRegex(RuntimeError, 'Query'):
            ad.fit_latents(ToyLatent(), [episode(eid=5)], norm(), self.device, penalty=.01)
        with self.assertRaisesRegex(RuntimeError, 'Query'):
            ad.finetune(ToyPlain(), [episode(eid=5)], norm(), self.device, lr=1e-4)

    def test_finetuning_updates_all_parameter_tensors_and_keeps_prior_copy(self):
        original = ToyPlain()
        model = copy.deepcopy(original)
        before = ad.model_state_digest(original)
        result = ad.finetune(model, [episode()], norm(), self.device, lr=1e-4, updates=5)
        self.assertEqual(ad.model_state_digest(original), before)
        self.assertNotEqual(ad.model_state_digest(model), before)
        self.assertTrue(result['all_parameter_tensors_receive_gradients'])
        self.assertTrue(all(p.requires_grad and p.grad is not None for p in model.parameters()))

    def test_sequential_weights_carry_and_optimizer_moments_reset(self):
        model = ToyPlain()
        first = ad.finetune(model, [episode(290)], norm(), self.device, lr=1e-4, updates=4)
        saved_after_first = copy.deepcopy(model)
        second = ad.finetune(model, [episode(291, shift=.8)], norm(), self.device, lr=1e-4, updates=4)
        repeated = ad.finetune(saved_after_first, [episode(291, shift=.8)], norm(), self.device, lr=1e-4, updates=4)
        self.assertEqual(first['after_state_sha256'], second['before_state_sha256'])
        self.assertEqual(second['after_state_sha256'], repeated['after_state_sha256'])
        self.assertTrue(second['optimizer_reset'])

    def test_prefix_restarts_do_not_reuse_previous_z(self):
        model = ToyLatent()
        a = ad.fit_latents(model, [episode()], norm(), self.device, penalty=.01, updates=5)
        ad.fit_latents(model, [episode(shift=.9)], norm(), self.device, penalty=.01, updates=8)
        b = ad.fit_latents(model, [episode()], norm(), self.device, penalty=.01, updates=5)
        np.testing.assert_array_equal(a['z_values'], b['z_values'])

    def test_zero_predictor_has_unit_component_scores_on_fitting_data(self):
        episodes = [episode(c, e) for c in range(180) for e in range(8)]
        normalizer = norm()
        # Use the exact float32 observation delta represented in the source data.
        delta = episodes[0]['next_state'].astype(float) - episodes[0]['state'].astype(float)
        normalizer['horizontal_velocity_delta_mse_scale'] = float((delta[:2, 9:11] ** 2).mean())
        scales = ad.derive_scales(episodes, normalizer)
        result = ad.score(ZeroModel(), episodes, normalizer, scales, self.device)
        for name in ad.METRICS:
            self.assertAlmostEqual(result['scores'][name], 1., places=12)
        self.assertEqual(result['aggregates']['composite']['cubes'], 180)
        self.assertEqual(result['aggregates']['composite']['episodes'], 1440)

    def test_missing_component_and_property_leakage_rejected(self):
        ep = episode()
        scales = dict(push_velocity_energy=1., coast_velocity_energy=1., full_state_energy=1.)
        with self.assertRaisesRegex(RuntimeError, 'Physical labels'):
            ad.score(ZeroModel(), [ep], norm(), scales, self.device, properties={180: [1., .1]})
        ep['phase'][:] = 'push'
        with self.assertRaisesRegex(RuntimeError, 'coverage'):
            ad.score(ZeroModel(), [ep], norm(), scales, self.device)

    def test_retired_test_rejected_before_reading_trace(self):
        for cid in [200, 259, 260, 289]:
            with self.assertRaisesRegex(RuntimeError, 'retired'):
                ad.load_episodes([cid], [0], root=Path('/nonexistent'))

    def test_fresh_trace_requires_begun_ledger_and_explicit_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data/episodes').mkdir(parents=True)
            (root / 'results').mkdir()
            ep = episode(290)
            np.savez(root / 'data/episodes/cube_290_episode_00.npz',
                     **{k: v for k, v in ep.items() if k not in ['cube_id', 'episode_id']},
                     cube_id=np.full(4, 290), episode_id=np.zeros(4, dtype=int))
            with self.assertRaisesRegex(RuntimeError, 'Fresh test motion'):
                ad.load_episodes([290], [0], root=root)
            with self.assertRaisesRegex(RuntimeError, 'ledger'):
                ad.load_episodes([290], [0], root=root, allow_test=True)
            (root / 'results/final_evaluation.json').write_text(json.dumps({'status': 'begun'}))
            loaded = ad.load_episodes([290], [0], root=root, allow_test=True)
            self.assertEqual(loaded[0]['cube_id'], 290)

    def test_nonfinite_observation_stops_before_adaptation(self):
        ep = episode()
        ep['next_state'][0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            ad.fit_latents(ToyLatent(), [ep], norm(), self.device, penalty=.01)

    def test_selection_complete_grid_ties_and_validation_boundary(self):
        def query():
            return dict(cube_ids=ad.VALIDATION_IDS, episode_ids=list(range(5, 10)),
                        scores={m: 1. for m in ad.METRICS},
                        aggregates={m: dict(cubes=20, episodes=100) for m in ad.METRICS})
        a = [dict(z_dim=d, penalty=p, seed=s, query=query()) for d in ad.Z_DIMS for p in ad.PENALTIES for s in ad.SEEDS]
        b = [dict(learning_rate=lr, seed=s, query=query()) for lr in ad.FINETUNE_LRS for s in ad.SEEDS]
        choice = ad.select_configuration(a, b)
        self.assertEqual(choice['A']['z_dim'], 4)
        self.assertEqual(choice['A']['penalty'], 1.)
        self.assertEqual(choice['B']['learning_rate'], 1e-5)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            ad.select_configuration(a[:-1], b)
        a[0]['query']['cube_ids'] = list(range(290, 320))
        with self.assertRaisesRegex(RuntimeError, 'non-validation'):
            ad.select_configuration(a, b)


if __name__ == '__main__':
    unittest.main()
