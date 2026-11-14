"""Verify held-out inference isolation and paired readout uncertainty."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.lift_slide_belief.model import DeltaModel, TrainedModel, TrainConfig
from experiments.lift_slide_belief.slip_study import (
    ARMS, LEARNING, evaluate_readouts, evaluate_same_tasks, exposure, paired_readout_tables, shove_errors,
)


def rows_for(ids):
    cube, task, episode = np.stack(np.meshgrid(ids, np.arange(3), np.arange(10), indexing='ij')).reshape(3, -1)
    state = np.zeros((len(cube), 51), dtype=np.float32)
    state[:, :3] = np.column_stack((cube, task, episode))
    return dict(cube_id=cube, task=task, episode_id=episode, step=np.zeros(len(cube), dtype=int),
                state=state, next_state=state.copy(), action=np.zeros((len(cube), 8)),
                cube_moving=np.ones(len(cube), dtype=bool), true_mass=cube*.02, true_friction=cube*.03)


class BeliefSpy:
    def __init__(self):
        self.cube_ids = np.array([0, 1])
        self.train_z = np.eye(2, 4)
        self.prior_mean = self.train_z.mean(0)
        self.prior_sd = np.ones(4)
        self.validation = {'passed':True}
        self.observations = []

    def fit_many_z(self, groups, steps):
        for group in groups:
            if set(group) != {'state','next_state','action','cube_moving'}:
                raise AssertionError('Physics or bookkeeping columns reached latent inference')
        self.observations.append(groups)
        return np.stack([group['state'][:, :4].mean(0) for group in groups])


class SlipStudyTest(unittest.TestCase):
    def test_exposure_keeps_rotating_losses_separate_from_held_grasps(self):
        rows = dict(cube_id=np.zeros(8), task=np.zeros(8), episode_id=np.repeat(np.arange(4), 2),
                    state=np.zeros((8, 51)), grasp_acquired=np.repeat([False, True, True, True], 2),
                    slip_after_acquisition=np.repeat([False, True, False, False], 2),
                    rotating_loss_after_acquisition=np.repeat([False, False, True, False], 2))
        result = exposure(rows)['lift_outcomes']
        self.assertEqual(result, dict(failed_pickup=1, slip_after_acquisition=1,
                                      rotating_loss_after_acquisition=1, held_after_acquisition=1))
        rows['slip_after_acquisition'][4:6] = True
        with self.assertRaises(ValueError):
            exposure(rows)

    def test_physical_forecast_rmse_reverses_delta_normalization(self):
        network = DeltaModel(51, 8)
        for parameter in network.parameters():
            parameter.data.zero_()
        network.delta_sd.fill_(.1)
        model = TrainedModel(network, np.array([0, 1]), np.zeros((2, 4)), {}, TrainConfig())
        state = np.zeros((2, 51), dtype=np.float32)
        next_state = state.copy()
        next_state[:, 19:21] = [.003, .004]
        next_state[:, 28:31] = [.006, .008, 0.]
        rows = dict(state=state, next_state=next_state, action=np.zeros((2, 8)),
                    cube_moving=np.ones(2, dtype=bool), step=np.arange(2),
                    force_start_step=np.zeros(2), stop_step=np.ones(2))
        result = shove_errors(model, rows, np.arange(2), np.zeros(4))
        self.assertAlmostEqual(result['cube_xy_rmse_mm'], 5, places=5)
        self.assertAlmostEqual(result['cube_velocity_rmse_mm_s'], 10, places=5)

    def test_task_support_does_not_include_prove_shoves(self):
        model = BeliefSpy()
        with patch('experiments.lift_slide_belief.slip_study.shove_errors',
                   return_value=dict(shove_error=0., shove_base43_error=0., moving_steps=5)):
            predictions, forecasts, bank = evaluate_readouts(model, rows_for([0,1]), rows_for([200,201]), seed=0, fit_steps=3)
        self.assertEqual(len(predictions), 6)
        self.assertEqual(len(forecasts), 12)
        self.assertEqual(len(model.observations), 4)
        expected = ({(0, ep) for ep in range(5)}, {(1, ep) for ep in range(5)},
                    {(task, ep) for task in (0,1) for ep in range(5)}, {(2, ep) for ep in range(5,10)})
        for groups, wanted in zip(model.observations, expected):
            for group in groups:
                self.assertEqual({tuple(row[1:3].astype(int)) for row in group['state']}, wanted)
        np.testing.assert_array_equal(bank['donor_cube_id'], [201,200])

    def test_same_task_prove_is_disjoint_and_wrong_donor_uses_same_task(self):
        model = BeliefSpy()
        test = rows_for([200,201])
        test['phase'] = (test['episode_id'] != 5).astype(int)
        test['cube_moving'] = test['episode_id'] % 2 == 0
        with patch('experiments.lift_slide_belief.slip_study.shove_errors',
                   return_value=dict(shove_error=0., shove_base43_error=0., moving_steps=5)):
            _, _, bank = evaluate_readouts(model, rows_for([0,1]), test, seed=0, fit_steps=3)
        calls = []
        def errors(model, rows, indices, z):
            calls.append((rows['state'][indices,:3].copy(), z.copy()))
            return torch.ones((len(indices),51)), torch.full((len(indices),51),1e-6)
        with patch('experiments.lift_slide_belief.slip_study.transition_errors', side_effect=errors):
            records = evaluate_same_tasks(model,test,seed=0,bank=bank)
        self.assertEqual(len(calls),16)
        self.assertEqual(len(records),32)
        for i,(states,z) in enumerate(calls):
            record = records[2*i]
            self.assertEqual(set(states[:,2]),{6,7,8,9})
            self.assertEqual(set(states[:,0]),{record['cube_id']})
            task = int(record['task']=='slow_slide')
            self.assertEqual(set(states[:,1]),{task})
            if record['condition']=='other_cube_same_task':
                self.assertEqual(z[0],record['donor_cube_id'])
                self.assertEqual(z[1],task)
            self.assertEqual(record['scored_steps'],4)
            self.assertEqual(records[2*i+1]['scored_steps'],2)
            self.assertAlmostEqual(record['cube_xyz_rmse_mm'],np.sqrt(3),places=5)

    def test_primary_table_uses_common_passing_seeds_and_paired_cube_draws(self):
        predictions = []
        gates = {arm:{str(seed):{'passed':not (arm=='random_grip' and seed==2)} for seed in range(3)} for arm in ARMS}
        for arm in ARMS:
            for seed in range(3):
                if not gates[arm][str(seed)]['passed']:
                    continue
                for cube in range(30):
                    for condition in LEARNING:
                        truth = .04+.4*cube/29
                        error = .02 if arm=='fixed_grip' else .01
                        if seed==2:
                            error = 100  # This unpaired fixed-arm seed must not enter the primary table.
                        predictions.append(dict(arm=arm,seed=seed,cube_id=cube,condition=condition,
                            true_mass=truth,predicted_mass=truth+error,
                            true_friction=.4+cube/50,predicted_friction=.4+cube/50+error))
        table, comparisons, common = paired_readout_tables(predictions,gates,[0,1,2],bootstrap_draws=500)
        self.assertEqual(common,[0,1])
        for row in table:
            self.assertEqual(row['paired_seed_count'],2)
            self.assertEqual(row['requested_seeds'],3)
            self.assertAlmostEqual(row['mass_mae_g'],20 if row['arm']=='fixed_grip' else 10)
        for row in comparisons:
            expected = 10 if row['quantity']=='mass' else .01
            self.assertAlmostEqual(row['mae_reduction'],expected)
            self.assertAlmostEqual(row['mae_reduction_ci95_low'],expected)
            self.assertAlmostEqual(row['mae_reduction_ci95_high'],expected)
