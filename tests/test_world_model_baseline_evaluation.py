"""Independent metric arithmetic, causal history, and autonomous rollout checks."""
import json
import unittest

import numpy as np
import torch
from torch import nn

from experiments.world_model_baselines.evaluate import evaluate


def episodes(specs):
    """specs contain cube, task, episode, and physical increments of coordinate zero."""
    parts = []
    for cube, task, episode, increments in specs:
        increments = np.asarray(increments, dtype=np.float32)
        n = len(increments)
        path = np.r_[0, np.cumsum(increments)]
        state = np.zeros((n+1, 51), dtype=np.float32)
        state[:, 0] = path
        action = np.zeros((n, 8), dtype=np.float32)
        action[:, 0] = np.arange(1, n+1)
        parts.append(dict(state=state[:-1], next_state=state[1:], action=action,
            cube_id=np.full(n, cube), task=np.full(n, task), episode_id=np.full(n, episode),
            step=np.arange(n), cube_moving=np.ones(n, dtype=bool)))
    return {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}


class SpyModel(nn.Module):
    def __init__(self, kind='B0', k=0):
        super().__init__()
        self.kind, self.history_k = kind, k
        for name, n, value in (('input_mean',59,0),('input_sd',59,1),
                              ('delta_mean',51,0),('delta_sd',51,1),
                              ('oracle_mean',2,0),('oracle_sd',2,1)):
            self.register_buffer(name, torch.full((n,), float(value)))
        self.one_calls, self.roll_calls = [], []
        self.nonfinite_step = None

    def forward(self, inputs, history=None, lengths=None, oracle=None):
        self.one_calls.append(tuple(None if x is None else x.clone() for x in (inputs,history,lengths,oracle)))
        return torch.zeros((len(inputs),51), device=inputs.device)

    def advance(self, state, action, history=None, lengths=None, oracle=None):
        self.roll_calls.append(tuple(None if x is None else x.clone() for x in (state,action,history,lengths,oracle)))
        predicted = state.clone()
        predicted[:,0] += action[:,0]
        if self.nonfinite_step == len(self.roll_calls):
            predicted[:,0] = float('nan')
        return predicted


class BaselineEvaluationTest(unittest.TestCase):
    def test_pooled_rows_and_feature_centered_r2_differ_from_cube_average(self):
        rows = episodes([(200,0,5,[1,1]),(201,0,5,[3,3,3])])
        result = evaluate(SpyModel(),rows,batch_size=2)
        score = result['overall']
        self.assertAlmostEqual(score['one_step_mse'],29/(5*51))
        self.assertAlmostEqual(score['one_step_sst'],4.8)
        self.assertAlmostEqual(score['one_step_r2'],1-29/4.8)
        self.assertAlmostEqual(result['per_cube']['200']['overall']['one_step_mse'],1/51)
        self.assertAlmostEqual(result['per_cube']['201']['overall']['one_step_mse'],9/51)
        self.assertEqual(score['rollout_starts'],0)
        self.assertIsNone(score['rollout_10_mse'])
        # All constant-target dimensions contribute errors; their SST is zero.
        constant = evaluate(SpyModel(),episodes([(200,0,5,[1,1])]))['overall']
        self.assertIsNone(constant['one_step_r2'])
        self.assertAlmostEqual(constant['one_step_mse'],1/51)

    def test_rollout_consumes_recorded_actions_and_only_predicted_future_history(self):
        rows = episodes([(200,0,5,np.full(12,100.))])
        model = SpyModel('B1',2)
        model.input_mean[0] = 10
        model.input_sd[0] = 2
        model.delta_sd[0] = 2
        result = evaluate(model,rows,selection=np.array([2]))
        self.assertEqual(len(model.one_calls),1)
        _,history,lengths,labels = model.one_calls[0]
        np.testing.assert_array_equal(lengths,[2])
        np.testing.assert_allclose(history[0,:,0],[-5,45])
        np.testing.assert_allclose(history[0,:,59],[45,95])
        self.assertIsNone(labels)
        self.assertEqual(len(model.roll_calls),10)
        for call, expected_action in zip(model.roll_calls,range(3,13)):
            self.assertEqual(float(call[1][0,0]),expected_action)
            self.assertIsNone(call[4])
        self.assertEqual(float(model.roll_calls[0][0][0,0]),200)
        self.assertEqual(float(model.roll_calls[1][0][0,0]),203)
        # The appended next-state is the predicted203, not the observed300.
        self.assertEqual(float(model.roll_calls[1][2][0,-1,59]),(203-10)/2)
        expected = ((200+sum(range(3,13))-1200)/2)**2/51
        self.assertAlmostEqual(result['overall']['rollout_10_mse'],expected)
        self.assertEqual(result['overall']['rollout_starts'],1)
        # One step later lacks ten remaining transitions and must not roll across the boundary.
        later = evaluate(SpyModel('B1',2),rows,selection=np.array([3]))
        self.assertEqual(later['overall']['rollout_starts'],0)

    def test_history_resets_on_cube_task_episode_and_selection_does_not_cut_past(self):
        rows = episodes([(200,0,5,[1,1,1]),(200,1,5,[1,1,1]),
                         (200,1,6,[1,1,1]),(201,1,6,[1,1,1])])
        model = SpyModel('B1',5)
        indices = np.array([2,3,6,9])
        evaluate(model,rows,selection=indices,batch_size=10)
        _,history,lengths,_ = model.one_calls[0]
        np.testing.assert_array_equal(lengths,[2,0,0,0])
        np.testing.assert_array_equal(history[1:],0)
        np.testing.assert_array_equal(history[0,2:],0)
        np.testing.assert_array_equal(history[0,:2,0],[0,1])
        np.testing.assert_array_equal(history[0,:2,59],[1,2])

    def test_rollout_fills_an_empty_history_then_caps_its_window(self):
        rows = episodes([(200,0,5,np.full(10,100.))])
        model = SpyModel('B1',5)
        evaluate(model,rows,selection=np.array([0]))
        self.assertEqual([int(call[3][0]) for call in model.roll_calls], [0,1,2,3,4,5,5,5,5,5])
        next_history = model.roll_calls[1][2][0]
        self.assertEqual(float(next_history[0,59]),1.)
        np.testing.assert_array_equal(next_history[1:],0)

    def test_empty_literal_early_slice_and_nonfinite_rollout_are_explicit(self):
        rows = episodes([(200,2,5,np.ones(16))])
        rows['cube_moving'][:5] = False
        model = SpyModel()
        model.nonfinite_step = 2
        result = evaluate(model,rows)
        early = result['early_late']['overall']['early']
        self.assertEqual(early['selected_rows'],5)
        self.assertEqual(early['moving_rows'],0)
        self.assertIsNone(early['one_step_mse'])
        self.assertEqual(result['overall']['moving_rows'],11)
        self.assertEqual(result['overall']['rollout_starts'],2)
        self.assertEqual(result['overall']['rollout_nonfinite_starts'],2)
        self.assertIsNone(result['overall']['rollout_10_mse'])
        self.assertTrue(np.isfinite(result['overall']['one_step_mse']))
        json.dumps(result,allow_nan=False)

    def test_nonfinite_one_step_is_counted_and_never_averaged_away(self):
        class Nonfinite(SpyModel):
            def forward(self,*args,**kwargs):
                output = super().forward(*args,**kwargs)
                output[0,1] = float('inf')
                return output
        result = evaluate(Nonfinite(),episodes([(200,0,5,[1,2,3])]))
        score = result['overall']
        self.assertEqual(score['moving_rows'],3)
        self.assertEqual(score['one_step_nonfinite_rows'],1)
        self.assertEqual(score['one_step_valid_rows'],2)
        self.assertIsNone(score['one_step_mse'])
        self.assertIsNone(score['one_step_r2'])
        json.dumps(result,allow_nan=False)

    def test_oracle_is_normalized_for_forward_and_raw_for_advance(self):
        rows = episodes([(200,0,5,np.ones(10))])
        rows['true_mass'] = np.full(10,.5)
        rows['true_friction'] = np.full(10,.8)
        model = SpyModel('B2')
        model.oracle_mean[:] = torch.tensor([.2,.4])
        model.oracle_sd[:] = torch.tensor([.1,.2])
        evaluate(model,rows,selection=np.array([0]))
        np.testing.assert_allclose(model.one_calls[0][3],[[3,2]],rtol=1e-6)
        np.testing.assert_allclose(model.roll_calls[0][4],[[.5,.8]],rtol=1e-6)
        self.assertIsNone(model.one_calls[0][1])
        self.assertIsNone(model.roll_calls[0][2])


if __name__ == '__main__':
    unittest.main()
