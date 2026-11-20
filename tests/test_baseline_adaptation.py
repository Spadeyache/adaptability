"""Focused tests of adaptation boundaries, optimizer state, and aggregation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.world_model_baselines.adaptation import (
    _adapt_cube, _adapt_sequence, _check_evaluation, _hierarchical, _pool_current,
    _prepare_cubes, _seed, aggregate_reports, render_report,
)


class TinyB0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kind = 'B0'
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('input_mean', torch.tensor([.25, -.25]))
        self.register_buffer('input_sd', torch.tensor([2., 3.]))
        self.register_buffer('delta_mean', torch.tensor([.5]))
        self.register_buffer('delta_sd', torch.tensor([2.]))

    @property
    def device(self):
        return self.weight.device

    def normalize(self, inputs, targets):
        return (inputs-self.input_mean)/self.input_sd, (targets-self.delta_mean)/self.delta_sd

    def forward(self, values):
        return values[:, :1]*self.weight+self.bias


def metrics(value=1., rows=10):
    return dict(selected_rows=rows, moving_rows=rows, one_step_mse=value,
                one_step_r2=1-value, one_step_sse=value*rows*51, one_step_sst=rows*51,
                rollout_starts=rows, rollout_10_mse=2*value,
                one_step_nonfinite_rows=0, rollout_nonfinite_starts=0)


def evaluation(value):
    return dict(overall=metrics(value), per_task={'lift': metrics(value)})


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_fresh_optimizer_all_weights_fixed_buffers_and_only_context_rows(self):
        model = TinyB0().requires_grad_(False)
        rows = dict(state=np.arange(4, dtype=np.float32)[:, None], action=np.zeros((4, 1), np.float32),
                    next_state=(np.arange(4, dtype=np.float32)+2)[:, None])
        # Poison evaluation targets: adaptation must never read them.
        rows['next_state'][2:] = np.nan
        selected = np.array([True, True, False, False])
        original_buffers = {name: value.clone() for name, value in model.named_buffers()}
        original_adam = torch.optim.Adam
        optimizers = []
        def fresh(parameters, **kwargs):
            result = original_adam(parameters, **kwargs)
            self.assertFalse(result.state)
            optimizers.append(result)
            return result
        with patch('experiments.world_model_baselines.adaptation.torch.optim.Adam', side_effect=fresh):
            first = _adapt_cube(model, rows, selected, learning_rate=.01, steps=3, batch_size=17, torch_seed=3)
            previous = model.bias.item()
            second = _adapt_cube(model, rows, selected, learning_rate=.01, steps=3, batch_size=17, torch_seed=4)
        self.assertEqual(len(optimizers), 2)
        self.assertIsNot(optimizers[0], optimizers[1])
        self.assertGreater(model.bias.item(), previous)
        self.assertNotEqual(model.weight.item(), 0.)
        for optimizer in optimizers:
            self.assertTrue(all(int(value['step']) == 3 for value in optimizer.state.values()))
        self.assertTrue(all(torch.equal(value, dict(model.named_buffers())[key]) for key,value in original_buffers.items()))
        self.assertEqual(first['context_rows'], 2)
        self.assertEqual(second['batch_size'], 17)
        self.assertTrue(first['replacement'])
        self.assertLess(second['context_mse_after'], first['context_mse_before'])

    def test_weight_carry_and_first_cube_retention_count_further_cubes(self):
        model = TinyB0()
        blocks = {cube:dict(rows={'cube':cube}, context=np.array([True,False]),
                            evaluation=np.array([False,True])) for cube in (200,201,202)}
        before_weights, scored = [], []
        def adapt(current, rows, context, **kwargs):
            before_weights.append(current.weight.item())
            with torch.no_grad():
                current.weight.add_(1.)
            return dict(torch_seed=kwargs['torch_seed'])
        def score(current, rows, selection):
            scored.append((rows['cube'],current.weight.item()))
            return evaluation(current.weight.item()), .01
        report = dict(current_cubes=[], retention={})
        with patch('experiments.world_model_baselines.adaptation._adapt_cube', side_effect=adapt), \
             patch('experiments.world_model_baselines.adaptation._timed_evaluation', side_effect=score):
            _adapt_sequence(model, blocks, [200,201,202], dict(steps_per_cube=100,batch_size=1024,retained_k=[0,1,2]),
                base_seed=0,order_id=1,learning_rate=1e-4,report=report)
        self.assertEqual(before_weights,[0.,1.,2.])
        self.assertEqual(scored,[(200,1.),(201,2.),(200,2.),(202,3.),(200,3.)])
        self.assertEqual([report['retention'][str(k)]['evaluation']['overall']['one_step_mse'] for k in (0,1,2)], [1.,2.,3.])
        self.assertTrue(report['retention']['0']['reused_current_evaluation'])
        self.assertEqual(_seed(0,1,0,200),report['current_cubes'][0]['adaptation']['torch_seed'])

    def test_context_eval_cube_blocks_and_property_labels_rejected(self):
        episodes = np.repeat(np.arange(10),120)
        rows = dict(cube_id=np.full(1200,200), task=np.zeros(1200,int), episode_id=episodes,
                    state=np.zeros((1200,51)), next_state=np.zeros((1200,51)), action=np.zeros((1200,8)))
        blocks = _prepare_cubes(rows, dict(context=episodes<5,eval=episodes>=5))
        self.assertEqual(blocks[200]['context'].sum(),600)
        self.assertEqual(blocks[200]['evaluation'].sum(),600)
        with self.assertRaisesRegex(ValueError,'physical labels'):
            _adapt_cube(TinyB0(),dict(true_mass=np.ones(2)),np.ones(2,bool),
                        learning_rate=.01,steps=1,batch_size=2,torch_seed=1)

    def test_nonfinite_counters_stop_but_empty_subsets_are_allowed(self):
        _check_evaluation(dict(one_step_mse=None,one_step_r2=None,rollout_10_mse=None,
                               moving_rows=0,one_step_nonfinite_rows=0,rollout_nonfinite_starts=0))
        with self.assertRaises(FloatingPointError):
            _check_evaluation(dict(rollout_10_mse=None,rollout_nonfinite_starts=1))
        with self.assertRaises(FloatingPointError):
            _check_evaluation(dict(one_step_mse=float('inf')))

    def test_current_cube_pooling_uses_rows_and_global_target_sst(self):
        values = [dict(evaluation=evaluation(1.)),dict(evaluation=evaluation(3.))]
        values[1]['evaluation']['per_task']['lift'] = metrics(3.,rows=30)
        pooled = _pool_current(values, target_sst=10000.)
        self.assertAlmostEqual(pooled['one_step_mse'],2.5)
        self.assertAlmostEqual(pooled['rollout_10_mse'],5.)
        self.assertAlmostEqual(pooled['one_step_r2'],1-5100/10000)

    def test_hierarchical_summary_and_report_use_seed_not_trajectory_sd(self):
        values=[(0,0,metrics(1.)),(0,1,metrics(3.)),(1,0,metrics(5.)),(1,1,metrics(7.))]
        result = _hierarchical(values)
        self.assertEqual(result['one_step_mse']['mean'],4.)
        self.assertAlmostEqual(result['one_step_mse']['sd_across_seeds'],np.sqrt(8.))
        self.assertAlmostEqual(result['one_step_mse']['within_seed_order_sd']['0'],np.sqrt(2.))
        records=[]
        for seed,order,value in values:
            records.append(dict(base_seed=seed,order_id=order,learning_rate=1e-4,
                current_cube_pooled=value,retention={str(k):value for k in [0,1,5,20]},
                retention_change_from_k0={str(k):metrics(0.) for k in [0,1,5,20]},
                **{name:{task:value for task in ['overall','lift','slow_slide','shove']} for name in
                   ['train_validation_before','train_validation_after','train_validation_change']}))
        config=dict(base_seeds=[0,1],orders=[[200],[201]],learning_rates=[1e-4],retained_k=[0,1,5,20])
        aggregate=aggregate_reports(records,config)
        self.assertTrue(aggregate['complete_design'])
        self.assertTrue(aggregate['complete_primary_metrics'])
        self.assertFalse(aggregate_reports(records[:1],config)['complete_design'])
        with tempfile.TemporaryDirectory() as folder:
            lines=render_report(aggregate,folder)
            self.assertEqual(len(lines),17)
            self.assertTrue((Path(folder)/'plot3_retention.png').exists())
            self.assertTrue((Path(folder)/'plot3_retention.pdf').exists())

    def test_missing_metric_is_preserved_and_prevents_a_complete_mean(self):
        missing=metrics(3.)
        missing['one_step_mse']=None
        result=_hierarchical([(0,0,metrics(1.)),(0,1,missing),(1,0,metrics(5.)),(1,1,metrics(7.))])
        value=result['one_step_mse']
        self.assertIsNone(value['mean'])
        self.assertIsNone(value['sd_across_seeds'])
        self.assertIsNone(value['seed_means']['0'])
        self.assertEqual(value['seed_means']['1'],6.)
        self.assertEqual(value['available_trajectories'],3)
        self.assertEqual(value['expected_trajectories'],4)
        self.assertEqual(len(value['raw']),4)
        self.assertIsNone(value['raw'][1]['value'])


if __name__ == '__main__':
    unittest.main()
