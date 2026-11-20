"""Independent checks of the full crossed design and literal retention schedule."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.world_model_baselines.adaptation import aggregate_reports, _adapt_sequence


def metrics(value):
    return dict(one_step_mse=float(value), one_step_r2=1-float(value), rollout_10_mse=2*float(value))


class IndependentAdaptationAudit(unittest.TestCase):
    def test_all_50_crossed_trajectories_and_seed_level_spread(self):
        config = dict(base_seeds=list(range(5)), orders=[list(range(200,230)) for _ in range(5)],
                      learning_rates=[1e-4,1e-5], retained_k=[0,1,5,20])
        records = []
        for rate in config['learning_rates']:
            for seed in config['base_seeds']:
                for order in range(5):
                    value = metrics(100*seed+order)
                    records.append(dict(base_seed=seed,order_id=order,learning_rate=rate,
                        current_cube_pooled=value,retention={str(k):value for k in config['retained_k']},
                        retention_change_from_k0={str(k):metrics(0) for k in config['retained_k']},
                        **{name:{task:value for task in ('overall','lift','slow_slide','shove')}
                           for name in ('train_validation_before','train_validation_after','train_validation_change')}))
        result = aggregate_reports(records,config)
        self.assertTrue(result['complete_design'])
        self.assertTrue(result['complete_primary_metrics'])
        self.assertEqual(result['completed_trajectories'],50)
        self.assertEqual(result['expected_trajectories'],50)
        for rate in config['learning_rates']:
            summary = result['learning_rates'][str(rate)]['current_cube_pooled']['one_step_mse']
            self.assertEqual(summary['mean'],202.)
            self.assertAlmostEqual(summary['sd_across_seeds'],np.std([2,102,202,302,402],ddof=1))
            self.assertEqual(summary['seed_count'],5)
            self.assertEqual(summary['trajectory_count'],25)
            self.assertEqual(len(summary['raw']),25)
            for value in summary['within_seed_order_sd'].values():
                self.assertAlmostEqual(value,np.std(np.arange(5),ddof=1))
        self.assertFalse(aggregate_reports(records[:-1],config)['complete_design'])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            aggregate_reports(records+[deepcopy(records[0])],config)
        missing = deepcopy(records)
        missing[0]['current_cube_pooled'] = dict(missing[0]['current_cube_pooled'],one_step_mse=None)
        incomplete = aggregate_reports(missing,config)
        self.assertTrue(incomplete['complete_design'])
        self.assertFalse(incomplete['complete_primary_metrics'])
        self.assertIsNone(incomplete['learning_rates']['0.0001']['current_cube_pooled']['one_step_mse']['mean'])
        self.assertEqual(incomplete['missing_primary_metrics'][0]['scope'],'current_cube_pooled')

    def test_retention_uses_the_same_first_cube_after_zero_one_five_twenty_further_fits(self):
        model = torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():
            model.weight.zero_()
        order = list(range(201,230))+[200]
        blocks = {cube:dict(rows={'cube_id':cube},context=np.array([True,False]),
                            evaluation=np.array([False,True])) for cube in order}
        fitted, scored = [], []
        def adapt(current,rows,context,**kwargs):
            np.testing.assert_array_equal(context,[True,False])
            fitted.append((rows['cube_id'],float(current.weight.detach())))
            with torch.no_grad():
                current.weight.add_(1)
            return dict(torch_seed=kwargs['torch_seed'])
        def score(current,rows,selection):
            np.testing.assert_array_equal(selection,[False,True])
            scored.append((rows['cube_id'],float(current.weight.detach())))
            return dict(cube_id=rows['cube_id'],weight=float(current.weight.detach())),0.
        report = dict(current_cubes=[],retention={})
        with patch('experiments.world_model_baselines.adaptation._adapt_cube',side_effect=adapt), \
             patch('experiments.world_model_baselines.adaptation._timed_evaluation',side_effect=score):
            _adapt_sequence(model,blocks,order,dict(steps_per_cube=100,batch_size=1024,retained_k=[0,1,5,20]),
                            base_seed=4,order_id=3,learning_rate=1e-4,report=report)
        self.assertEqual([cube for cube,_ in fitted],order)
        self.assertEqual([weight for _,weight in fitted],list(map(float,range(30))))
        self.assertEqual(len(scored),33)
        for k in [0,1,5,20]:
            value = report['retention'][str(k)]
            self.assertEqual(value['further_cubes'],k)
            self.assertEqual(value['evaluation']['cube_id'],201)
            self.assertEqual(value['evaluation']['weight'],float(k+1))


if __name__ == '__main__':
    unittest.main()
