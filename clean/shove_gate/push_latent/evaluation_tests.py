"""Synthetic contracts only: never use real held-out motion or fit dynamics."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import evaluation as ev


def normalizer():
    return dict(state_mean=torch.zeros(15), state_std=torch.ones(15),
                action_mean=torch.zeros(3), action_std=torch.ones(3),
                horizontal_velocity_delta_mse_scale=1.)


class RecordedHistory(nn.Module):
    variant = 'history'
    def __init__(self):
        super().__init__(); self.seen = []
    def forward(self, state, action, context=None, history=None, z=None):
        self.seen.append(history.clone())
        delta = torch.zeros_like(state)
        delta[:, :3] = state[:, 9:12]*.005
        delta[:, 9:11] = .01
        return delta


class EvaluationContracts(unittest.TestCase):
    def test_equal_episode_cube_aggregation_and_nonfinite_stop(self):
        result = ev.aggregate([1.]+[3.]*9+[10.], np.array([0]+[1]*9+[2]),
                              np.array([260,260,261]), np.ones(11,dtype=bool))
        self.assertEqual(result['mean'],6.)
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            ev.aggregate([1.,np.inf],np.array([0,1]),np.array([260,261]),np.ones(2,dtype=bool))

    def test_c1_twenty_percent_and_spread_are_both_required(self):
        plain = ev.moments([1.]*5)
        self.assertTrue(ev.c1_decision(plain,ev.moments([.8]*5))['passed'])
        self.assertFalse(ev.c1_decision(plain,ev.moments([.80001]*5))['passed'])
        noisy = ev.moments([.6,1,1,1,1.4])
        self.assertFalse(ev.c1_decision(noisy,ev.moments([v*.8 for v in noisy['seed_values']]))['passed'])
        self.assertFalse(ev.c1_decision(plain,ev.moments([.5]*5),coverage=False)['passed'])

    def test_c2_half_gap_and_c3_both_strict_controls(self):
        p,o,z = [ev.moments([x]*5) for x in [1.,.4,.7]]
        c2,c3 = ev.c2_c3_decisions(p,o,z,ev.moments([.8]*5),ev.moments([.9]*5))
        self.assertTrue(c2['passed']);self.assertAlmostEqual(c2['recovered_fraction'],.5)
        self.assertTrue(c3['passed'])
        c2,c3 = ev.c2_c3_decisions(p,o,ev.moments([.70001]*5),ev.moments([.8]*5),ev.moments([.9]*5))
        self.assertFalse(c2['passed'])
        _,c3 = ev.c2_c3_decisions(p,o,z,ev.moments([.7]*5),ev.moments([.9]*5))
        self.assertFalse(c3['passed'])
        _,c3 = ev.c2_c3_decisions(p,o,z,ev.moments([.9]*5),ev.moments([.7]*5))
        self.assertFalse(c3['passed'])

    def test_wrong_z_is_fixed_derangement_independent_of_properties(self):
        mapping = ev.wrong_z_mapping(list(reversed(ev.TEST_CUBES)))
        self.assertEqual(mapping[260],261);self.assertEqual(mapping[289],260)
        self.assertEqual(set(mapping),set(mapping.values()))
        self.assertTrue(all(a!=b for a,b in mapping.items()))

    def test_history_rollout_appends_predictions_and_ignores_future_targets(self):
        state=torch.zeros((2,15));state[:,9]=torch.tensor([.3,.6])
        force=torch.tensor([[2.,0.,0.],[1.,0.,0.]])
        history=torch.arange(2*10*33,dtype=torch.float32).reshape(2,10,33)/1000
        remaining=torch.tensor([12,15]);push=torch.tensor([3,0])
        targets=torch.zeros((2,10,15));nxt=torch.zeros((2,15));stop=torch.zeros((2,2),dtype=torch.float64)
        a=RecordedHistory();b=RecordedHistory()
        first=ev.predict_batch(a,state,force,None,history,None,normalizer(),push,remaining,targets,nxt,stop)
        second=ev.predict_batch(b,state,force,None,history,None,normalizer(),push,remaining,targets+999,nxt+999,stop)
        self.assertEqual(len(a.seen),15)
        for one,two in zip(a.seen,b.seen):torch.testing.assert_close(one,two,rtol=0,atol=0)
        torch.testing.assert_close(a.seen[0],history,rtol=0,atol=0)
        torch.testing.assert_close(a.seen[1][:,-1,:15],state,rtol=0,atol=0)
        torch.testing.assert_close(a.seen[1][0,-1,15:18],force[0],rtol=0,atol=0)
        torch.testing.assert_close(a.seen[1][1,-1,15:18],torch.zeros(3),rtol=0,atol=0)
        np.testing.assert_array_equal(first['stopping_position_error_m'],second['stopping_position_error_m'])
        self.assertTrue(np.all(second['rollout10_nmse']>first['rollout10_nmse']))

    def test_inactive_forecasts_are_not_evaluated_after_planned_horizon(self):
        class PoisonAfterHorizon(nn.Module):
            variant='plain'
            def __init__(self):super().__init__();self.batch_sizes=[]
            def forward(self,state,action,context=None,history=None,z=None):
                self.batch_sizes.append(len(state));delta=torch.zeros_like(state);delta[:,0]=1.
                # Row0 is only safe for its single requested forecast step.
                poisoned=(state[:,1]==1)&(state[:,0]>0)
                delta[poisoned]=float('nan')
                return delta
        model=PoisonAfterHorizon();state=torch.zeros((2,15));state[0,1]=1.
        out=ev.predict_batch(model,state,torch.zeros((2,3)),None,None,None,normalizer(),
            torch.zeros(2,dtype=torch.long),torch.tensor([1,12]),torch.zeros((2,10,15)),
            torch.zeros((2,15)),torch.zeros((2,2),dtype=torch.float64))
        self.assertEqual(model.batch_sizes,[2]+[1]*11)
        self.assertTrue(np.isfinite(out['stopping_position_error_m']).all())

    def test_nonfinite_actual_prediction_stops_instead_of_dropping_it(self):
        class Nonfinite(nn.Module):
            variant='plain'
            def forward(self,state,action,context=None,history=None,z=None):return torch.full_like(state,float('nan'))
        with self.assertRaisesRegex(RuntimeError,'Nonfinite predictive output'):
            ev.predict_batch(Nonfinite(),torch.zeros((1,15)),torch.zeros((1,3)),None,None,None,normalizer(),
                torch.zeros(1,dtype=torch.long),torch.tensor([10]),torch.zeros((1,10,15)),
                torch.zeros((1,15)),torch.zeros((1,2),dtype=torch.float64))

    def test_mass_readout_uses_frozen_affine_coefficients(self):
        zs=np.array([[0.,1.],[1.,0.],[1.,1.]])
        decoder=dict(coefficients=[.2,.1,.2],training_mean_mass_kg=.35)
        truth={260:.4,261:.3,262:.5}
        score=ev.mass_score(zs,[260,261,262],decoder,truth)
        self.assertAlmostEqual(score['r2'],1.)
        self.assertAlmostEqual(score['mae_kg'],0.)
        self.assertTrue(all(set(r)=={'cube_id','prediction_kg'} for r in score['predicted_mass_by_cube']))

    def test_c1_failure_finishes_history_report_without_test_z_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'results').mkdir()
            (root/'results/normalizer.json').write_text('{}')
            (root/'results/action_probe.json').write_text('{}')
            def checkpoint(path,device):
                variant,seed=path.stem.split('_seed_')
                return SimpleNamespace(variant=variant),dict(seed=int(seed))
            def synthetic_score(_root,model,cp,cp_path,pack,props,device,name,**kwargs):
                value={'plain':1.,'oracle':.9,'history':.85}[name]
                return dict(model_name=name,seed=cp['seed'],checkpoint_sha256='synthetic',
                    groups={g:{m:dict(mean=value,cubes=30,episodes=150) for m in ev.METRICS} for g in ev.GROUPS})
            with patch.object(ev,'freeze_context',return_value={'synthetic':True}),\
                 patch.object(ev,'make_query_pack',return_value={}),\
                 patch.object(ev.models,'property_lookup',return_value={}),\
                 patch.object(ev.models,'load_checkpoint',side_effect=checkpoint),\
                 patch.object(ev,'evaluate_model',side_effect=synthetic_score),\
                 patch.object(ev.models,'fit_zs',side_effect=AssertionError('C1 failure must not adapt test z')) as fit:
                result=ev.run(root,'cpu')
                self.assertFalse(result['c1']['passed']);self.assertTrue(result['stopped_after_c1'])
                self.assertIsNone(result['c2']);self.assertIsNone(result['c3'])
                self.assertIn('history',result['by_model']);fit.assert_not_called()
                self.assertEqual(json.loads((root/'results/final_evaluation.json').read_text())['status'],'complete')
                self.assertFalse(list((root/'results').glob('prefix_scores*')))

    def test_c1_pass_runs_fixed_prefixes_controls_and_frozen_decoder_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for folder in ['results','data','checkpoints']:(root/folder).mkdir()
            (root/'results/normalizer.json').write_text('{}')
            (root/'results/action_probe.json').write_text('{}')
            config=dict(z_dim=2,variant='z2',learning_rate=.01,updates=50)
            (root/'results/selected_config.json').write_text(json.dumps(config))
            decoder=dict(coefficients=[.2,.1,.01],training_mean_mass_kg=.26)
            (root/'results/mass_readouts.json').write_text(json.dumps({'by_seed':{str(s):decoder for s in ev.SEEDS}}))
            (root/'data/cubes.json').write_text(json.dumps({'cubes':[dict(cube_id=c,mass_kg=.25+.001*(c-260)) for c in ev.TEST_CUBES]}))
            for seed in ev.SEEDS:(root/'checkpoints'/f'z2_seed_{seed}.pt').write_bytes(b'synthetic')
            fit_calls=[];wrong_calls=[]
            def checkpoint(path,device):
                variant,seed=path.stem.split('_seed_')
                embedding=nn.Embedding(180,2);nn.init.zeros_(embedding.weight)
                return SimpleNamespace(variant=variant,embedding=embedding,seed=int(seed)),dict(seed=int(seed),normalizer={})
            def load_support(root,cubes,episodes,allow_test=False):
                self.assertTrue(allow_test)
                return [dict(cube_id=c,episode_id=e) for c in cubes for e in episodes]
            def fit(model,support,norm,initial,lr,updates,device):
                count=max(e['episode_id'] for e in support)+1
                self.assertEqual(sorted({e['episode_id'] for e in support}),list(range(count)))
                self.assertEqual(lr,.01);self.assertEqual(updates,50)
                torch.testing.assert_close(initial,torch.zeros(2),rtol=0,atol=0)
                fit_calls.append((model.seed,count))
                z=np.array([[(c-260)/100,count] for c in ev.TEST_CUBES],dtype=np.float32)
                return dict(cube_ids=ev.TEST_CUBES,snapshots={50:z},log=[],weights_unchanged=True)
            def score(_root,model,cp,cp_path,pack,props,device,name,z_by_cube=None,**kwargs):
                if name.startswith('fitted_z_'):value=1-.1*int(name.rsplit('_',1)[1])
                else:value={'plain':1.,'oracle':.4,'history':.6,'fitted_z':.5,'mean_z':.9,'wrong_z':.85}[name]
                if name=='wrong_z':
                    wrong_calls.append(model.seed)
                    np.testing.assert_allclose(z_by_cube[260],[.01,5],rtol=1e-7,atol=0)
                    np.testing.assert_allclose(z_by_cube[289],[0,5],rtol=0,atol=0)
                return dict(model_name=name,seed=cp['seed'],checkpoint_sha256='synthetic',
                    groups={g:{m:dict(mean=value,cubes=30,episodes=150) for m in ev.METRICS} for g in ev.GROUPS})
            with patch.object(ev,'freeze_context',return_value={'synthetic':True}),\
                 patch.object(ev,'make_query_pack',return_value={}),\
                 patch.object(ev.models,'property_lookup',return_value={}),\
                 patch.object(ev.models,'load_checkpoint',side_effect=checkpoint),\
                 patch.object(ev.models,'load_episodes',side_effect=load_support),\
                 patch.object(ev,'evaluate_model',side_effect=score),\
                 patch.object(ev.models,'fit_zs',side_effect=fit):
                result=ev.run(root,'cpu')
                self.assertTrue(result['c1']['passed']);self.assertTrue(result['c2']['passed']);self.assertTrue(result['c3']['passed'])
                self.assertEqual(fit_calls,[(s,k) for s in ev.SEEDS for k in range(1,6)])
                self.assertEqual(wrong_calls,ev.SEEDS)
                self.assertEqual(set(result['by_model']),set(ev.BASELINES+ev.CONTROLS))
                self.assertEqual(set(result['prefix_curve']),set(map(str,range(1,6))))
                self.assertAlmostEqual(result['mass_readout']['r2']['mean'],1.,places=10)


if __name__=='__main__':unittest.main()
