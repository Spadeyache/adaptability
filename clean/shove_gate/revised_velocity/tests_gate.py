"""Scientific contract checks for gate metrics and leak-free recursion; no training."""
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from clean.shove_gate.revised_velocity.models_gate import (
    DynamicsMLP, GROUPS, KINDS, SEEDS, equal_episode_cube_mean,
    fit_normalizer, load_episodes, rollout_batch, summarize_seed_results, evaluation_data_manifest, digest, filename, PRIMARY_METRIC, TEST_CUBES, extension_decision, validation_score,
)
from clean.shove_gate.revised_velocity.physics_floor import horizontal_floor, compute_metrics


class ExactDynamics(nn.Module):
    kind='plain'
    def forward(self,state,action,context=None):
        delta=torch.zeros_like(state)
        delta[:,:3]=state[:,9:12]*.005
        delta[:,9:12]=action*.005
        return delta


def norm():
    return dict(state_mean=torch.zeros(15),state_std=torch.ones(15),
                action_mean=torch.zeros(3),action_std=torch.ones(3),horizontal_velocity_delta_mse_scale=1.)


class GateContracts(unittest.TestCase):
    def test_retired_cubes_are_forbidden_even_with_test_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'Retired test cubes'):
                load_episodes(Path(directory),[200],[0],allow_test=True)
        self.assertEqual(TEST_CUBES,list(range(230,260)))

    def test_delta_energy_uses_equal_episode_then_cube_weights(self):
        episodes=[]
        for cid,eid,length,dv in [(0,0,1,1.),(0,1,9,3.),(1,0,1,10.)]:
            state=np.zeros((length,15));state[:,9]=20.
            nxt=state.copy();nxt[:,9:11]+=dv
            episodes.append(dict(cube_id=cid,episode_id=eid,state=state,next_state=nxt,
                                 phase=np.full(length,'coast'),action=np.zeros((length,3))))
        properties={c:np.array([.3,.5]) for c in range(200)}
        normalization=fit_normalizer(episodes,properties)
        self.assertEqual(normalization['horizontal_velocity_delta_mse_scale'],52.5)
        # This differs from a length-weighted pooled mean, (1+81+100)/11.
        self.assertNotAlmostEqual(normalization['horizontal_velocity_delta_mse_scale'],182/11)

    def test_selection_prefers_horizontal_delta_over_all_state_metric(self):
        class Fixed(nn.Module):
            kind='plain'
            def __init__(self, vector):super().__init__();self.vector=vector
            def forward(self,state,action,context=None):return self.vector.expand(len(state),-1)
        target=torch.zeros((2,15));target[:,9:11]=1.
        horizontal_good=torch.zeros((1,15));horizontal_good[:,9:11]=1.;horizontal_good[:,:3]=100.
        horizontal_bad=torch.zeros((1,15))
        pack=dict(x=torch.zeros((2,15)),action=torch.zeros((2,3)),context=torch.zeros((2,2)),target=target,
                  horizontal_state_std=torch.ones(2),horizontal_velocity_delta_mse_scale=1.,
                  horizontal_true_delta=target[:,9:11].double(),
                  row_episode=np.array([0,0]),cube_ids=np.array([0]),moving_coast=np.array([True,True]))
        a=validation_score(Fixed(horizontal_good),pack);b=validation_score(Fixed(horizontal_bad),pack)
        self.assertEqual(a['primary'],0.)
        self.assertLess(a['primary'],b['primary'])
        self.assertGreater(a['secondary_all_step'],b['secondary_all_step'])
        # Selection must use measured raw delta, not reconstruct a rounded target.
        pack['horizontal_true_delta']=torch.zeros((2,2),dtype=torch.float64)
        self.assertEqual(validation_score(Fixed(horizontal_bad),pack)['primary'],0.)

    def test_extension_threshold_either_family_and_plateau(self):
        def logs(recent):
            return {kind:[dict(update=i,validation_nmse=(recent if i>7500 and kind=='oracle' else 1.))
                          for i in range(500,10001,500)] for kind in KINDS}
        boundary=extension_decision(logs(.999),10000)
        self.assertTrue(boundary['extend'])
        self.assertTrue(boundary['families']['oracle']['still_improving'])
        self.assertFalse(boundary['families']['plain']['still_improving'])
        self.assertFalse(extension_decision(logs(.9991),10000)['extend'])
        self.assertFalse(extension_decision(logs(1.1),10000)['extend'])

    def test_physics_helper_matches_requested_formula_and_stop_clip_is_separate(self):
        state=np.zeros((2,15));state[:,9:11]=[[.3,.4],[.001,0.]]
        expected=-.5*9.81*.005*np.array([[.6,.8],[1.,0.]])
        np.testing.assert_allclose(horizontal_floor(state,.5),expected,rtol=0,atol=1e-16)
        clipped=horizontal_floor(state,.5,clip_at_zero=True)
        np.testing.assert_allclose(clipped[1],[-.001,0.],rtol=0,atol=1e-16)
        nxt=state.copy();nxt[:,9:11]+=expected
        episodes=[dict(cube_id=230,episode_id=0,state=state,next_state=nxt,
                       phase=np.full(2,'coast'),combined_friction=.5)]
        metric=compute_metrics(episodes,1.)['metrics']['requested_unclipped']
        self.assertAlmostEqual(metric['nmse'],0.,places=20)

    def test_equal_cube_episode_weighting_ignores_long_episode_dominance(self):
        values=np.array([1.]+[3.]*9+[10.])
        ids=np.array([0]+[1]*9+[2])
        result=equal_episode_cube_mean(values,ids,np.array([0,0,1]))
        self.assertEqual(result['mean'],6.)
        self.assertEqual(result['episodes'],3)
        self.assertEqual(result['cubes'],2)

    def test_divergence_is_not_silently_dropped(self):
        result=equal_episode_cube_mean([1.,np.inf],np.array([0,1]),np.array([0,1]))
        self.assertIsNone(result['mean'])
        self.assertEqual(result['nonfinite_starts'],1)

    def test_test_motion_guard_precedes_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'Test motion'):
                load_episodes(Path(directory),[230],[0])

    def test_normalization_rejects_validation_or_test_sources(self):
        for cube,episode in [(0,8),(200,0)]:
            with self.assertRaisesRegex(ValueError,'nontraining'):
                fit_normalizer([dict(cube_id=cube,episode_id=episode)],{})

    def test_heldout_byte_mutation_is_detected_without_loading_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for kind in ['episodes','raw']:(root/'data'/kind).mkdir(parents=True)
            (root/'results').mkdir()
            records=[]
            for cube in TEST_CUBES:
                for episode in range(10):
                    row=dict(cube_id=cube,episode_id=episode)
                    for kind in ['episodes','raw']:
                        path=root/'data'/kind/filename(cube,episode)
                        # Deliberately not valid NPZ: manifest verification must
                        # only hash bytes, never decode test observations.
                        path.write_bytes(f'{kind}:{cube}:{episode}'.encode())
                        row[kind+'_sha256']=digest(path)
                    records.append(row)
            (root/'results/metrics.json').write_text(json.dumps(records))
            (root/'results/full_audit.json').write_text('{"status":"passed"}')
            self.assertEqual(len(evaluation_data_manifest(root)),602)
            (root/'data/raw'/filename(230,0)).write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                evaluation_data_manifest(root)

    def test_plain_cannot_read_context_and_architecture_is_three_by_256(self):
        torch.manual_seed(1)
        plain=DynamicsMLP('plain')
        state=torch.randn(2,15);action=torch.randn(2,3)
        a=plain(state,action,torch.zeros(2,2))
        b=plain(state,action,torch.full((2,2),999.))
        self.assertTrue(torch.equal(a,b))
        widths=[layer.out_features for layer in plain.net if isinstance(layer,nn.Linear)]
        self.assertEqual(widths,[256,256,256,15])
        with self.assertRaisesRegex(ValueError,'Oracle requires'):
            DynamicsMLP('oracle')(state,action)

    def test_recursive_metrics_match_exact_dynamics_and_fixed_horizon(self):
        model=ExactDynamics();state=torch.zeros((2,15));state[:,9]=torch.tensor([.3,.6])
        force=torch.tensor([[2.,-1.,0.],[0.,0.,0.]])
        push=torch.tensor([3,0]);remaining=torch.tensor([12,20])
        trajectories=[]
        for row in range(2):
            x=state[row:row+1].clone();trace=[]
            for k in range(int(remaining[row])):
                command=force[row:row+1] if k<int(push[row]) else torch.zeros((1,3))
                x=x+model(x,command);trace.append(x[0].clone())
            trajectories.append(trace)
        ten=torch.stack([torch.stack(t[:10]) for t in trajectories])
        next_state=ten[:,0];stop=torch.stack([t[-1][:2] for t in trajectories]).double()
        result=rollout_batch(model,state,force,None,norm(),push,remaining,ten,next_state,stop)
        for value in result.values():np.testing.assert_allclose(value,0,rtol=0,atol=1e-12)
        # Label targets affect scoring, never recursive inputs or terminal state.
        altered=rollout_batch(model,state,force,None,norm(),push,remaining,ten+100,next_state+100,stop)
        np.testing.assert_array_equal(altered['stopping_position_error_m'],result['stopping_position_error_m'])
        self.assertTrue(np.all(altered['rollout10_nmse']>0))

    @staticmethod
    def fake_results(ratio):
        records=[]
        for kind in KINDS:
            for seed in SEEDS:
                mean=1. if kind=='plain' else ratio
                groups={g:{metric:dict(mean=mean,cubes=30) for metric in
                           [PRIMARY_METRIC,'one_step_nmse','rollout10_nmse','rollout10_final_nmse','stopping_position_error_m']}
                        for g in GROUPS}
                records.append(dict(kind=kind,seed=seed,groups=groups))
        return records

    def test_twenty_percent_boundary_and_failure_rule(self):
        self.assertTrue(summarize_seed_results(self.fake_results(.8))['primary_gate_passed'])
        self.assertFalse(summarize_seed_results(self.fake_results(.81))['primary_gate_passed'])
        missing=self.fake_results(.5)
        missing[0]['groups']['moving_coast'][PRIMARY_METRIC]['mean']=None
        self.assertFalse(summarize_seed_results(missing)['primary_gate_passed'])
        old_metrics_better=self.fake_results(.81)
        for record in old_metrics_better:
            if record['kind']=='oracle':
                for group in GROUPS:
                    for metric in ['one_step_nmse','rollout10_nmse','rollout10_final_nmse','stopping_position_error_m']:
                        record['groups'][group][metric]['mean']=.01
        self.assertFalse(summarize_seed_results(old_metrics_better)['primary_gate_passed'])


if __name__=='__main__':unittest.main()
