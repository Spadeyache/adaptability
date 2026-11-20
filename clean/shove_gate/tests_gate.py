"""Scientific contract checks for gate metrics and leak-free recursion; no training."""
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from clean.shove_gate.models_gate import (
    DynamicsMLP, GROUPS, KINDS, SEEDS, equal_episode_cube_mean,
    fit_normalizer, load_episodes, rollout_batch, summarize_seed_results, evaluation_data_manifest, digest, filename,
)


class ExactDynamics(nn.Module):
    kind='plain'
    def forward(self,state,action,context=None):
        delta=torch.zeros_like(state)
        delta[:,:3]=state[:,9:12]*.005
        delta[:,9:12]=action*.005
        return delta


def norm():
    return dict(state_mean=torch.zeros(15),state_std=torch.ones(15),
                action_mean=torch.zeros(3),action_std=torch.ones(3))


class GateContracts(unittest.TestCase):
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
                load_episodes(Path(directory),[200],[0])

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
            for cube in range(200,230):
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
            (root/'data/raw'/filename(200,0)).write_bytes(b'changed')
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
                           ['one_step_nmse','rollout10_nmse','rollout10_final_nmse','stopping_position_error_m']}
                        for g in GROUPS}
                records.append(dict(kind=kind,seed=seed,groups=groups))
        return records

    def test_twenty_percent_boundary_and_failure_rule(self):
        self.assertTrue(summarize_seed_results(self.fake_results(.8))['primary_gate_passed'])
        self.assertFalse(summarize_seed_results(self.fake_results(.81))['primary_gate_passed'])
        missing=self.fake_results(.5)
        missing[0]['groups']['moving_coast']['one_step_nmse']['mean']=None
        self.assertFalse(summarize_seed_results(missing)['primary_gate_passed'])


if __name__=='__main__':unittest.main()
