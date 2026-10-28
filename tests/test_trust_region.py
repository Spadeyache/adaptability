"""Exact rollout KL transactions, full Adam restoration and frozen integration."""
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent,Normal,kl_divergence

from robot_adaptation.policy.model import Agent
from robot_adaptation.policy.trust_region import RolloutKLGuard,diagonal_gaussian_kl,effective_gaussian,guarded_ppo_update,OPTIMIZATION_SEMANTICS
from robot_adaptation.policy import train


class ToyPolicy(nn.Module):
    optimization_mode='joint'
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(()))
        self.critic=nn.Parameter(torch.tensor(2.))
        self.actor_logstd=nn.Parameter(torch.zeros((1,5)))
        self.strength_logstd=nn.Parameter(torch.zeros((1,1)))
        self.register_buffer('marker',torch.tensor(71.))
    def action_mean(self,obs):
        return torch.cat((self.weight*obs[:,:1],torch.zeros((len(obs),4),device=obs.device)),dim=-1)


def same_state(test,a,b):
    test.assertEqual(type(a),type(b))
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(a.keys(),b.keys())
        for k in a:same_state(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):same_state(test,x,y)
    else:test.assertEqual(a,b)


class TrustRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.previous)
    def setUp(self):
        random_context=torch.random.fork_rng(devices=[])
        random_context.__enter__();self.addCleanup(random_context.__exit__,None,None,None)
        numpy_state=np.random.get_state();self.addCleanup(np.random.set_state,numpy_state)
        previous_threads=torch.get_num_threads();self.addCleanup(torch.set_num_threads,previous_threads)
        torch.manual_seed(22);np.random.seed(22)
    def make(self,lr=.1):
        policy=ToyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=lr)
        return policy,optimizer,torch.ones((20,1))

    def test_exact_old_to_new_formula_matches_distribution_and_is_asymmetric(self):
        old_mean=torch.randn(7,5);new_mean=torch.randn(7,5)
        old_logstd=torch.randn(7,5)*.2;new_logstd=torch.randn(7,5)*.2
        expected=kl_divergence(Independent(Normal(old_mean.double(),old_logstd.double().exp()),1),Independent(Normal(new_mean.double(),new_logstd.double().exp()),1))
        actual=diagonal_gaussian_kl(old_mean,old_logstd,new_mean,new_logstd)
        torch.testing.assert_close(actual,expected)
        self.assertFalse(torch.allclose(actual,diagonal_gaussian_kl(new_mean,new_logstd,old_mean,old_logstd)))
        torch.testing.assert_close(diagonal_gaussian_kl(old_mean,old_logstd,old_mean,old_logstd),torch.zeros(7,dtype=torch.float64))

    def test_effective_logstd_excludes_dead_fifth_movement_std(self):
        policy,_,obs=self.make()
        with torch.no_grad():policy.actor_logstd[0,4]=100.;policy.strength_logstd[0,0]=-3.
        _,logstd=effective_gaussian(policy,obs)
        torch.testing.assert_close(logstd[:,:4],torch.zeros((20,4)))
        torch.testing.assert_close(logstd[:,4],torch.full((20,),-3.))

    def test_full_rollout_detects_change_hidden_at_selected_minibatch(self):
        policy,optimizer,obs=self.make()
        obs[0]=0.;obs[1:]=100.
        guard=RolloutKLGuard(policy,optimizer,obs,max_exact_kl=.01,max_backtracks=0)
        result=guard.step(lambda:-policy.weight)
        self.assertFalse(result.accepted)
        self.assertEqual(policy.weight.item(),0.)
        self.assertEqual(optimizer.state_dict()['state'],{})
        self.assertTrue(guard.stopped)

    def test_comparator_stays_original_and_rejection_restores_adam_history(self):
        policy,optimizer,obs=self.make()
        guard=RolloutKLGuard(policy,optimizer,obs,max_exact_kl=.03,max_backtracks=0)
        self.assertTrue(guard.step(lambda:-policy.weight).accepted)
        self.assertTrue(guard.step(lambda:-policy.weight).accepted)
        before_model=deepcopy(policy.state_dict());before_adam=deepcopy(optimizer.state_dict())
        result=guard.step(lambda:-policy.weight)
        self.assertFalse(result.accepted)
        same_state(self,before_model,policy.state_dict());same_state(self,before_adam,optimizer.state_dict())
        self.assertAlmostEqual(guard.exact_kl()[0],.02,places=6)
        self.assertEqual(guard.old_mean.count_nonzero().item(),0)
        self.assertEqual(guard.metrics['accepted_updates'],2)

    def test_nonfinite_poststep_restores_all_model_buffers_and_lazy_adam(self):
        policy,optimizer,obs=self.make()
        before_model=deepcopy(policy.state_dict());before_adam=deepcopy(optimizer.state_dict())
        guard=RolloutKLGuard(policy,optimizer,obs,max_backtracks=2)
        original=optimizer.step
        def broken_step():
            original()
            with torch.no_grad():policy.critic.fill_(float('nan'));policy.marker.fill_(-99.)
        with patch.object(optimizer,'step',broken_step):result=guard.step(lambda:-policy.weight)
        self.assertFalse(result.accepted)
        self.assertEqual(guard.metrics['nonfinite_candidates'],3)
        same_state(self,before_model,policy.state_dict());same_state(self,before_adam,optimizer.state_dict())

    def test_optimizer_exception_rolls_back_mutation_and_step_counters(self):
        policy,optimizer,obs=self.make()
        # Seed Adam history before guarding, to distinguish zeroing from restoration.
        (-policy.weight).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        before_model=deepcopy(policy.state_dict());before_adam=deepcopy(optimizer.state_dict())
        guard=RolloutKLGuard(policy,optimizer,obs)
        original=optimizer.step
        def throwing_step():original();raise RuntimeError('after mutation')
        with patch.object(optimizer,'step',throwing_step),self.assertRaisesRegex(RuntimeError,'after mutation'):
            guard.step(lambda:-policy.weight)
        same_state(self,before_model,policy.state_dict());same_state(self,before_adam,optimizer.state_dict())

    def test_lr_backtracking_is_temporary_and_new_rollout_restores_budget(self):
        policy,optimizer,obs=self.make(lr=.2)
        guard=RolloutKLGuard(policy,optimizer,obs,max_exact_kl=.0015,max_backtracks=2)
        first=guard.step(lambda:-policy.weight)
        self.assertTrue(first.accepted);self.assertEqual(first.backtracks,2)
        self.assertAlmostEqual(policy.weight.item(),.05,places=6)
        self.assertEqual(optimizer.param_groups[0]['lr'],.2)
        exhausted=guard.step(lambda:-policy.weight)
        self.assertFalse(exhausted.accepted)
        self.assertEqual(optimizer.param_groups[0]['lr'],.2)
        next_rollout=RolloutKLGuard(policy,optimizer,obs,max_exact_kl=.0015,max_backtracks=2)
        self.assertEqual(next_rollout.base_lrs,[.2])
        self.assertTrue(next_rollout.step(lambda:-policy.weight).accepted)
        self.assertAlmostEqual(policy.weight.item(),.1,places=6)
        self.assertEqual(optimizer.param_groups[0]['lr'],.2)

    def test_nonfinite_loss_and_initial_critic_are_rejected(self):
        policy,optimizer,obs=self.make()
        guard=RolloutKLGuard(policy,optimizer,obs)
        result=guard.step(lambda:policy.weight*float('nan'))
        self.assertFalse(result.accepted);self.assertEqual(result.trials,1)
        self.assertEqual(optimizer.state_dict()['state'],{})
        with torch.no_grad():policy.critic.fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'Starting model'):
            RolloutKLGuard(policy,optimizer,obs)

    def test_real_five_action_ppo_update_changes_weights_within_bound(self):
        agent=Agent(optimization_mode='joint',observation_count=46)
        observations=torch.randn(64,46)
        with torch.no_grad():
            actions,old_logprobs,_,values=agent.get_action_and_value(observations)
        optimizer=torch.optim.Adam(agent.parameters(),lr=1e-3)
        before=deepcopy(agent.state_dict())
        args=SimpleNamespace(max_grad_norm=.5,update_epochs=2,minibatch_size=16,norm_adv=True,clip_coef=.2,clip_vloss=False,ent_coef=0.,vf_coef=.5)
        metrics=guarded_ppo_update(agent,optimizer,observations,actions,old_logprobs,torch.randn(64),values.flatten()+torch.randn(64),values.flatten(),args,max_exact_kl=.01)
        self.assertGreater(metrics['accepted_updates'],0)
        self.assertLessEqual(metrics['max_accepted_exact_kl'],.01)
        self.assertLessEqual(metrics['rollout_exact_kl'],.01)
        self.assertEqual(metrics['effective_actions'],5)
        self.assertEqual(metrics['rollout_observations'],64)
        self.assertEqual(optimizer.param_groups[0]['lr'],1e-3)
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in agent.state_dict().items()))
        for k,v in agent.state_dict().items():self.assertEqual(before[k].shape,v.shape)

    def test_prepare_enabled_and_disabled_preserve_architecture_and_upstream_source(self):
        with TemporaryDirectory() as directory,patch('subprocess.Popen',side_effect=AssertionError('No launch')),redirect_stdout(StringIO()):
            settings=dict(checkpoint=None,initialize_from_pretrained=False,condition_on_friction=True,friction_profile='independent_grid',num_eval_envs=15,total_timesteps=9000,run_root=directory)
            enabled=train.main(dict(settings,guard_policy_updates=True,learning_rate=1e-5),prepare_only=True)
            disabled=train.main(settings,prepare_only=True)
            import json
            manifest=json.loads((enabled/'manifest.json').read_text())
            self.assertEqual(manifest['observations'],46)
            self.assertEqual(manifest['optimization_semantics'],OPTIMIZATION_SEMANTICS)
            self.assertTrue(manifest['trust_region']['enabled'])
            self.assertEqual(manifest['trust_region_sha256'],train.digest(enabled/'trust_region.py'))
            guarded=(enabled/'ppo_runtime.py').read_text();normal=(disabled/'ppo_runtime.py').read_text()
            compile(guarded,str(enabled/'ppo_runtime.py'),'exec')
            self.assertIn('guarded_ppo_update(',guarded);self.assertNotIn('guarded_ppo_update(',normal)
            self.assertIn('optimizer.step()',normal);self.assertNotIn('optimizer.step()',guarded)
            command=json.loads((enabled/'launch_command.json').read_text())['command']
            self.assertNotIn('--guard-policy-updates',command)
            self.assertIn('"a") as guard_log:',guarded)


if __name__=='__main__':unittest.main()
