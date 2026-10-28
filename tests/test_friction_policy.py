"""Friction extension: checkpoint preservation, input isolation, crossed design."""
from io import BytesIO
import unittest
import torch

from robot_adaptation.policy.model import Agent, initialize_from_pretrained
from robot_adaptation.policy.train import resolve_settings
from robot_adaptation.simulation.training_lift import friction_experiment
from experiments.physics_conditioning.evaluate import inject_parameters, paired_summary


class FrictionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_migration_preserves_both_branched_actions_and_critic_for_any_friction(self):
        torch.manual_seed(71)
        old = Agent(optimization_mode="joint")
        with torch.no_grad():
            old.strength_mean[-1].bias[4] = -.31
        stream = BytesIO()
        torch.save(old.state_dict(), stream)
        stream.seek(0)
        new = initialize_from_pretrained(Agent(optimization_mode="joint", observation_count=46), stream)
        obs = torch.randn(32, 45)
        for mu in (.1, .5, 1.4):
            augmented = torch.cat((obs, torch.full((len(obs),1),mu)), dim=-1)
            torch.testing.assert_close(new.get_action(augmented,deterministic=True),old.get_action(obs,deterministic=True),rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(new.get_value(augmented),old.get_value(obs),rtol=1e-5,atol=1e-6)
        for branch in (new.actor_mean,new.strength_mean,new.critic):
            self.assertEqual(branch[0].weight[:,45].count_nonzero().item(),0)

    def test_added_input_receives_policy_gradients(self):
        agent=Agent(optimization_mode="joint",observation_count=46)
        obs=torch.randn(16,46)
        obs[:,45]=torch.linspace(.2,1.,16)
        actions=agent.get_action(obs,deterministic=True).detach()+.25
        _,logp,_,value=agent.get_action_and_value(obs,actions)
        (-logp.mean()+value.square().mean()).backward()
        for branch in (agent.actor_mean,agent.strength_mean,agent.critic):
            self.assertGreater(branch[0].weight.grad[:,45].abs().sum().item(),0)

    def test_injection_preserves_other_sensors_and_original(self):
        obs=torch.randn(3,46)
        before=obs.clone()
        result=inject_parameters(obs,mass_kg=.12,friction=.7)
        torch.testing.assert_close(obs,before)
        keep=[i for i in range(46) if i not in (42,45)]
        torch.testing.assert_close(result[:,keep],obs[:,keep])
        torch.testing.assert_close(result[:,42],torch.full((3,),.12*9.81))
        torch.testing.assert_close(result[:,45],torch.full((3,),.7))
        with self.assertRaises(ValueError):
            inject_parameters(obs,friction=-.1)

    def test_grid_independently_crosses_every_mass_and_friction(self):
        settings,_,_=resolve_settings(dict(condition_on_friction=True,friction_profile="independent_grid",num_eval_envs=15))
        experiment=friction_experiment(settings,settings["masses_kg"])
        conditions=experiment["conditions"]
        self.assertEqual(len(conditions),15)
        for mass in settings["masses_kg"]:
            self.assertEqual({r["cube_static_friction"] for r in conditions if r["mass_kg"]==mass},{.2,.5,1.})
        self.assertFalse(experiment["hidden_from_policy"])
        self.assertFalse(experiment["changes_with_actual_mass"])
        with self.assertRaises(ValueError):
            resolve_settings(dict(condition_on_friction=True))
        with self.assertRaises(ValueError):
            resolve_settings(dict(condition_on_friction=True,friction_profile="independent_grid",num_eval_envs=10))

    def test_paired_metrics_match_scenes_and_do_not_confuse_ties_with_improvement(self):
        rows=[]
        for seed in range(3):
            for mode in ("true","wrong_friction"):
                rows.append(dict(seed=seed,mode=mode,mass_kg=.1,friction=.2,sustained_success=True))
        comparison=paired_summary(rows)["wrong_friction"]
        self.assertEqual(comparison["true_minus_comparison_success"],0)
        self.assertEqual(comparison["paired_bootstrap_95_interval"],[0,0])


if __name__=="__main__":
    unittest.main()
