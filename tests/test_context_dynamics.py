"""Regressions for native controller parity, command delay and hidden-data boundaries."""
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.context_dynamics.collect import policy_obs, task_index
from robot_adaptation.simulation.context_dynamics import make_env, PandaMotion
from robot_adaptation.estimators.context_dynamics.model import rotation_columns


class ContextDynamicsTest(unittest.TestCase):
    def test_task_dispatch(self):
        for i in range(12):
            self.assertEqual(task_index({'motion_variant':'grip_probe'},i,False),3)
            self.assertEqual(task_index({},i,False),i%4)
            self.assertEqual(task_index({},i,True),0)
            paired = {'motion_variant':'paired'}
            self.assertEqual(task_index(paired,2*i,False),task_index(paired,2*i+1,False))

    def test_cpu_scene_and_delay(self):
        cfg = json.loads(Path('configs/context_dynamics.json').read_text())
        limits = [.25,40.]
        env = make_env('PickCube-v1',cfg,limits)
        self.addCleanup(env.close)
        arm = PandaMotion(env,limits,cfg['finger_friction'])
        nominal = np.array([.1,.4,.3,1.,1.,1.,0,0.,0.,0.])
        cmd = np.array([.13,-.1,-.2,-.05,.9],np.float32)
        arm.set_episode(nominal,123)
        initial = arm.state()
        target = env.get_obs()[0].numpy()
        np.testing.assert_allclose(policy_obs(arm,initial,target[42],env.goal_site.pose.p[0].numpy(),
                                              True),target,atol=1e-6)
        env._step_action(torch.from_numpy(cmd[None]))
        canonical = arm.state()
        arm.set_episode(nominal,123)
        arm.step(cmd)
        np.testing.assert_allclose(arm.state(),canonical,atol=1.5e-4)
        varied = np.array([1.,.1,1.2,1.5,.5,.5,3,.01,-.01,.01])
        arm.set_episode(varied,123)
        np.testing.assert_allclose(arm.state(),initial,atol=1e-6)
        for _ in range(3):
            np.testing.assert_array_equal(arm.step(cmd),np.zeros(5))
        np.testing.assert_array_equal(arm.step(np.zeros(5)),cmd)
        expected = (.25+(cmd[4]+1)*.5*(40.-.25))*1.5
        self.assertAlmostEqual(arm.fingers[0].force_limit,expected,places=4)

    def test_orientation_sign(self):
        q = torch.randn(7,4)
        torch.testing.assert_close(rotation_columns(q),rotation_columns(-q))
