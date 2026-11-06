import unittest
import torch
from robot_adaptation.policy.rma.model import Agent, Adapter, History, split_observation


class RMATest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)

    def test_no_privileged_leakage(self):
        obs = torch.randn(2,46)
        changed = obs.clone(); changed[:,42] += 100; changed[:,45] += 10
        state, physical = split_observation(obs)
        other, other_physical = split_observation(changed)
        torch.testing.assert_close(state, other)
        self.assertFalse(torch.equal(physical, other_physical))
        adapter = Adapter(); h = History(2)
        pred = adapter(h.append(state, torch.zeros(2,5)))
        h.reset(); altered = adapter(h.append(other, torch.zeros(2,5)))
        torch.testing.assert_close(pred, altered)

    def test_history_partial_reset_and_alignment(self):
        h = History(2,3)
        h.append(torch.ones(2,44), torch.full((2,5), .25))
        self.assertEqual(h.data[0,-1,-1], 1)
        torch.testing.assert_close(h.data[0,-1,44:49], torch.full((5,),.25))
        h.reset(torch.tensor([True,False]))
        self.assertEqual(h.data[0].sum(), 0)
        self.assertGreater(h.data[1].sum(), 0)

    def test_joint_gradient_and_adapter_freeze(self):
        policy = Agent()
        obs = torch.randn(4,46)
        action, logp, _, value = policy.get_action_and_value(obs)
        (logp.mean()+value.mean()).backward()
        self.assertGreater(sum(p.grad.abs().sum() for p in policy.encoder.parameters()),0)
        policy.requires_grad_(False)
        before = {k:v.clone() for k,v in policy.state_dict().items()}
        adapter = Adapter(); optimizer = torch.optim.Adam(adapter.parameters())
        history = History(4); state, physical = split_observation(obs)
        for _ in range(3):
            loss = (adapter(history.append(state, action.detach()))-policy.encode(physical).detach()).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        for k,v in policy.state_dict().items():
            torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
