"""Transactional PPO steps bounded against one fixed rollout behavior policy.

The constraint is the mean over ALL rollout observations of the analytic
KL(old Gaussian || candidate Gaussian), summed across five effective actions.
It concerns raw Gaussian actions before the simulator clips controller inputs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

OPTIMIZATION_SEMANTICS = 'joint_ppo_rollout_exact_kl_transactional_v1'


@torch.no_grad()
def effective_gaussian(agent, observations):
    """Use movement XYZ/opening and the independent strength branch, all five."""
    if getattr(agent, 'optimization_mode', None) != 'joint':
        raise ValueError('Exact five-action guarding requires joint optimization')
    mean = agent.action_mean(observations)
    logstd = torch.cat((agent.actor_logstd[:, :4], agent.strength_logstd), dim=-1).expand_as(mean)
    return mean.detach(), logstd.detach()


def diagonal_gaussian_kl(old_mean, old_logstd, new_mean, new_logstd):
    """Analytic KL(old||new), summed over actions for every observation."""
    old_mean, old_logstd, new_mean, new_logstd = [value.double() for value in (old_mean, old_logstd, new_mean, new_logstd)]
    variance_ratio = (2 * (old_logstd - new_logstd)).exp()
    squared_mean_shift = (old_mean - new_mean).square() * (-2 * new_logstd).exp()
    result = (new_logstd - old_logstd + .5 * (variance_ratio + squared_mean_shift - 1)).sum(-1)
    return result.clamp_min(0)


def _finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tensors(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


@dataclass(frozen=True)
class StepResult:
    accepted: bool
    trials: int
    backtracks: int
    exact_kl_mean: float
    exact_kl_state_max: float
    reason: str
    accepted_lr_scale: float | None


class RolloutKLGuard:
    """One comparator per rollout; rejected steps restore model AND Adam state.

    Every minibatch starts its bounded search at the configured rollout LR.
    Temporary trial LRs never persist. When no useful bounded step fits, the
    caller stops remaining updates; the next rollout starts with a fresh budget.
    """
    def __init__(self, agent, optimizer, observations, *, max_exact_kl=.05,
                 max_backtracks=6, backtrack_factor=.5, max_grad_norm=.5):
        if not math.isfinite(max_exact_kl) or max_exact_kl <= 0:
            raise ValueError('max_exact_kl must be positive and finite')
        if type(max_backtracks) is not int or max_backtracks < 0:
            raise ValueError('max_backtracks must be a nonnegative integer')
        if not math.isfinite(backtrack_factor) or not 0 < backtrack_factor < 1:
            raise ValueError('backtrack_factor must be between zero and one')
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError('max_grad_norm must be positive and finite')
        if not _finite_tensors(agent.state_dict()) or not _finite_tensors(optimizer.state_dict()):
            raise ValueError("Starting model and optimizer must be finite")
        self.agent, self.optimizer = agent, optimizer
        self.observations = observations.detach().clone()
        if self.observations.ndim != 2 or len(self.observations) == 0 or not _finite_tensors(self.observations):
            raise ValueError('Use a nonempty finite full rollout')
        mean, logstd = effective_gaussian(agent, self.observations)
        self.old_mean, self.old_logstd = mean.clone(), logstd.clone()
        if not _finite_tensors((self.old_mean, self.old_logstd)):
            raise ValueError('The behavior Gaussian is nonfinite')
        self.max_exact_kl = max_exact_kl
        self.max_backtracks = max_backtracks
        self.backtrack_factor = backtrack_factor
        self.max_grad_norm = max_grad_norm
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        if any(not math.isfinite(lr) or lr <= 0 for lr in self.base_lrs):
            raise ValueError('Configured trial learning rates must be positive and finite')
        self.stopped = False
        self.metrics = dict(accepted_updates=0, rejected_candidates=0, backtracked_candidates=0,
                            backtracked_updates=0, stopped_remaining_updates=False,
                            nonfinite_candidates=0, max_accepted_exact_kl=0.,
                            max_accepted_state_kl=0., max_candidate_exact_kl=0.,
                            min_accepted_lr_scale=None, configured_trial_learning_rates=list(self.base_lrs))

    @torch.no_grad()
    def exact_kl(self):
        mean, logstd = effective_gaussian(self.agent, self.observations)
        per_state = diagonal_gaussian_kl(self.old_mean, self.old_logstd, mean, logstd)
        if not _finite_tensors(per_state):
            return float('inf'), float('inf')
        return float(per_state.mean()), float(per_state.max())

    def _restore(self, model_state, optimizer_state):
        self.agent.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(deepcopy(optimizer_state))
        self.optimizer.zero_grad(set_to_none=True)

    def _restore_lrs(self):
        for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = lr

    def step(self, loss_closure):
        if self.stopped:
            raise RuntimeError('Rollout budget exhausted; begin a new rollout before more updates')
        model_state = {name: value.detach().clone() for name, value in self.agent.state_dict().items()}
        optimizer_state = deepcopy(self.optimizer.state_dict())
        accepted = False
        reason = 'no_bounded_candidate_fits'
        mean_kl, state_kl = self.exact_kl()
        trials = 0
        try:
            for trial in range(self.max_backtracks + 1):
                trials += 1
                if trial:
                    self._restore(model_state, optimizer_state)
                    self.metrics['backtracked_candidates'] += 1
                scale = self.backtrack_factor ** trial
                for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                    group['lr'] = lr * scale
                self.optimizer.zero_grad(set_to_none=True)
                loss = loss_closure()
                finite = isinstance(loss, torch.Tensor) and loss.numel() == 1 and bool(torch.isfinite(loss))
                if finite:
                    loss.backward()
                    finite = _finite_tensors([p.grad for p in self.agent.parameters() if p.grad is not None])
                if finite:
                    norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    finite = bool(torch.isfinite(norm))
                finite_before_step = finite
                if finite:
                    self.optimizer.step()
                    finite = _finite_tensors(self.agent.state_dict()) and _finite_tensors(self.optimizer.state_dict())
                mean_kl, state_kl = self.exact_kl() if finite else (float('inf'), float('inf'))
                if math.isfinite(mean_kl):
                    self.metrics['max_candidate_exact_kl'] = max(self.metrics['max_candidate_exact_kl'], mean_kl)
                if finite and mean_kl <= self.max_exact_kl:
                    accepted = True
                    self.metrics['accepted_updates'] += 1
                    self.metrics['backtracked_updates'] += int(trial > 0)
                    self.metrics['max_accepted_exact_kl'] = max(self.metrics['max_accepted_exact_kl'], mean_kl)
                    self.metrics['max_accepted_state_kl'] = max(self.metrics['max_accepted_state_kl'], state_kl)
                    previous = self.metrics['min_accepted_lr_scale']
                    self.metrics['min_accepted_lr_scale'] = scale if previous is None else min(previous, scale)
                    return StepResult(True, trials, trial, mean_kl, state_kl, 'accepted', scale)
                self.metrics['rejected_candidates'] += 1
                if not finite or not math.isfinite(mean_kl):
                    self.metrics['nonfinite_candidates'] += 1
                    reason = 'nonfinite_candidate'
                    # A nonfinite loss/gradient is identical at the restored state.
                    # A finite-gradient overflow after Adam may fit a smaller LR.
                    if not finite_before_step:
                        break
            self.stopped = True
            self.metrics['stopped_remaining_updates'] = True
            self._restore(model_state, optimizer_state)
            mean_kl, state_kl = self.exact_kl()
            return StepResult(False, trials, trials - 1, mean_kl, state_kl, reason, None)
        finally:
            if not accepted:
                # Includes closures/optimizers raising after mutating state.
                self._restore(model_state, optimizer_state)
            self._restore_lrs()


def guarded_ppo_update(agent, optimizer, observations, actions, old_logprobs,
                       advantages, returns, old_values, args, *, max_exact_kl=.05,
                       max_backtracks=6, backtrack_factor=.5):
    """Pinned PPO clipped losses with bounded, rollout-wide transactional steps."""
    guard = RolloutKLGuard(agent, optimizer, observations, max_exact_kl=max_exact_kl,
                          max_backtracks=max_backtracks, backtrack_factor=backtrack_factor,
                          max_grad_norm=args.max_grad_norm)
    indices = np.arange(len(observations))
    accepted_losses = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), args.minibatch_size):
            selected = indices[start:start + args.minibatch_size]
            minibatch_advantages = advantages[selected]
            if args.norm_adv:
                minibatch_advantages = ((minibatch_advantages - minibatch_advantages.mean()) /
                                        (minibatch_advantages.std() + 1e-8))
            captured = {}
            def loss_closure():
                _, new_logprob, entropy, new_value = agent.get_action_and_value(observations[selected], actions[selected])
                ratio = (new_logprob - old_logprobs[selected]).exp()
                pg_loss = torch.maximum(-minibatch_advantages * ratio,
                    -minibatch_advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                new_value = new_value.view(-1)
                value_error = (new_value - returns[selected]).square()
                if args.clip_vloss:
                    clipped = old_values[selected] + (new_value - old_values[selected]).clamp(-args.clip_coef, args.clip_coef)
                    value_error = torch.maximum(value_error, (clipped - returns[selected]).square())
                value_loss = .5 * value_error.mean()
                entropy_loss = entropy.mean()
                captured.update(policy_loss=float(pg_loss.detach()), value_loss=float(value_loss.detach()), entropy=float(entropy_loss.detach()))
                return pg_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss
            result = guard.step(loss_closure)
            if result.accepted:
                accepted_losses.append(dict(captured))
            else:
                break
        if guard.stopped:
            break
    with torch.no_grad():
        _, logprob, entropy, value = agent.get_action_and_value(observations, actions)
        logratio = logprob - old_logprobs
        ratio = logratio.exp()
        mean_kl, state_kl = guard.exact_kl()
        values = accepted_losses or [dict(policy_loss=0.,value_loss=float(.5*(value.flatten()-returns).square().mean()),entropy=float(entropy.mean()))]
        metrics = {name: float(np.mean([row[name] for row in values])) for name in ('policy_loss','value_loss','entropy')}
        metrics.update(guard.metrics, rollout_exact_kl=mean_kl, rollout_state_kl_max=state_kl,
                       old_approx_kl=float((-logratio).mean()), approx_kl=float(((ratio-1)-logratio).mean()),
                       clipfrac=float(((ratio-1).abs()>args.clip_coef).float().mean()),
                       max_exact_kl=max_exact_kl, rollout_observations=len(observations), effective_actions=5,
                       optimization_semantics=OPTIMIZATION_SEMANTICS)
    return metrics
