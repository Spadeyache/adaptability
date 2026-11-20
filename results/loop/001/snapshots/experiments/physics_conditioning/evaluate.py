"""Evaluate fixed policies with matched physical scenes and changed supplied inputs.

This is inference only. Never interpret input sensitivity as task benefit.
Success is measured at the final step and sustained across the final ten steps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from robot_adaptation.paths import ROOT
from robot_adaptation.policy.train import load_checkpoint
from robot_adaptation.simulation.training_lift import LiftTask, normal_forces, read_inputs

MODES = ("true", "fixed_friction", "wrong_friction", "fixed_mass", "wrong_mass", "fixed_both")


def inject_parameters(vector, mass_kg=None, friction=None, gravity=9.81):
    """Replace supplied estimates only; the physical environment stays unchanged."""
    if vector.ndim != 2 or vector.shape[1] != 46 or not torch.isfinite(vector).all():
        raise ValueError("Expected finite [N,46] physics-conditioned observations")
    result = vector.clone()
    for index, value, scale in ((42, mass_kg, gravity), (45, friction, 1.0)):
        if value is not None:
            value = torch.as_tensor(value, device=vector.device, dtype=vector.dtype)
            if not torch.isfinite(value).all() or torch.any(value <= 0):
                raise ValueError("Parameter estimates must be finite and positive")
            result[:, index] = value * scale
    return result


def paired_summary(rows, bootstrap_seed=73517, metric="sustained_success"):
    """Paired bootstrap of matched (mass,friction,seed) starts; descriptive CI."""
    true = {(r["mass_kg"], r["friction"], r["seed"]): r for r in rows if r["mode"] == "true"}
    rng = np.random.default_rng(bootstrap_seed)
    results = {}
    for mode in MODES[1:]:
        other = {(r["mass_kg"], r["friction"], r["seed"]): r for r in rows if r["mode"] == mode}
        keys = sorted(set(true) & set(other))
        if not keys:
            continue
        differences = np.asarray([float(true[k][metric]) - float(other[k][metric]) for k in keys])
        # The entire physical grid shares a seeded batch: resample whole seed blocks.
        block_seeds = sorted({key[2] for key in keys})
        block_means = np.asarray([differences[[i for i,key in enumerate(keys) if key[2] == seed]].mean() for seed in block_seeds])
        samples = block_means[rng.integers(len(block_seeds), size=(10000, len(block_seeds)))].mean(axis=1)
        results[mode] = dict(metric=metric, bootstrap_samples=10000, n_pairs=len(keys), n_seed_blocks=len(block_seeds), true_minus_comparison_success=float(differences.mean()),
                             paired_bootstrap_95_interval=np.quantile(samples, [.025, .975]).tolist(),
                             true_only_successes=int((differences > 0).sum()), comparison_only_successes=int((differences < 0).sum()))
    return results


def evaluate(checkpoint, output=None, *, seeds=(91000, 91001, 91002, 91003, 91004),
             masses=(.04, .064, .1, .25, .5), frictions=(.2, .5, 1.),
             modes=MODES, horizon=50, backend="physx_cuda"):
    if not seeds or horizon < 10 or any(mode not in MODES for mode in modes):
        raise ValueError("Use held-out seeds, >=10 steps, and supported input modes")
    checkpoint = Path(checkpoint).resolve()
    agent, manifest = load_checkpoint(checkpoint)
    if manifest.get("observations") != 46:
        raise ValueError("This evaluator requires the explicit 46-input policy")
    saved = manifest["settings"]
    if backend == "physx_cpu":
        raise ValueError("Use GPU crossed evaluation; CPU smoke checks are available in tests")
    output = Path(output or ROOT / "runs" / "physics_conditioning" / ("evaluation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")))
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = LiftTask(num_envs=len(masses)*len(frictions), masses_kg=list(masses),
                   condition_on_friction=True, friction_profile="independent_grid", cube_friction_coefficients=list(frictions),
                   finger_friction_coefficient=saved["finger_friction_coefficient"],
                   goal_clearance_m=saved["goal_clearance_m"], grip_force_limits_n=saved["grip_force_limits_n"],
                   grip_force_cost=saved["grip_force_cost"], sim_backend=backend, render_backend="cpu")
    agent = agent.to(env.device).eval().requires_grad_(False)
    gravity = float(torch.as_tensor(env.scene.px.get_config().gravity).norm())
    actual_mass = np.tile(masses, len(frictions))
    actual_mu = np.repeat(frictions, len(masses))
    fixed_material_readbacks = env.friction_details()
    physical_readbacks = []
    for index, body in enumerate(env.cube._bodies):
        material = body.get_collision_shapes()[0].physical_material
        actual = dict(mass_kg=float(body.mass), static_friction=float(material.static_friction), dynamic_friction=float(material.dynamic_friction))
        if not np.allclose([actual["mass_kg"], actual["static_friction"], actual["dynamic_friction"]], [actual_mass[index], actual_mu[index], actual_mu[index]]):
            env.close()
            raise RuntimeError("Physical grid readback differs from evaluation labels")
        physical_readbacks.append(actual)
    rows, traces, sensitivity, strength_direction = [], [], [], []
    reset_fingerprints = {}
    try:
        for seed in seeds:
            for mode in modes:
                obs, _ = env.reset(seed=int(seed), options={"reconfigure": True})
                _, initial = read_inputs(env, obs)
                state = torch.cat((initial, env.cube.linear_velocity, env.cube.angular_velocity), dim=-1)
                fingerprint = hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()
                previous = reset_fingerprints.setdefault(int(seed), fingerprint)
                if fingerprint != previous:
                    raise RuntimeError("Matched input intervention starts from different physical state")
                rewards = np.zeros(env.num_envs)
                force_sum = np.zeros(env.num_envs)
                limit_sum = np.zeros(env.num_envs)
                lifted_steps = np.zeros(env.num_envs, dtype=int)
                had_lift = np.zeros(env.num_envs, dtype=bool)
                lost = np.zeros(env.num_envs, dtype=bool)
                successes = []
                for step in range(horizon):
                    _, vector = read_inputs(env, obs)
                    mass = .04 if mode in ("fixed_mass", "fixed_both") else None
                    mu = .5 if mode in ("fixed_friction", "fixed_both") else None
                    if mode == "wrong_mass":
                        mass = torch.as_tensor(.54 - actual_mass, device=env.device)
                    if mode == "wrong_friction":
                        mu = torch.as_tensor(1.2 - actual_mu, device=env.device)
                    supplied = inject_parameters(vector, mass, mu, gravity)
                    with torch.no_grad():
                        raw_action = agent.get_action(supplied, deterministic=True)
                        if mode == "true" and step in (0, 15, 30, 45):
                            low = agent.get_action(inject_parameters(vector, friction=.2), deterministic=True).clamp(-1, 1)
                            high = agent.get_action(inject_parameters(vector, friction=1.), deterministic=True).clamp(-1, 1)
                            sensitivity.append(float((high-low).abs().mean().item()))
                            held = vector[:,18] > .5
                            if held.any():
                                strength_direction.append(float((low[held,4]-high[held,4]).mean().item()))
                    obs, reward, _, _, info = env.step(raw_action.clamp(-1, 1))
                    grasped = info["is_grasped"].detach().cpu().numpy().astype(bool)
                    lifted = grasped & (info["clearance_m"].detach().cpu().numpy() > .03)
                    lost |= had_lift & ~grasped
                    had_lift |= lifted
                    lifted_steps += lifted
                    forces = normal_forces(env).mean(-1).detach().cpu().numpy()
                    limits = info["grip_limit_n"].detach().cpu().numpy()
                    force_sum += forces*lifted
                    limit_sum += limits*lifted
                    rewards += reward.detach().cpu().numpy()
                    successes.append(info["success"].detach().cpu().numpy().astype(bool))
                    if seed == seeds[0] and mode in ("true", "wrong_friction"):
                        traces.append(dict(seed=int(seed), mode=mode, step=step, observation=vector.detach().cpu().tolist(),
                                           supplied=supplied.detach().cpu().tolist(), action=raw_action.detach().cpu().tolist(),
                                           grasped=grasped.tolist(), success=successes[-1].tolist(), forces_n=forces.tolist()))
                final = successes[-1]
                sustained = np.all(np.stack(successes[-10:]), axis=0)
                for i in range(env.num_envs):
                    rows.append(dict(seed=int(seed), mode=mode, mass_kg=float(actual_mass[i]), friction=float(actual_mu[i]),
                                     final_success=bool(final[i]), sustained_success=bool(sustained[i]),
                                     lifted_ever=bool(had_lift[i]), grasp_lost_after_lift=bool(lost[i]), reward_sum=float(rewards[i]),
                                     lifted_steps=int(lifted_steps[i]),
                                     lifted_force_n=float(force_sum[i]/lifted_steps[i]) if lifted_steps[i] else None,
                                     lifted_motor_limit_n=float(limit_sum[i]/lifted_steps[i]) if lifted_steps[i] else None))
                print(json.dumps(dict(seed=seed, mode=mode, final_success=int(final.sum()), sustained_success=int(sustained.sum()), n=env.num_envs)), flush=True)
    finally:
        env.close()
    grouped = {}
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode]
        grouped[mode] = {key: float(np.mean([r[key] for r in selected])) for key in ("final_success", "sustained_success", "lifted_ever", "grasp_lost_after_lift", "reward_sum")}
        for force_key in ("lifted_force_n", "lifted_motor_limit_n"):
            values = [r[force_key] for r in selected if r[force_key] is not None]
            grouped[mode][force_key] = float(np.mean(values)) if values else None
        grouped[mode]["episodes_with_lifted_contact"] = sum(r["lifted_steps"] > 0 for r in selected)
    by_physics = []
    for mode in modes:
        for mass in masses:
            for mu in frictions:
                selected = [r for r in rows if r["mode"] == mode and np.isclose(r["mass_kg"], mass) and np.isclose(r["friction"], mu)]
                by_physics.append(dict(mode=mode, mass_kg=float(mass), friction=float(mu), n=len(selected),
                                       final_success=float(np.mean([r["final_success"] for r in selected])),
                                       sustained_success=float(np.mean([r["sustained_success"] for r in selected]))))
    paired = paired_summary(rows, metric="final_success")
    paired_sustained = paired_summary(rows, metric="sustained_success")
    friction_benefit = paired.get("wrong_friction", {})
    interval = friction_benefit.get("paired_bootstrap_95_interval", [-1, 1])
    conclusion = "supported_on_tested_cases" if interval[0] > 0 else "inconclusive"
    summary = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   source_manifest=manifest, physical_readbacks=physical_readbacks, fixed_material_readbacks=fixed_material_readbacks, reset_fingerprints=reset_fingerprints, seeds=list(seeds), masses_kg=list(masses), frictions=list(frictions), horizon=horizon,
                   metrics=grouped, metrics_by_physics=by_physics, paired_comparisons=paired, paired_sustained_comparisons=paired_sustained, friction_input_sensitivity_mean_abs_action=float(np.mean(sensitivity)),
                   counterfactual_low_minus_high_friction_strength_action=(float(np.mean(strength_direction)) if strength_direction else None),
                   strength_direction_definition="On grasped states, supplied mu0.2 strength minus supplied mu1.0 strength; positive means lower estimated friction requests more motor force. Secondary diagnostic only.",
                   accurate_friction_improves_final_success=conclusion,
                   limitation="Matched seed-block bootstrap across tested starts; static=dynamic, fixed fingers/table, simulation state observations. Input sensitivity alone does not show benefit.",
                   elapsed_seconds=time.monotonic()-started)
    (output / "episodes.json").write_text(json.dumps(rows, indent=2)+"\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    with (output / "traces.jsonl").open("w") as f:
        for row in traces:
            f.write(json.dumps(row)+"\n")
    print("Evaluation saved:", output, flush=True)
    return summary, output


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=91000)
    parser.add_argument("--horizon", type=int, default=50)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output, seeds=tuple(range(args.seed_start,args.seed_start+args.seed_count)), horizon=args.horizon)


if __name__ == "__main__":
    cli()
