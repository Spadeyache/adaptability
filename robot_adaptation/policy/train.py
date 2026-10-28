"""THE TRAINING SCRIPT: edit SETTINGS; --prepare-only saves a setup without training.

simulation/training_lift.py defines the world; policy/model.py defines actor and critic.
The original ManiSkill PPO loop collects experience and adjusts the networks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import shlex
import sys
import uuid
import torch
from robot_adaptation.simulation.training_lift import ACTION_DESCRIPTION, ENV_ID, ENV_SETTINGS, HORIZON, INPUT_FIELDS, INPUT_SIZES, ORIGIN, LEGACY_ORIGIN, FRICTION_KEYS, friction_settings, friction_experiment, FRICTION_ORIGIN, FRICTION_INPUT_FIELDS, FRICTION_INPUT_SIZES
from robot_adaptation.policy.model import Agent, SOURCE_COMMIT, POLICY_ARCHITECTURE, FRICTION_POLICY_ARCHITECTURE, initialize_from_pretrained
from robot_adaptation.simulation.training_hold import HOLD_ENV_ID
from robot_adaptation.policy.trust_region import OPTIMIZATION_SEMANTICS

HERE = Path(__file__).resolve().parent
from robot_adaptation.paths import ROOT

SIMULATION_DIR = ROOT / "robot_adaptation" / "simulation"
SOURCE = HERE / "examples" / "maniskill_pick_cube" / "ppo_upstream.py"
SETTINGS = {
    "seed": 1,
    "condition_on_friction": False, # Explicit 46-input mode; preserves historical 45-input defaults.
    "training_stage": "carry_joint", # Learn movement/opening AND strength in the full picking task.
    # hold_strength and carry_strength retain the earlier strength-only training.
    "total_timesteps": 10_500_000, # 3,500 rollout/learning cycles; override with --iterations.
    "num_envs": 60,               # 12 robots at each of five physical masses.
    "num_steps": 50,              # 60 * 50 = 3,000 transitions before learning.
    "num_minibatches": 10,        # 300 transitions per optimizer step.
    "num_eval_envs": 10,          # Two separate evaluation robots per mass.
    "num_eval_steps": 50,
    "eval_freq": 25,
    "guard_policy_updates": False, # Optional exact rollout-wide Gaussian KL transactions.
    "max_exact_kl": 0.05,
    "guard_max_backtracks": 6,
    "guard_backtrack_factor": 0.5,
    "learning_rate": 1e-4,         # Smaller updates while extending the trained policy.
    "gamma": 0.8,
    "gae_lambda": 0.9,
    "update_epochs": 4,
    "clip_coef": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.1,
    "finite_horizon_gae": True,
    "partial_reset": False,      # Continue holding after reaching the target, until the time limit.
    "eval_partial_reset": False, # Check the entire attempt, including holding.
    "control_mode": "pd_ee_delta_pos",
    "sim_backend": "physx_cuda",
    "masses_kg": ENV_SETTINGS["masses_kg"],
    "goal_clearance_m": ENV_SETTINGS["goal_clearance_m"],
    "grip_force_limits_n": ENV_SETTINGS["grip_force_limits_n"],
    # Hidden simulator friction: heavier cubes are deliberately more slippery.
    **{key: ENV_SETTINGS[key] for key in FRICTION_KEYS},
    "grip_force_cost": 0.10,       # Trial: double the previous measured-force penalty (0.05).
    "initialize_from_pretrained": True, # Fallback for checkpoint=None; published 42-input policy.
    "checkpoint": "latest",      # Newest completed grip-aware or previous weighted policy.
    "run_root": str(ROOT / "runs" / "policy"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(path):
    path = Path(path).resolve()
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("This checkpoint has no weighted-run manifest; old custom checkpoints require a new run")
    saved = json.loads(manifest_path.read_text())
    conditioned = saved.get("artifact_origin") == FRICTION_ORIGIN
    expected_sizes = FRICTION_INPUT_SIZES if conditioned else INPUT_SIZES
    if (saved.get("artifact_origin") not in (ORIGIN, FRICTION_ORIGIN) or saved.get("input_sizes") != expected_sizes
            or saved.get("action_description") != ACTION_DESCRIPTION):
        raise ValueError("Choose a matching 45/46-input five-action grip-aware checkpoint; training converts previous weighted policies")
    mode = saved.get("optimization_mode", "strength_only")
    agent = initialize_from_pretrained(Agent(optimization_mode=mode, observation_count=sum(expected_sizes)), path)
    agent.eval()
    return agent, saved


def latest_checkpoint(allow_legacy=False, condition_on_friction=False):
    """Select completed main-training models; exclude check runs and unsupported layouts."""
    choices = sorted([*(ROOT / "runs" / "policy").glob("*/final_ckpt.pt"),
                      *(ROOT / "runs" / "learning").glob("*/final_ckpt.pt")],
                     key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in choices:
        manifest = path.parent / "manifest.json"
        if not manifest.exists():
            continue
        saved = json.loads(manifest.read_text())
        if saved.get("status") != "complete":
            continue
        current = (saved.get("artifact_origin") == ORIGIN and
                   saved.get("input_sizes") == INPUT_SIZES and
                   saved.get("action_description") == ACTION_DESCRIPTION)
        legacy = (allow_legacy and saved.get("artifact_origin") == LEGACY_ORIGIN and
                  saved.get("observations") == 43 and saved.get("actions") == 4)
        conditioned = (condition_on_friction and saved.get("artifact_origin") == FRICTION_ORIGIN and
                       saved.get("input_sizes") == FRICTION_INPUT_SIZES and
                       saved.get("action_description") == ACTION_DESCRIPTION)
        if conditioned or current or legacy:
            return path
    raise FileNotFoundError("No compatible completed training model. Set checkpoint=None to use the published policy.")


def resolve_settings(settings=None):
    """Validate without creating environments, run folders, or learning updates."""
    settings = {**SETTINGS, **(settings or {})}
    unknown = set(settings) - set(SETTINGS)
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    if settings['training_stage'] not in ('hold_strength', 'carry_strength', 'carry_joint'):
        raise ValueError('Choose hold_strength, carry_strength, or carry_joint')
    optimization_mode = 'joint' if settings['training_stage'] == 'carry_joint' else 'strength_only'
    if settings['training_stage'] == 'hold_strength' and (settings['partial_reset'] or settings['eval_partial_reset']):
        raise ValueError('Hold practice needs synchronized full resets: partial_reset=False and eval_partial_reset=False')
    settings.update(friction_settings(settings))
    if type(settings["condition_on_friction"]) is not bool:
        raise ValueError("condition_on_friction must be a boolean")
    if settings["condition_on_friction"] and settings["friction_profile"] != "independent_grid":
        raise ValueError("Friction conditioning requires independently crossed physics")
    if type(settings["guard_policy_updates"]) is not bool:
        raise ValueError("guard_policy_updates must be a boolean")
    if settings["guard_policy_updates"] and optimization_mode != "joint":
        raise ValueError("Five-action KL guarding requires carry_joint training")
    if type(settings["guard_max_backtracks"]) is not int or settings["guard_max_backtracks"] < 0:
        raise ValueError("guard_max_backtracks must be a nonnegative integer")
    factor = settings["guard_backtrack_factor"]
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or not 0 < factor < 1:
        raise ValueError("guard_backtrack_factor must be between zero and one")
    training_env_id = HOLD_ENV_ID if settings['training_stage'] == 'hold_strength' else ENV_ID
    for key in ("total_timesteps", "num_envs", "num_steps", "num_minibatches", "num_eval_envs", "num_eval_steps", "eval_freq", "update_epochs"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    batch = settings["num_envs"] * settings["num_steps"]
    if settings["total_timesteps"] % batch or batch % settings["num_minibatches"] or batch // settings["num_minibatches"] < 2:
        raise ValueError("Use whole rollouts and evenly divided minibatches with at least two samples")
    if settings["eval_freq"] < 2 or settings["num_eval_steps"] % HORIZON:
        raise ValueError("Use eval_freq >= 2 and complete 50-step evaluation episodes")
    if settings["control_mode"] != "pd_ee_delta_pos" or settings["sim_backend"] not in ("physx_cpu", "physx_cuda"):
        raise ValueError("Use the native arm controller with strength extension and CPU/CUDA physics")
    masses = settings["masses_kg"]
    if not masses or any(not isinstance(m, (int, float)) or not 0 < m < float("inf") for m in masses):
        raise ValueError("Choose positive finite masses")
    if settings["sim_backend"] == "physx_cpu":
        if settings["num_envs"] != 1 or settings["num_eval_envs"] != 1:
            raise ValueError("CPU physics supports one robot; its mass cycles at reset")
    elif settings["num_envs"] % len(masses) or settings["num_eval_envs"] % len(masses):
        raise ValueError(f"With {len(masses)} masses, num_envs ({settings['num_envs']}) and num_eval_envs ({settings['num_eval_envs']}) must both be multiples of {len(masses)}")
    if settings["sim_backend"] == "physx_cuda" and settings["friction_profile"] == "independent_grid":
        conditions = len(masses) * len(settings["cube_friction_coefficients"])
        if settings["num_envs"] % conditions or settings["num_eval_envs"] % conditions:
            raise ValueError(f"Independent physics requires training/eval environment counts divisible by {conditions}")
    if type(settings["seed"]) is not int or not 0 <= settings["seed"] < 2**31:
        raise ValueError("seed must be a nonnegative 31-bit integer")
    for key in ("learning_rate", "max_grad_norm", "target_kl", "max_exact_kl"):
        value = settings[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    return settings, training_env_id, optimization_mode



def _snapshot_sources(output):
    """Freeze runnable sources without depending on future package edits.

    The saved PPO runtime imports these flat modules from its own directory.
    Existing run folders remain immutable; only newly prepared runs use this.
    """
    sources = {
        "lift_task.py": SIMULATION_DIR / "training_lift.py",
        "hold_task.py": SIMULATION_DIR / "training_hold.py",
        "grip_controller.py": SIMULATION_DIR / "grip_controller.py",
        "ppo.py": HERE / "model.py",
        "train_ppo.py": HERE / "train.py",
        "trust_region.py": HERE / "trust_region.py",
    }
    # A frozen trainer may itself be used later without the source package.
    flat_names = {"training_lift.py": "lift_task.py", "training_hold.py": "hold_task.py",
                  "model.py": "ppo.py", "train.py": "train_ppo.py"}
    replacements = {
        "from robot_adaptation.simulation.training_lift import ": "from lift_task import ",
        "from robot_adaptation.simulation.training_hold import ": "from hold_task import ",
        "from robot_adaptation.simulation.grip_controller import ": "from grip_controller import ",
        "from robot_adaptation.policy.model import ": "from ppo import ",
        "from robot_adaptation.policy.trust_region import ": "from trust_region import ",
        "from robot_adaptation.paths import ROOT": "ROOT = Path(" + repr(str(ROOT)) + ")",
        'SIMULATION_DIR = ROOT / "robot_adaptation" / "simulation"': "SIMULATION_DIR = HERE",
    }
    for filename, source in sources.items():
        if not source.is_file():
            source = HERE / flat_names.get(source.name, source.name)
        frozen = source.read_text()
        for original, replacement in replacements.items():
            frozen = frozen.replace(original, replacement)
        (output / filename).write_text(frozen)


def main(settings=None, *, prepare_only=False):
    """Prepare reproducible sources/weights, then launch only when requested.

    prepare_only=True never creates a simulator or starts the PPO subprocess.
    Stored weights are initialization copies, not trained/updated checkpoints.
    """
    settings, training_env_id, optimization_mode = resolve_settings(settings)
    masses = settings["masses_kg"]
    conditioned = settings["condition_on_friction"]
    observation_count = 46 if conditioned else 45
    optimized_actions = (["x", "y", "z", "finger_opening", "grip_strength_limit"]
                         if optimization_mode == "joint" else ["grip_strength_limit"])
    frozen_actions = [] if optimization_mode == "joint" else ["x", "y", "z", "finger_opening"]
    torch.set_num_threads(4)
    torch.manual_seed(settings["seed"])
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output = Path(settings["run_root"]).resolve() / name
    output.mkdir(parents=True, exist_ok=False)
    # Freeze standalone flat modules for this run; package imports are rewritten
    # only in these generated snapshots, never in active source files.
    _snapshot_sources(output)
    pinned = output / "examples" / "maniskill_pick_cube"
    pinned.mkdir(parents=True)
    (pinned / "ppo_upstream.py").write_bytes(SOURCE.read_bytes())
    license_bytes = (HERE / "MANISKILL_LICENSE").read_bytes()
    (pinned / "MANISKILL_LICENSE").write_bytes(license_bytes)
    (output / "MANISKILL_LICENSE").write_bytes(license_bytes)
    original = SOURCE.read_text()
    old = 'env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")'
    replacement = 'env_kwargs = ' + repr(dict(obs_mode="state", render_mode="rgb_array", sim_backend=settings["sim_backend"],
        render_backend="cpu", reward_mode="normalized_dense", masses_kg=masses, goal_clearance_m=settings["goal_clearance_m"],
        grip_force_limits_n=settings["grip_force_limits_n"], grip_force_cost=settings["grip_force_cost"],
        **friction_settings(settings), condition_on_friction=conditioned,
        sensor_configs={"shader_pack": "minimal"}, human_render_camera_configs={"shader_pack": "minimal"}))
    if original.count(old) != 1:
        raise RuntimeError("Pinned upstream source changed")
    runtime = original.replace(old, replacement).replace("import mani_skill.envs\n", "import mani_skill.envs\nimport lift_task\nimport hold_task\n", 1)
    start, end = runtime.index("class Agent(nn.Module):"), runtime.index("class Logger:")
    runtime = runtime[:start] + "from ppo import Agent\n\n" + runtime[end:]
    agent_anchor = "agent = Agent(envs).to(device)"
    optimizer_anchor = "optim.Adam(agent.parameters(),"
    if runtime.count(agent_anchor) != 1 or runtime.count(optimizer_anchor) != 1:
        raise RuntimeError("Pinned PPO model/optimizer integration changed")
    runtime = runtime.replace(agent_anchor, f"agent = Agent(envs, optimization_mode={optimization_mode!r}).to(device)")
    runtime = runtime.replace(optimizer_anchor, "optim.Adam((p for p in agent.parameters() if p.requires_grad),")
    runtime = runtime.replace("import torch\n", "import torch\ntorch.set_num_threads(4)\n", 1)
    runtime = runtime.replace("eval_envs.step(agent.get_action(eval_obs, deterministic=True))", "eval_envs.step(clip_action(agent.get_action(eval_obs, deterministic=True)))")
    runtime = runtime.replace("runs/{run_name}", "{run_name}")
    runtime = runtime.replace("torch.load(args.checkpoint)", "torch.load(args.checkpoint, map_location=device, weights_only=True)")
    if settings["guard_policy_updates"]:
        runtime = runtime.replace("from ppo import Agent\n", "from ppo import Agent\nfrom trust_region import guarded_ppo_update\n", 1)
        runtime = runtime.replace("import os\n", "import os\nimport json\n", 1)
        update_start = runtime.index("        # Optimizing the policy and value network\n")
        update_end_anchor = "        update_time = time.time() - update_time\n"
        update_end = runtime.index(update_end_anchor, update_start) + len(update_end_anchor)
        guarded = """        # Exact KL uses one immutable behavior comparator across this full rollout.
        agent.train()
        update_time = time.time()
        guard_metrics = guarded_ppo_update(agent, optimizer, b_obs, b_actions, b_logprobs,
            b_advantages, b_returns, b_values, args,
            max_exact_kl=MAX_KL, max_backtracks=MAX_BACKTRACKS, backtrack_factor=BACKTRACK_FACTOR)
        v_loss = torch.as_tensor(guard_metrics["value_loss"])
        pg_loss = torch.as_tensor(guard_metrics["policy_loss"])
        entropy_loss = torch.as_tensor(guard_metrics["entropy"])
        old_approx_kl = torch.as_tensor(guard_metrics["old_approx_kl"])
        approx_kl = torch.as_tensor(guard_metrics["approx_kl"])
        clipfracs = [guard_metrics["clipfrac"]]
        with open(f"{run_name}/trust_region.jsonl", "a") as guard_log:
            guard_log.write(json.dumps(dict(iteration=iteration, global_step=global_step, **guard_metrics), allow_nan=False) + "\\n")
        print("Exact rollout KL:", guard_metrics["rollout_exact_kl"], "accepted:", guard_metrics["accepted_updates"],
              "rejected:", guard_metrics["rejected_candidates"], "backtracked:", guard_metrics["backtracked_updates"], flush=True)
        for metric_name in ("rollout_exact_kl", "rollout_state_kl_max", "accepted_updates", "rejected_candidates", "backtracked_updates"):
            logger.add_scalar("trust_region/" + metric_name, guard_metrics[metric_name], global_step)
        update_time = time.time() - update_time
"""
        guarded = guarded.replace("MAX_KL", repr(settings["max_exact_kl"])).replace("MAX_BACKTRACKS", repr(settings["guard_max_backtracks"])).replace("BACKTRACK_FACTOR", repr(settings["guard_backtrack_factor"]))
        runtime = runtime[:update_start] + guarded + runtime[update_end:]
    (output / "ppo_runtime.py").write_text(runtime)
    initialization = dict(kind="random")
    checkpoint = settings["checkpoint"]
    if checkpoint:
        source_checkpoint = latest_checkpoint(allow_legacy=True, condition_on_friction=conditioned) if checkpoint == "latest" else Path(checkpoint).resolve()
        previous = json.loads((source_checkpoint.parent / "manifest.json").read_text())
        if previous.get("artifact_origin") not in (ORIGIN, LEGACY_ORIGIN, FRICTION_ORIGIN):
            raise ValueError("Choose a grip-aware or previous weighted training checkpoint")
        agent = initialize_from_pretrained(Agent(optimization_mode=optimization_mode, observation_count=observation_count), source_checkpoint)
        initialization = dict(kind="latest_trained_policy" if settings["checkpoint"] == "latest" else "trained_policy",
                              checkpoint=str(source_checkpoint), sha256=digest(source_checkpoint),
                              source_observations=previous.get("observations"), source_actions=previous.get("actions"),
                              source_settings=previous.get("settings"), optimizer_restarted=True,
                              source_policy_architecture=previous.get("policy_architecture", "shared_actor"),
                              branch_initialization="retain existing branches, or copy a shared actor into independent movement and strength branches",
                              optimization_mode=optimization_mode,
                              added_force_columns="zero when extending 43 inputs", added_strength_action="maximum when extending four actions")
    elif settings["initialize_from_pretrained"]:
        sys.path.insert(0, str(SOURCE.parent))
        from run_pretrained import download_model
        published, _, url = download_model("pd_ee_delta_pos")
        agent = initialize_from_pretrained(Agent(optimization_mode=optimization_mode, observation_count=observation_count), published)
        initialization = dict(kind="published_42_input_policy", checkpoint=str(published), url=url,
                              sha256=digest(published), added_weight_and_force_columns="zero", added_strength_action="maximum", weight_input_initially_used=False)
    else:
        agent = Agent(optimization_mode=optimization_mode, observation_count=observation_count)
    if conditioned:
        initialization["added_friction_column"] = dict(index=45, initialization="retained" if initialization.get("source_observations") == 46 else "zero" if settings["checkpoint"] or settings["initialize_from_pretrained"] else "random")
    # Always save the exact starting weights, including random initialization.
    # A later --from-prepared run does not depend on a moving latest selection.
    checkpoint = str(output / "initialization.pt")
    torch.save(agent.state_dict(), checkpoint)
    initialization["snapshot_sha256"] = digest(checkpoint)
    manifest = dict(artifact_origin=FRICTION_ORIGIN if conditioned else ORIGIN, environment=training_env_id, evaluation_environment=training_env_id, deployment_environment=ENV_ID, observations=observation_count, actions=5,
                    policy_architecture=FRICTION_POLICY_ARCHITECTURE if conditioned else POLICY_ARCHITECTURE, optimization_mode=optimization_mode,
                    optimized_actions=optimized_actions, frozen_actions=frozen_actions,
                    planned_iterations=settings["total_timesteps"] // (settings["num_envs"] * settings["num_steps"]),
                    optimization_semantics=OPTIMIZATION_SEMANTICS if settings["guard_policy_updates"] else "pinned_upstream_clipped_ppo_sampled_kl_stop",
                    trust_region=dict(enabled=settings["guard_policy_updates"], max_exact_kl=settings["max_exact_kl"],
                                      criterion="full rollout mean KL(old Gaussian || candidate Gaussian), summed over five effective raw actions",
                                      comparator="fixed rollout behavior, cached once before updates", max_backtracks=settings["guard_max_backtracks"],
                                      backtrack_factor=settings["guard_backtrack_factor"], rollback="all model state and Adam moments/step counters",
                                      learning_rate_policy="temporary bounded trials; restore configured LR; stop remaining updates on exhaustion; reset budget next rollout"),
                    trust_region_sha256=digest(output / "trust_region.py"),
                    ppo_likelihood="joint five-action Gaussian" if optimization_mode == "joint" else "strength-only Gaussian",
                    action_storage="unclipped sampled actions with matching log probabilities; clipping only at simulator application",
                    action_description=ACTION_DESCRIPTION, input_fields=FRICTION_INPUT_FIELDS if conditioned else INPUT_FIELDS, input_sizes=FRICTION_INPUT_SIZES if conditioned else INPUT_SIZES,
                    settings=settings, horizon=HORIZON, initialization=initialization,
                    conditioning_parameters=(dict(object_weight_n=dict(index=42, units="newtons", source="true mass times gravity during training"),
                                                  object_friction_coefficient=dict(index=45, units="dimensionless", definition="object material coefficient; static=dynamic, fixed finger material")) if conditioned else None),
                    source_commit=SOURCE_COMMIT, source_sha256=digest(SOURCE),
                    task_sha256=digest(output / "lift_task.py"), model_sha256=digest(output / "ppo.py"), controller_sha256=digest(output / "grip_controller.py"),
                    versions={p: version(p) for p in ("torch", "mani_skill", "sapien")},
                    differences_from_example=["floating goal range", "varied physical masses/inertias", "true weight and measured finger forces appended to 42 inputs",
                                              "independent per-finger motor strength action", "held-goal success and measured-force cost",
                                              "smaller balanced parallel batch", "warm start from latest trained policy", "full-episode holding during training",
                                              "both independent actor branches trainable" if optimization_mode == "joint" else "independent strength branch and frozen movement",
                                              "joint five-action PPO likelihood" if optimization_mode == "joint" else "strength-only PPO likelihood",
                                              "explicit supplied object friction and independently crossed physics" if conditioned else "explicitly recorded hidden friction profile",
                                              "physical hold preparation" if settings["training_stage"] == "hold_strength" else "full pickup and carrying"],
                    friction_experiment=friction_experiment(settings, masses),
                    training_stage=settings["training_stage"],
                    checkpoint_contents="model weights only; continuation restarts Adam and simulation", status="prepared")
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    command = [sys.executable, "-u", str(output / "ppo_runtime.py"), "--exp-name", str(output), "--env-id", training_env_id, "--no-capture-video", "--no-track"]
    excluded = set(FRICTION_KEYS) | {"guard_policy_updates", "max_exact_kl", "guard_max_backtracks", "guard_backtrack_factor", "condition_on_friction", "training_stage", "run_root", "masses_kg", "goal_clearance_m", "initialize_from_pretrained", "checkpoint", "sim_backend", "grip_force_limits_n", "grip_force_cost"}
    for key, value in settings.items():
        if key in excluded:
            continue
        flag = key.replace("_", "-")
        command.extend(["--" + ("" if value else "no-") + flag] if isinstance(value, bool) else ["--" + flag, str(value)])
    if checkpoint:
        command.extend(["--checkpoint", checkpoint])
    if settings["sim_backend"] == "physx_cpu":
        # Rebuild CPU scenes at reset to clear stale pairwise-contact caches.
        command.extend(["--no-cuda", "--reconfiguration-freq", "1"])
    child_env = dict(os.environ)
    child_env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + child_env.get("LD_LIBRARY_PATH", "")
    launch = dict(command=command, cwd=str(ROOT), environment_overrides={"LD_LIBRARY_PATH": child_env["LD_LIBRARY_PATH"]})
    (output / "launch_command.json").write_text(json.dumps(launch, indent=2) + "\n")
    print("Policy setup (" + settings["training_stage"] + ", " + optimization_mode + "):", output, flush=True)
    print("Optimized actions:", ", ".join(optimized_actions), flush=True)
    print("Planned rollout/learning cycles:", manifest["planned_iterations"], flush=True)
    print("Initial checkpoint:", initialization.get("checkpoint", "random"), flush=True)
    print("Friction conditions:", manifest["friction_experiment"], flush=True)
    if prepare_only:
        print("PREPARED ONLY: no simulation, PPO updates, or training subprocess started.", flush=True)
        print("To train later with these settings and source weights: " + shlex.join([
            sys.executable, "-m", "robot_adaptation.policy.train", "--from-prepared", str(output)]), flush=True)
        return output
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        with (output / "console.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        manifest.update(status="complete" if code == 0 else "failed", exit_code=code)
        if code:
            raise RuntimeError("Training failed; see " + str(output / "console.log"))
    except BaseException:
        if manifest["status"] == "running":
            manifest["status"] = "interrupted_or_failed"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Save validated settings, runtime and initialization without starting training")
    parser.add_argument("--iterations", type=int, help="Number of rollout/learning cycles (not PPO minibatch epochs)")
    parser.add_argument("--condition-on-friction", action="store_true", help="Train with 46 observations and independently crossed mass/friction")
    parser.add_argument("--guard-policy-updates", action="store_true", help="Bound full-rollout exact Gaussian KL with transactional backtracking")
    parser.add_argument("--max-exact-kl", type=float)
    parser.add_argument("--guard-max-backtracks", type=int)
    parser.add_argument("--guard-backtrack-factor", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "physx_cuda"))
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--num-eval-envs", type=int)
    parser.add_argument("--checkpoint", help="Existing model checkpoint path, or latest")
    parser.add_argument("--from-prepared", type=Path, help="Use saved settings and pin the source weights from an earlier preparation; creates a new run")
    args = parser.parse_args(argv)
    settings = dict(SETTINGS)
    if args.from_prepared:
        prepared = json.loads((args.from_prepared / "manifest.json").read_text())
        if prepared.get("artifact_origin") not in (ORIGIN, FRICTION_ORIGIN) or prepared.get("status") != "prepared":
            parser.error("--from-prepared requires a prepared grip-policy run")
        settings = prepared["settings"]
        snapshot = args.from_prepared.resolve() / "initialization.pt"
        expected_hash = prepared["initialization"].get("snapshot_sha256")
        if not snapshot.is_file() or expected_hash is None or digest(snapshot) != expected_hash:
            parser.error("Prepared initialization snapshot is missing or changed; prepare again")
        settings["checkpoint"] = str(snapshot)
    if args.condition_on_friction:
        settings.update(condition_on_friction=True, friction_profile="independent_grid", num_eval_envs=15)
    if args.guard_policy_updates:
        settings["guard_policy_updates"] = True
    for key in ("sim_backend", "num_envs", "num_eval_envs", "learning_rate", "max_exact_kl", "guard_max_backtracks", "guard_backtrack_factor"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if args.checkpoint:
        settings["checkpoint"] = args.checkpoint
    if args.iterations is not None:
        if args.iterations <= 0:
            parser.error("--iterations must be positive")
        settings["total_timesteps"] = args.iterations * settings["num_envs"] * settings["num_steps"]
    return main(settings, prepare_only=args.prepare_only)


if __name__ == "__main__":
    cli()
