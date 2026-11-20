"""Immutable experiment definitions and split, created once before collection."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = [101, 211, 307, 401, 503]
PROTOCOL = {
    "version": 1,
    "train_cubes": 200, "test_cubes": 30, "out_of_range_cubes": 30,
    "episodes_per_cube": 10,
    "train_episodes": list(range(8)), "validation_episodes": [8, 9],
    "fit_episodes": list(range(5)), "final_score_episodes": list(range(5, 10)),
    "seeds": SEEDS, "iteration_budget": 20,
    "target_state_columns": list(range(48)),
    "normalization": "coordinate mean and population std of training states only; std floor 0.001 in each native unit; actions separately standardized, same floor",
    "model_target": "(next_state - state) / training_state_std",
    "moving": "norm(next cube position - cube position) > 0.0001 m OR relative orientation angle > 0.001 rad",
    "one_step_nmse": "mean squared normalized next-state error over all 48 coordinates, first average moving transitions within each cube, then equal-weight cubes",
    "r2": "1 - sum normalized squared next-state residual / sum normalized squared deviation from scoring-set mean next-state; all moving transitions pooled; undefined if denominator zero",
    "rollout_error": "10-step recursive rollout; average normalized squared state error across all ten predicted times and 48 coordinates; start at every eligible moving transition with ten remaining actions; equal-weight cubes",
    "success_metric": "one_step_nmse; R2 and 10-step rollout always reported, never substituted for selection",
    "std": "sample standard deviation across five seed-level aggregate scores (ddof=1)",
    "S1": "(Plain - fitted)/(Plain - Oracle) >= 0.5, with positive Plain-Oracle gap",
    "S2": "fitted mean + fitted std < shuffled mean - shuffled std",
    "S3": "fitted mean <= 1.1 * History mean",
    "S4": "F1-F9 pass, including no detected dimension-related memorization",
    "F2": "each of five held-out-cube action-only probe seeds has R2 < 0.1 for each property",
    "F4": "fitted z validation mean+std < shuffled mean-std, and fitted mean < mean-z mean",
    "F5": "on a dimension increase, flag jointly learned-z validation NMSE improvement paired with worsening fresh-fitted validation NMSE; uses validation only, not held-out test",
    "history_K": 20,
    "train_updates": 2000, "batch_size": 256,
    "default_width": 128, "default_depth": 2,
    "default_learning_rate": 0.001,
    "default_z_dim": 2, "default_z_l2": 0.0001,
    "default_fit_steps": 200, "default_fit_lr": 0.02,
    "default_moving_loss_weight": 1.0,
    "default_prediction_horizon": 1,
    "probe_updates": 400,
    "pilot_cube_ids": list(range(10)),
    "final_evaluation": "one sealed candidate, all seeds and conditions together; append begin record before opening held-out episodes; never tune afterwards",
    "shuffled_z": "seeded random derangement of fitted z within the scored cube group",
    "another_cube_z": "learned embedding of predetermined different training cube (cube_id+1) modulo 200; no property-based matching",
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")

def initialize():
    data = ROOT / "data"
    seal = data / "split_seal.json"
    if seal.exists():
        verify()
        return
    for name in ["split.json", "cubes.json", "protocol.json"]:
        if (data / name).exists():
            raise RuntimeError(f"Unsealed existing {name}; do not regenerate")
    split = {"seed": 2026111507,
        "train": list(range(200)), "test": list(range(200, 230)),
        "out_of_range": list(range(230, 260)),
        "training_episodes": list(range(8)), "validation_episodes": [8, 9],
        "fit_episodes": list(range(5)), "final_score_episodes": list(range(5, 10))}
    cubes = []
    for cube_id in range(260):
        rng = np.random.default_rng([split["seed"], cube_id, 0])
        group = "train" if cube_id < 200 else "test" if cube_id < 230 else "out_of_range"
        lo, hi = (.55, .90) if group == "out_of_range" else (.04, .50)
        cubes.append({"cube_id": cube_id, "split": group,
                      "mass_kg": float(rng.uniform(lo, hi)), "friction": float(rng.uniform(.4, 1.0))})
    write_new(data / "split.json", split)
    write_new(data / "cubes.json", {"cubes": cubes,
        "sampling": "Independent uniform draws for mass and friction; fixed cube identities; property values stored only in this file."})
    write_new(data / "protocol.json", PROTOCOL)
    write_new(seal, {name: digest(data / name) for name in ["split.json", "cubes.json", "protocol.json"]})
    verify()

def verify():
    data = ROOT / "data"
    seal = json.loads((data / "split_seal.json").read_text())
    for name, expected in seal.items():
        if digest(data / name) != expected:
            raise RuntimeError(f"Immutable protocol file changed: {name}")
    if json.loads((data / "protocol.json").read_text()) != PROTOCOL:
        raise RuntimeError("Protocol code differs from sealed protocol")
    return json.loads((data / "split.json").read_text())

if __name__ == "__main__":
    initialize()
    print(json.dumps({"status": "sealed", "split_sha256": digest(ROOT / "data/split.json")}))
