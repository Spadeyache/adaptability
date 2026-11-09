"""Plot the measured PickCube coverage and export a labeled transition from the saved data."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np


MOTION = {
    "policy_noise": ("Noisy policy", "#1e6470"),
    "scripted_push": ("Scripted push", "#be663e"),
    "random_poke": ("Random poke", "#70639d"),
}


def export_example(root, protocol):
    path = root / "data" / "shard_0000.npz"
    with np.load(path, allow_pickle=False) as shard:
        states = shard["state"]
        push = protocol["sources"].index("scripted_push")
        train_pushes = (shard["source_id"] == push) & ~shard["is_validation"]
        displacement = np.linalg.norm(np.diff(states[:, :, 26:29], axis=1), axis=-1)
        episode, step = np.argwhere(train_pushes[:, None] & (displacement > .001))[0]

        def state_at(t):
            return {name: states[episode, t, start:end].tolist()
                    for name, (start, end) in protocol["state"].items()}

        example = {
            "shard": str(path.relative_to(root)),
            "row": int(episode),
            "step": int(step),
            "episode_id": int(shard["episode_id"][episode]),
            "reset_seed": int(shard["seed"][episode]),
            "source": "scripted_push",
            "split": "train",
            "state": state_at(step),
            "command": dict(zip(protocol["command"], shard["command"][episode, step].tolist())),
            "next_state": state_at(step + 1),
            "constants": dict(zip(protocol["constants"], shard["constants"][episode].tolist())),
            "applied": dict(zip(protocol["command"], shard["applied"][episode, step].tolist())),
        }
    (root / "transition.json").write_text(json.dumps(example, indent=2, allow_nan=False) + "\n")


def plot_collection(root, rows, protocol):
    fig, (pilot_ax, physics_ax, motion_ax) = plt.subplots(1, 3, figsize=(13.8, 4.5), layout="constrained")
    pilot = root / "pilot" / "shard_0000.npz"
    with np.load(pilot, allow_pickle=False) as shard:
        xyz = shard["state"][:, :, 26:29]
        travel_cm = 100 * np.linalg.norm(xyz - xyz[:, :1], axis=-1)
        for source, displacement in zip(shard["source_id"], travel_cm):
            color = MOTION[protocol["sources"][source]][1]
            pilot_ax.plot(displacement, color=color, lw=1.6, alpha=.85)
    pilot_ax.set(title="Cube motion in the 10-episode pilot", xlabel="Control step",
                ylabel="Distance from start (cm)", xlim=(0, protocol["config"]["steps"]), ylim=(0, None))

    mass_g = np.array([float(row["mass_kg"]) * 1000 for row in rows])
    friction = np.array([float(row["cube_friction"]) for row in rows])
    min_mass, max_mass = np.array(protocol["config"]["mass_kg"]) * 1000
    mass_bins = np.geomspace(min_mass, max_mass, 17)
    friction_bins = np.linspace(*protocol["config"]["cube_friction"], 17)
    counts, _, _ = np.histogram2d(mass_g, friction, bins=(mass_bins, friction_bins))
    if counts.sum() != len(rows):
        raise ValueError("The physics plot must include every episode")
    tiles = physics_ax.pcolormesh(mass_bins, friction_bins, counts.T, cmap="viridis", vmin=0)
    physics_ax.set(xscale="log", title=f"Physics across {len(rows):,} episodes",
                  xlabel="Cube mass (g, log scale)", ylabel="Cube friction coefficient")
    physics_ax.set_xticks([20, 50, 100, 200, 500, 1000])
    physics_ax.xaxis.set_major_formatter(ScalarFormatter())
    fig.colorbar(tiles, ax=physics_ax, label="Episodes per bin", shrink=.85)

    for key, (label, color) in MOTION.items():
        peaks = sorted(float(row["peak_cube_displacement_mm"]) / 10 for row in rows if row["source"] == key)
        motion_ax.ecdf(peaks, complementary=True, color=color, lw=2, label=label)
    motion_ax.set(xscale="symlog", xlim=(0, None), ylim=(0, 1.01),
                  title="Cube travel by action source", ylabel="Share exceeding that distance",
                  xlabel="Peak distance from start (cm)")
    motion_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    motion_ax.xaxis.set_major_formatter(ScalarFormatter())
    for axis in (pilot_ax, motion_ax):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", linewidth=.6, alpha=.25)
    handles = [Line2D([0], [0], color=color, lw=2.5, label=name) for name, color in MOTION.values()]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False)
    fig.savefig(root / "collection.png", dpi=170)
    plt.close(fig)


def write_card(root, summary, protocol, rows):
    pilot = json.loads((root / "pilot" / "complete.json").read_text())
    cfg = protocol["config"]
    card = [
        "# PickCube trajectories with randomized physics",
        "",
        f"{summary['transition_count']:,} control steps across {summary['episodes']:,} complete "
        f"{cfg['steps']}-step episodes. ManiSkill ran CUDA PhysX on the {summary['gpu']} "
        f"with {cfg['shard_episodes']} environments per batch.",
        "",
        f"The training split contains {summary['train_transitions']:,} transitions and the validation "
        f"split contains {summary['validation_transitions']:,}. Each of the {summary['shards']} NumPy "
        "shards includes a whole-episode validation mask. The pilot is stored separately.",
        "",
        "| Actions | Train episodes | Validation episodes | Cube moved >1 cm | Median peak travel |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, (label, _) in MOTION.items():
        values = summary["sources"][key]
        card.append(f"| {label} | {values['train']:,} | {values['validation']:,} | "
                    f"{values['moved_over_1cm']:,} | {values['median_peak_mm']/10:.1f} cm |")
    long_travel = sum(float(row["peak_cube_displacement_mm"]) > 1000 for row in rows)
    ranges = summary["realized_ranges"]
    mass_g = np.array(ranges["mass_kg"]) * 1000
    friction = ranges["cube_friction"]
    grip = ranges["grip_scale"]
    bins = [np.geomspace(*cfg["mass_kg"], 5), np.linspace(*cfg["cube_friction"], 5),
            np.linspace(*cfg["grip_scale"], 5)]
    occupied = {}
    for split in ("train", "validation"):
        per_source = []
        for source in MOTION:
            points = [[float(r["mass_kg"]), float(r["cube_friction"]), float(r["grip_scale"])]
                      for r in rows if r["source"] == source and r["split"] == split]
            counts, _ = np.histogramdd(np.asarray(points), bins=bins)
            per_source.append(counts > 0)
        occupied[split] = np.logical_and.reduce(per_source).sum()
    card.extend([
        "",
        f"Peak travel measures the greatest 3-D distance from the initial cube position. "
        f"In {long_travel:,} episodes the cube traveled over 1 m. The dataset includes those "
        "trajectories, misses, and short movements. The episode index records each peak.",
        "",
        f"Observed cube mass ranged from {mass_g[0]:.2f} to {mass_g[1]:.2f} g, cube friction "
        f"from {friction[0]:.4f} to {friction[1]:.4f}, and gripper force scale from "
        f"{grip[0]:.4f}× to {grip[1]:.4f}×. Table friction, arm damping and stiffness scales, "
        "command delay, and the three object-frame center-of-mass offsets complete the ten labels.",
        "",
        "Dividing mass into four equal log intervals and friction and grip scale into four "
        f"equal linear intervals gives 64 combinations. All three action sources occur in "
        f"{occupied['train']}/64 training combinations and {occupied['validation']}/64 "
        "validation combinations.",
        "",
        "## Read a transition",
        "",
        "At row `i` and control step `t`, use `state[i, t]`, `command[i, t]`, "
        "`state[i, t+1]` and `constants[i]`. Actions contain three normalized translations, "
        "finger opening and finger strength. The separate `applied[i, t]` records the action "
        "after the simulated delay. The frozen policy received an independently drawn mass guess.",
        "",
        "The first seven joint positions are angles in radians and the last two are finger positions "
        "in meters. The state also records joint velocities, tool and cube poses, total finger opening, "
        "and world-axis cube linear and angular velocities. Pose coordinates are local to each scene.",
        "",
        "## Collection checks",
        "",
        f"All {pilot['transition_count']} pilot transitions were reopened before full collection. "
        f"{pilot['moved_over_1cm']} of {pilot['episodes']} pilot cubes moved over 1 cm. "
        "The collector checked native rigid-body, surface and drive properties after reset. "
        "The final scan checked state and action timing, delay queues, both finger force limits, "
        "and episode identities and splits for every shard.",
        "",
        f"The scenes were {cfg['scene_spacing_m']:g} m apart. The largest absolute cube x or y "
        f"coordinate in the saved trajectories was {summary['max_abs_cube_xy_m']:.2f} m from its "
        "scene origin.",
        "",
        "[Schema and config](data/dataset.json) · [First data shard](data/shard_0000.npz) · "
        "[Episode index](data/episodes.csv) · [Named example](transition.json) · "
        "[Full-run checks](data/complete.json) · [Checksums](data/checksums.json) · "
        "[Pilot checks](pilot/complete.json)",
        "",
        "![Pilot motion, physics coverage and cube travel by action source](collection.png)",
    ])
    (root / "DATASET.md").write_text("\n".join(card) + "\n")


def report(root):
    folder = root / "data"
    summary = json.loads((folder / "complete.json").read_text())
    protocol = json.loads((folder / "dataset.json").read_text())
    with (folder / "episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not summary["verified"] or len(rows) != summary["episodes"]:
        raise ValueError("The episode index does not match the verified dataset")
    export_example(root, protocol)
    plot_collection(root, rows, protocol)
    write_card(root, summary, protocol, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root)
