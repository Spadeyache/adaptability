"""Summarize saved shove measurements; never simulates, fits, or trains a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
METRICS = {
    "stopping_distance_m": ("Stopping distance after release", "m"),
    "launch_speed_m_s": ("Launch speed at release", "m/s"),
    "time_to_stop_s": ("Time to stop after release", "s"),
}
COLORS = ["#2471a3", "#b85b16", "#208464"]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, precision=5):
    return f"{float(value):.{precision}g}"


def summarize(rows, cubes):
    """Use range of grid means versus the largest within-cell repeat range."""
    result = []
    caps = sorted({float(r["cap_n"]) for r in rows})
    for cap in caps:
        for sweep in ("mass", "friction"):
            subset = [r for r in rows if float(r["cap_n"]) == cap and r["sweep"] == sweep]
            ids = sorted({int(r["cube_id"]) for r in subset}, key=lambda i: cubes[i]["grid_index"])
            if not ids:
                continue
            groups = [[r for r in subset if int(r["cube_id"]) == cube_id] for cube_id in ids]
            entry = {"cap_n": cap, "sweep": sweep, "n_trials": len(subset),
                     "cube_ids": ids, "repeat_counts": [len(g) for g in groups],
                     "complete": len(ids) == 7 and all(len(g) == 3 for g in groups),
                     "metrics": {}}
            for key in METRICS:
                values = [np.asarray([float(r[key]) for r in group]) for group in groups]
                means = np.asarray([v.mean() for v in values])
                spread = max(float(np.ptp(v)) for v in values)
                effect = float(np.ptp(means))
                entry["metrics"][key] = {
                    "mean_by_cube": means.tolist(),
                    "min_by_cube": [float(v.min()) for v in values],
                    "max_by_cube": [float(v.max()) for v in values],
                    "first_mean": float(means[0]), "last_mean": float(means[-1]),
                    "range_of_means": effect, "max_repeat_range": spread,
                    "signal_exceeds_repeat_range": effect > spread,
                    "signal_smaller_than_repeat_range": effect < spread,
                    "nonincreasing_grid_means": bool(np.all(np.diff(means) <= 1e-10)),
                    "effect_to_spread_ratio": effect / spread if spread > 0 else None,
                }
            proxy = []
            for r in subset:
                speed, distance = float(r["launch_speed_m_s"]), float(r["stopping_distance_m"])
                if speed > .02 and distance > .0001:
                    proxy.append(speed * speed / (2 * distance))
            entry["deceleration_proxy"] = {
                "definition": "launch_speed**2 / (2*stopping_distance); only speed > 0.02 m/s and distance > 0.0001 m",
                "n_eligible": len(proxy), "min_m_s2": min(proxy) if proxy else None,
                "max_m_s2": max(proxy) if proxy else None,
            }
            entry["nonlaunch_count"] = sum(float(r["launch_speed_m_s"]) <= .02 for r in subset)
            result.append(entry)
    return result


def plot_sweeps(summary, cubes, output, keys, name):
    fig, axes = plt.subplots(len(keys), 2, figsize=(11.6, 3.8 * len(keys)), squeeze=False)
    caps = sorted({e["cap_n"] for e in summary})
    for row_index, key in enumerate(keys):
        for col_index, sweep in enumerate(("mass", "friction")):
            ax = axes[row_index, col_index]
            for color, cap in zip(COLORS, caps):
                item = next((e for e in summary if e["cap_n"] == cap and e["sweep"] == sweep), None)
                if item is None:
                    continue
                prop = "mass_kg" if sweep == "mass" else "friction"
                x = np.asarray([cubes[i][prop] for i in item["cube_ids"]])
                measured = item["metrics"][key]
                mean = np.asarray(measured["mean_by_cube"])
                lo, hi = np.asarray(measured["min_by_cube"]), np.asarray(measured["max_by_cube"])
                ax.plot(x, mean, "o-", color=color, ms=4, lw=1.6, label=f"{cap:g} N cap")
                ax.fill_between(x, lo, hi, color=color, alpha=.18)
                ax.errorbar(x, mean, yerr=np.maximum(0, np.vstack((mean-lo, hi-mean))),
                            fmt="none", ecolor=color, capsize=3, lw=1)
            ax.set_xlabel("Cube mass (kg)" if sweep == "mass" else "Cube material friction coefficient")
            title, unit = METRICS[key]
            ax.set_ylabel(f"{title} ({unit})")
            ax.set_title("Friction fixed at 0.7" if sweep == "mass" else "Mass fixed at 0.27 kg")
            ax.grid(alpha=.2)
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False)
    fig.suptitle("Fixed 80 ms shove: measured outcomes", fontsize=15)
    fig.text(.5, .012, "Lines: repeat means. Shading and bars: observed min–max (three identical resets per condition).\n"
             "Zero-width spread is numerical repeatability, not a physical uncertainty bound. Idealized rail pusher on an infinite plane.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .065 if len(keys) > 1 else .13, 1, .96))
    for ext in ("png", "svg"):
        fig.savefig(output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_forces(rows, output):
    caps = sorted({float(r["cap_n"]) for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.8))
    force_summary = []
    for index, (color, cap) in enumerate(zip(COLORS, caps)):
        subset = [r for r in rows if float(r["cap_n"]) == cap]
        peak = np.asarray([r["max_contact_force_n"] for r in subset], dtype=float)
        mean = np.asarray([r["mean_push_contact_force_n"] for r in subset], dtype=float)
        ax.hlines(cap, index-.32, index+.32, color="black", lw=2,
                  label="Configured cap" if index == 0 else None)
        # Horizontal offsets are visual separation only, not simulated noise.
        offsets = np.linspace(-.14, .14, len(subset))
        ax.scatter(index+offsets, peak, color=color, alpha=.55, s=17,
                   label="Peak measured force in each run" if index == 0 else None)
        ax.errorbar(index+.24, mean.mean(),
                    yerr=[[mean.mean()-mean.min()], [mean.max()-mean.mean()]],
                    fmt="s", color=color, capsize=4,
                    label="Pulse mean: average and range across runs" if index == 0 else None)
        force_summary.append({"cap_n": cap, "n_trials": len(subset),
                              "min_run_peak_n": float(peak.min()), "max_run_peak_n": float(peak.max()),
                              "min_pulse_mean_n": float(mean.min()), "max_pulse_mean_n": float(mean.max()),
                              "max_cap_ratio": max(float(r["cap_ratio"]) for r in subset)})
    ax.set_xticks(range(len(caps)), [f"{cap:g} N" for cap in caps])
    ax.set_xlabel("Configured contact-force cap")
    ax.set_ylabel("Measured pusher–cube contact force (N)")
    ax.set_title("Contact-force check across every saved trial")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    fig.text(.5, .01, "Each physics step: sum of contact-point impulse magnitudes / 0.001 s.\n"
             "This conservative quantity bounds the magnitude of the resultant contact force.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(output / f"contact_force.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return force_summary


def write_report(root, rows, summary, force_summary, sources, cubes):
    correction = ("**Correction discovered in the subsequent varied-shove pilot:** the force values below come "
                  "from PhysX's normal-only contact impulse field. They are not a bound on full 3D contact force. "
                  "The original independent momentum audit verified only the pushing-axis component. The saved "
                  "motion measurements and sensitivity plots are unaffected. See "
                  "[the corrected instrumentation](../shove_gate/README.md).\n\n")
    complete = len(rows) == 126 and len(summary) == 6 and all(e["complete"] for e in summary)
    failed = [e for e in summary if e["metrics"]["stopping_distance_m"]["signal_smaller_than_repeat_range"]]
    zero_signal = [e for e in summary if e["metrics"]["stopping_distance_m"]["range_of_means"] == 0]
    caps = sorted({float(r["cap_n"]) for r in rows})
    lines = ["# Clean shove sensitivity experiment", "",
             f"Analyzed **{len(rows)} saved trials**; " + ("all 126 planned trials are present." if complete else "this is a partial collection; uncollected strengths or conditions are not inferred."), "",
             "**This experiment uses an idealized 5 g spherical pusher on a horizontal rail and an infinite flat plane.** "
             "The plane avoids table-edge falls when light cubes travel far. It reuses the clean cube-property setter, "
             "4 cm cube geometry, SAPIEN/PhysX contact simulation, and fixed floor material friction of 0.3. "
             "These results validate this shove configuration; they do not establish that the Panda arm controller delivers the same shove.", "",
             "No world model was trained. The action policy has no mass/friction input. True properties are read from "
             "`data/cubes.json` only to install simulator properties and to label this post-run analysis. "
             "Original `clean/` training/test data and its sealed evaluation are separate.", "",
             "The same approach and pulse settings are used for every cube: 0.005 m/s approach target, "
             "0.03 N preload for 0.1 s, then an 80 ms force pulse commanded at 95% of the selected cap. "
             "The pusher releases and is parked away; the cube slides freely. Physics is sampled every 1 ms. "
             "Mass sweeps use seven values from 0.04 to 0.50 kg at friction 0.7; friction sweeps use seven "
             "values from 0.4 to 1.0 at mass 0.27 kg. Each condition has three fresh, identical resets.", "",
             "## Actual contact force", "",
             "Measured force is the sum of all pusher–cube contact-point impulse magnitudes divided by the physics "
             "timestep. This upper-bounds the resultant-force magnitude and includes approach, preload, pulse, "
             "and release. It is a timestep-averaged simulator contact force, not a continuous-time hardware impact peak.", "",
             "| Cap (N) | Largest measured step force (N) | Peak / cap | Pulse-mean range across runs (N) |", "|---:|---:|---:|---:|"]
    for f in force_summary:
        lines.append(f"| {f['cap_n']:g} | {number(f['max_run_peak_n'])} | {number(100*f['max_cap_ratio'])}% | "
                     f"{number(f['min_pulse_mean_n'])}–{number(f['max_pulse_mean_n'])} |")
    violations = sum(float(r["cap_ratio"]) > 1.000001 for r in rows)
    lines += ["", f"Measured cap violations: **{violations}**. "
              "The statement is empirical for the saved runs and sampled physics steps. "
              "The 5 g pusher's own acceleration makes delivered contact force vary with cube mass even under the same command; "
              "the pulse-mean ranges above expose that difference.", "", f"![Measured contact force]({root / 'results/contact_force.png'})", "",
              "## Dependence compared with repeat spread", "",
              "Effect size below is the range of the seven condition means. Repeat spread is the largest "
              "within-condition max–min across its three runs. Endpoint columns follow increasing mass or friction. "
              "These are observed ranges, not confidence intervals. Identical deterministic resets do not measure "
              "sensitivity to placement error, actuator noise, or model mismatch.", "",
              "| Sweep | Cap (N) | Distance: first → last (m) | Distance effect (m) | Largest repeat spread (m) | "
              "Launch speed: first → last (m/s) | Speed effect (m/s) | Largest repeat spread (m/s) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for e in summary:
        d, v = e["metrics"]["stopping_distance_m"], e["metrics"]["launch_speed_m_s"]
        lines.append(f"| {e['sweep']} | {e['cap_n']:g} | {number(d['first_mean'])} → {number(d['last_mean'])} | "
                     f"{number(d['range_of_means'])} | {number(d['max_repeat_range'])} | "
                     f"{number(v['first_mean'])} → {number(v['last_mean'])} | {number(v['range_of_means'])} | "
                     f"{number(v['max_repeat_range'])} |")
    lines += ["", f"![Distance and launch-speed sweeps]({root / 'results/sweeps.png'})", ""]
    for sweep in ("mass", "friction"):
        items = [e for e in summary if e["sweep"] == sweep]
        if not items:
            continue
        resolved = all(e["metrics"]["stopping_distance_m"]["signal_exceeds_repeat_range"] for e in items)
        monotone = all(e["metrics"]["stopping_distance_m"]["nonincreasing_grid_means"] for e in items)
        lines.append(f"**{sweep.capitalize()}:** " +
                     ("Stopping-distance changes exceed the observed repeat spread at every collected strength. " if resolved else
                      "At least one collected strength does not resolve a stopping-distance change above repeat spread. ") +
                     ("Mean distance decreases monotonically across the sampled grid." if monotone else
                      "At least one sampled curve is not monotonically decreasing; inspect the plotted means."))
        lines.append("")
    if failed:
        lines += ["**Stop condition met:** the stopping-distance dependence is smaller than repeat spread for " +
                  ", ".join(f"{e['sweep']} at {e['cap_n']:g} N" for e in failed) +
                  ". Do not continue collection or claim resolved dependence in these conditions.", ""]
    elif zero_signal:
        lines += ["At least one sweep has zero measured effect; zero repeat spread does not convert that into evidence of dependence.", ""]
    else:
        lines += ["The requested stop condition (dependence smaller than repeat spread) was not met in the collected stopping-distance sweeps.", ""]
    lines += ["## Can mass and friction be distinguished?", "",
              "**Stopping distance alone is ambiguous:** increasing either property can shorten the coast. "
              "The additional observables are launch speed and free-slide deceleration. A heavier cube can launch "
              "more slowly under a given shove; friction also changes how quickly a launched cube slows after release. "
              "The table below checks the kinematic deceleration proxy `v_launch² / (2 × stopping_distance)` "
              "only where launch speed exceeds 0.02 m/s and distance exceeds 0.1 mm. It is a summary of measured "
              "motion, not a fitted world model or a claim that exact Coulomb sliding holds.", "",
              "| Sweep | Cap (N) | Eligible runs | Deceleration proxy range (m/s²) | Runs ≤ 0.02 m/s at release |",
              "|---|---:|---:|---:|---:|"]
    for e in summary:
        p = e["deceleration_proxy"]
        interval = "undefined" if not p["n_eligible"] else f"{number(p['min_m_s2'])}–{number(p['max_m_s2'])}"
        lines.append(f"| {e['sweep']} | {e['cap_n']:g} | {p['n_eligible']} | {interval} | {e['nonlaunch_count']} |")
    lines += ["", "Joint launch/coast observations can distinguish the mechanisms locally when their measured responses differ. "
              "Runs with negligible launch are uninformative about free sliding. These two axis sweeps establish "
              "sensitivity around the fixed middle values; they do **not** prove unique mass/friction recovery over "
              "the full two-dimensional parameter range. The friction axis is the cube material setting, not a "
              "direct measurement of the combined cube–floor coefficient.", "",
              ]
    # A concrete comparison is selected only from saved condition means.
    pairs = []
    for cap in caps:
        mass_item = next((e for e in summary if e["cap_n"] == cap and e["sweep"] == "mass"), None)
        friction_item = next((e for e in summary if e["cap_n"] == cap and e["sweep"] == "friction"), None)
        if mass_item is None or friction_item is None:
            continue
        for mi, mid in enumerate(mass_item["cube_ids"]):
            for fi, fid in enumerate(friction_item["cube_ids"]):
                if cubes[mid]["mass_kg"] <= .27 or cubes[fid]["friction"] <= .7:
                    continue
                md = mass_item["metrics"]["stopping_distance_m"]["mean_by_cube"][mi]
                fd = friction_item["metrics"]["stopping_distance_m"]["mean_by_cube"][fi]
                if min(md, fd) > .0001:
                    pairs.append((abs(md-fd)/max(md, fd), cap, mass_item, friction_item, mi, fi))
    if pairs:
        _, cap, me, fe, mi, fi = min(pairs, key=lambda item: item[0])
        mc, fc = cubes[me["cube_ids"][mi]], cubes[fe["cube_ids"][fi]]
        md, fd = [entry["metrics"]["stopping_distance_m"]["mean_by_cube"][idx]
                  for entry, idx in ((me, mi), (fe, fi))]
        mv, fv = [entry["metrics"]["launch_speed_m_s"]["mean_by_cube"][idx]
                  for entry, idx in ((me, mi), (fe, fi))]
        lines += [f"A measured near-match illustrates the ambiguity: at {cap:g} N, the "
                  f"{number(1000*mc['mass_kg'])} g cube with friction {mc['friction']:g} stops after "
                  f"{number(1000*md)} mm, while the {number(1000*fc['mass_kg'])} g cube with friction "
                  f"{fc['friction']:g} stops after {number(1000*fd)} mm. Their launch speeds are "
                  f"{number(mv)} and {number(fv)} m/s. The distances are close, not identical; "
                  "zero repeat spread here does not establish indistinguishability under a specified sensor-noise model.", ""]
    strongest = max(caps)
    mass_item = next((e for e in summary if e["cap_n"] == strongest and e["sweep"] == "mass"), None)
    friction_item = next((e for e in summary if e["cap_n"] == strongest and e["sweep"] == "friction"), None)
    if mass_item and friction_item and all(e["deceleration_proxy"]["n_eligible"] for e in (mass_item, friction_item)):
        mp, fp = mass_item["deceleration_proxy"], friction_item["deceleration_proxy"]
        lines += [f"At {strongest:g} N, the mass sweep's deceleration proxy spans "
                  f"{number(mp['min_m_s2'])}–{number(mp['max_m_s2'])} m/s², whereas the friction sweep "
                  f"spans {number(fp['min_m_s2'])}–{number(fp['max_m_s2'])} m/s². "
                  "This supports using the free-coast portion to distinguish the two effects in this tested slice.", ""]
    lines += ["## Which push is clearest?", ""]
    for sweep in ("mass", "friction"):
        items = [e for e in summary if e["sweep"] == sweep]
        if items:
            best = max(items, key=lambda e: e["metrics"]["stopping_distance_m"]["range_of_means"])
            value = best["metrics"]["stopping_distance_m"]["range_of_means"]
            lines += [f"For {sweep}, the largest observed stopping-distance range is **{best['cap_n']:g} N** "
                      f"({number(value)} m). This compares absolute separation; with zero repeat spread, "
                      "an infinite signal-to-noise ratio is neither calculated nor used to rank strengths.", ""]
    nonlaunch = {cap: sum(float(r["launch_speed_m_s"]) <= .02 for r in rows if float(r["cap_n"]) == cap) for cap in caps}
    friction_speed_effects = [e["metrics"]["launch_speed_m_s"]["range_of_means"] for e in summary if e["sweep"] == "friction"]
    if friction_speed_effects:
        lines += ["For launch speed alone, friction produces a similar absolute separation at the collected strengths: "
                  f"{number(min(friction_speed_effects))}–{number(max(friction_speed_effects))} m/s across the full friction range. "
                  "The strongest-push advantage is clearer in stopping distance than in friction-related launch-speed change.", ""]
    lines += ["Negligible-launch counts by cap: " + "; ".join(f"{cap:g} N: {count}" for cap, count in nonlaunch.items()) +
              ". Stronger pulses may improve separation while increasing required travel space.", "",
              "## Stop time and physical checks", "",
              "Time to stop is the first below-threshold state of the sustained stop window; the confirmation "
              "window itself is excluded. See the collection protocol for exact speed and angular-speed thresholds.", "",
              "| Sweep | Cap (N) | Stop time: first → last (s) | Range of means (s) | Largest repeat spread (s) |",
              "|---|---:|---:|---:|---:|"]
    for e in summary:
        t = e["metrics"]["time_to_stop_s"]
        lines.append(f"| {e['sweep']} | {e['cap_n']:g} | {number(t['first_mean'])} → {number(t['last_mean'])} | "
                     f"{number(t['range_of_means'])} | {number(t['max_repeat_range'])} |")
    lines += ["", f"![Time-to-stop sweeps]({root / 'results/time_to_stop.png'})", "",
              f"Stopped runs: {sum(bool(r['stopped']) for r in rows)}/{len(rows)}. "
              f"Largest recorded tilt: {number(max(float(r['max_tilt_deg']) for r in rows))} degrees. "
              f"Cube center-height range: {number(min(float(r['min_cube_z_m']) for r in rows))}–"
              f"{number(max(float(r['max_cube_z_m']) for r in rows))} m. "
              f"Largest post-release pusher contact force: {number(max(float(r['max_slide_contact_force_n']) for r in rows))} N.", "",
              "## Saved evidence and reproduction", "",
              "Every plotted point comes from `results/metrics.json`, whose trial IDs refer to saved simulator traces. "
              "No curve smoothing, synthetic samples, fitted trend lines, or added repeat noise are used. "
              "`results/summary.json` records effect sizes, repeat ranges, and source hashes. "
              "`results/trial_metrics.csv` contains every measured trial outcome, indexed by cube ID; "
              "true property labels remain in `data/cubes.json`. Figures are saved as both PNG and SVG.", "", "```bash", "cd /home/ubuntu/faliure",
              "PYTHONPATH=clean/vendor:. MPLCONFIGDIR=/tmp/clean-mpl .venv/bin/python -m clean.shove.analyze", "```", "",
              "Input SHA-256 hashes:", ""]
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in sources.items())
    rendered="\n".join(lines)+"\n"
    first,rest=rendered.split("\n",1)
    (root / "RESULTS.md").write_text(first+"\n\n"+correction+rest.lstrip("\n"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "results"
    cube_path, metric_path = root / "data/cubes.json", output / "metrics.json"
    cubes = {int(c["cube_id"]): c for c in read_json(cube_path)["cubes"]}
    rows = read_json(metric_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("metrics.json must contain a nonempty list of saved trial metrics")
    identities = [(int(r["cube_id"]), float(r["cap_n"]), int(r["repeat"])) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate cube/cap/repeat: refusing to inflate sample counts")
    for row in rows:
        if cubes[int(row["cube_id"])]["sweep"] != row["sweep"]:
            raise ValueError("Metric sweep disagrees with cube registry")
        for key in METRICS:
            if not np.isfinite(float(row[key])):
                raise ValueError(f"Nonfinite {key} in trial {row['trial_id']}")
    sources = {"data/cubes.json": sha256(cube_path), "results/metrics.json": sha256(metric_path)}
    for name in ("data/protocol.json", "data/trials.json"):
        if (root / name).exists():
            sources[name] = sha256(root / name)
    summary = summarize(rows, cubes)
    plot_sweeps(summary, cubes, output, ["stopping_distance_m", "launch_speed_m_s"], "sweeps")
    plot_sweeps(summary, cubes, output, ["time_to_stop_s"], "time_to_stop")
    force_summary = plot_forces(rows, output)
    result = {"n_trials": len(rows), "source_sha256": sources, "sweeps": summary, "contact_force": force_summary,
              "repeat_spread": "max-minus-min of three identical fresh resets; not physical-noise uncertainty",
              "no_world_model_training": True}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "trial_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(root, rows, summary, force_summary, sources, cubes)
    print(json.dumps({"n_trials": len(rows), "report": str(root / "RESULTS.md"), "figures": str(output)}))


if __name__ == "__main__":
    main()
