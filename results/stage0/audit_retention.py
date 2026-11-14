"""Compare the recorded filters under the original raw-estimator settling criterion."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[1]
RUN = ROOT / "runs/mass_belief/nov17_v1"


def load_errors(config):
    """Index by sequence, trial and observed step. A row supplies all paired predictions."""
    n, k, h = (config[field] for field in ("sequences", "trials", "horizon"))
    shape = (n, k, h)
    fields = {"carried": "carry_mean_g", "reset": "reset_mean_g", "raw": "raw_g"}
    errors = {arm: np.full(shape, np.nan) for arm in fields}
    seen = np.zeros(shape, dtype=bool)
    with (RUN / "steps.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            s, t, p = int(row["sequence"]), int(row["trial"])-1, int(row["step"])-1
            if not (0 <= s < n and 0 <= t < k and 0 <= p < h) or seen[s, t, p]:
                raise ValueError(f"Unexpected or duplicate observation at {(s, t+1, p+1)}")
            truth = float(row["truth_g"])
            if not np.isclose(truth, config["heavy_g"]) or int(row["seed"]) != config["test_seed"]+k*s+t:
                raise ValueError(f"Unexpected object or reset seed at {(s, t+1, p+1)}")
            seen[s, t, p] = True
            for arm, field in fields.items():
                errors[arm][s, t, p] = abs(float(row[field])-truth)
    if not seen.all() or not all(np.isfinite(arm).all() for arm in errors.values()):
        raise ValueError("Incomplete or non-finite step history")
    np.testing.assert_array_equal(errors["carried"][:, 0], errors["reset"][:, 0])
    return errors


def capped_steps(errors, tolerance, hold):
    """Length of prefix before the last sustained in-tolerance suffix. Non-attainment costs H."""
    horizon = errors.shape[-1]
    steps = np.arange(1, horizon+1)
    # The count of observations before the final good suffix equals the last bad step.
    last_bad = np.where(errors > tolerance, steps, 0).max(axis=-1)
    return np.where(horizon-last_bad >= hold, last_bad, horizon)


def compare(carry, reset, horizon, draws):
    # Retain whole sequences and both filters when resampling. All trials have equal weight.
    difference = (reset-carry).mean(axis=1)
    lo, hi = np.quantile(difference[draws].mean(axis=1), [.025, .975])
    return {
        "attempts_per_arm": int(carry.size),
        "carried_reached": int((carry < horizon).sum()),
        "reset_reached": int((reset < horizon).sum()),
        "carried_mean_capped_steps_before_entry": float(carry.mean()),
        "reset_mean_capped_steps_before_entry": float(reset.mean()),
        "paired_mean_saving_reset_minus_carried_steps": float(difference.mean()),
        "paired_95pct_interval_for_saving_steps": [float(lo), float(hi)],
    }


def main():
    config = json.loads((RUN / "config.json").read_text())
    summary = json.loads((RUN / "summary.json").read_text())
    errors = load_errors(config)
    h, k, n = (config[field] for field in ("horizon", "trials", "sequences"))
    tol, hold = config["settle_tolerance_g"], config["settle_consecutive_steps"]
    counts = {arm: capped_steps(e, tol, hold) for arm, e in errors.items()}
    raw = counts["raw"]
    observed_raw = {"settled": int((raw < h).sum()), "episodes": raw.size,
                    "median_step": float(np.median(raw[raw < h]+1))}
    if observed_raw != summary["settling"]:
        raise ValueError("Raw settling result no longer reproduces the source report")
    # Retain the original report's bootstrap draw count and sequence-level seed.
    seed, n_draws = config["seed"]+1, 10000
    draws = np.random.default_rng(seed).integers(n, size=(n_draws, n))
    carry, reset = counts["carried"], counts["reset"]
    output = {
        "dataset": str(RUN.relative_to(ROOT)),
        "scope": {"mass_g": config["heavy_g"], "friction": config["friction"],
                  "trial_count": k, "sequences": n, "steps_per_trial": h,
                  "policy_mass_guess_g": config["probe_g"]},
        "criterion": {"tolerance_g_inclusive": tol,
                      "required_consecutive_recorded_steps_through_trial_end": hold},
        "metric": ("Count recorded observations preceding the first entry that holds through trial end. "
                   "Count the full horizon on non-attainment. Zero means already within tolerance at the "
                   "first post-action observation and sustained through the end. The log has no step zero. "
                   "The entry is identified retrospectively, using true mass for scoring."),
        "uncertainty": {"method": "Paired percentile bootstrap, resampling whole five-trial sequences",
                        "draws": n_draws, "seed": seed, "level": .95},
        "raw_estimator_crosscheck": observed_raw,
        "by_trial": [{"trial": i+1, **compare(carry[:, i:i+1], reset[:, i:i+1], h, draws)}
                     for i in range(k)],
        "trials_two_through_five": compare(carry[:, 1:], reset[:, 1:], h, draws),
        "first_qualifying_observed_step_by_sequence_then_trial": {
            arm: [[None if step == h else int(step+1) for step in sequence] for sequence in counts[arm]]
            for arm in ("carried", "reset")
        },
        "limitations": ("Retrospective comparison of a supervised mass estimator and filter on one "
                        "400 g, fixed-friction setting. Five same-physics trials per sequence versus ten "
                        "requested. Intervals quantify reset-sequence variability at this setting. "
                        "The study did not test faster inference with the proposed frozen world model, "
                        "or transfer of a starting guess between cubes."),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (RUN/"steps.csv", RUN/"config.json", RUN/"summary.json", RUN/"source/experiments/mass_belief/report.py")
        },
    }
    (DEST / "retention_check.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
