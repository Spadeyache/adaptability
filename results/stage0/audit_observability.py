"""Re-score archived single-attempt contact estimator errors against the training mean."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ARCHIVE = ROOT / "runs/interaction_estimator"
RUNS = ["gru_v2_corrected", "gru_v2_seed102", "gru_v2_seed103"]


def read(path):
    return json.loads(path.read_text())


def score(rows, prior, bootstrap):
    targets = np.array([r["target"] for r in rows])
    baseline = np.abs(targets - prior)
    estimator = np.array([r["attempts"]["1"] for r in rows])
    draws = np.random.default_rng(bootstrap["seed"]).integers(
        len(rows), size=(bootstrap["draws"], len(rows)))
    # Resample identical condition indices for both arms and both properties.
    reduction = baseline-estimator
    intervals = np.quantile(reduction[draws].mean(axis=1), [.025, .975], axis=0)
    comparison = []
    for i, (unit, scale) in enumerate([("mass_g", 1000), ("object_friction_coefficient", 1)]):
        comparison.append(dict(property=unit, training_mean_guess_mae=scale*baseline[:,i].mean(),
                               single_attempt_mae=scale*estimator[:,i].mean(),
                               paired_mae_reduction=scale*reduction[:,i].mean(),
                               reduction_ci95=(scale*intervals[:,i]).tolist()))
    return dict(conditions=[r["condition_id"] for r in rows], metrics=comparison)


def main():
    folder = ARCHIVE / "contact_dataset_v1"
    episode_file, inventory_file = folder / "episodes.json", OUT / "lift_inventory.json"
    episodes = read(episode_file)
    train = [r for r in episodes if r["split"] == "train"]
    counts = Counter(r["condition_id"] for r in train)
    assert len(counts) == 26 and set(counts.values()) == {10}
    prior = np.mean([r["target"] for r in train], axis=0)
    np.testing.assert_allclose(prior, read(folder / "normalization.json")["prior"], rtol=1e-14)
    inside = {c["condition_id"] for c in read(inventory_file)["conditions"]}
    test = {r["condition_id"] for r in episodes if r["split"] == "test"}
    assert test.isdisjoint(counts)
    source_files = [episode_file, inventory_file, folder / "normalization.json"]
    output = []
    for run in RUNS:
        path, manifest_path = ARCHIVE / run / "test_metrics.json", ARCHIVE / run / "manifest.json"
        report, manifest = read(path), read(manifest_path)
        source_files.extend([path, manifest_path])
        assert report["units"] == ["mass_kg", "object_friction_coefficient"]
        rows = report["conditions"]
        assert len(rows) == len(test) and {r["condition_id"] for r in rows} == test
        assert all(r["episodes"] == r["informative_episodes"] == 10 for r in rows)
        np.testing.assert_allclose(prior, report["prior_mean"], rtol=1e-14)
        targets = np.array([r["target"] for r in rows])
        single = np.array([r["attempts"]["1"] for r in rows])
        np.testing.assert_allclose(np.abs(targets-prior).mean(0), report["prior_mean_mae"], rtol=1e-14)
        np.testing.assert_allclose(single.mean(0), report["attempts"]["1"]["mae"], rtol=1e-14)
        output.append(dict(
            training_seed=manifest["seed"], ordering_seed=report["seed"],
            orderings_per_condition=report["orderings_per_condition"], bootstrap=report["bootstrap"],
            all_archived_test=score(rows, prior, report["bootstrap"]),
            in_range_archived_test=score([r for r in rows if r["condition_id"] in inside], prior, report["bootstrap"]),
            saved_condition_errors=[dict(condition_id=r["condition_id"], target=r["target"],
                                         single_attempt_mae=r["attempts"]["1"]) for r in rows]))
    data = dict(
        definition="Positive reduction means smaller mean absolute error than the constant training-mean guess.",
        source_error_units=["mass_kg", "object_friction_coefficient"],
        training_mean_guess=prior.tolist(), training_conditions=len(counts), training_episodes=len(train),
        in_range_training_conditions=len(inside.intersection(counts)), test_conditions=len(test),
        in_range_test_conditions=len(inside.intersection(test)),
        test_reset_seeds=sorted({r["seed"] for r in episodes if r["split"] == "test"}),
        training_mean_weighting="Mean of the 260 training episode labels; ten starts per physical condition "
                                "give each of 26 settings equal weight.",
        weighting="Each test condition contributes equally. Its archived one-attempt error averages the absolute "
                  "error of the first attempt across 64 sampled episode orders for its ten starts.",
        limitations="The same ten reset seeds recur across settings. Percentile intervals resample conditions "
                    "with the observed starts and sampled orderings held fixed. The eight-condition range filter "
                    "retains the original frozen checkpoint and training mean; training included lower-friction "
                    "settings. All actions come from lift attempts. Conditions, orderings and training seeds "
                    "are separate units; the three fits evaluate the same trajectories.",
        results=output,
        sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    (OUT / "observability_check.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
