"""Matched-design integrity, correlated-start bootstrap and survivor conditioning."""
from copy import deepcopy
import unittest

from experiments.physics_conditioning.checkpoint_report import (
    MODES, analyze, budget_interpretation, normalize_plan, seed_block_difference,
    validate_manifest_consistency, validate_rows,
)


def row(label="2001", grid="training", mass=.1, seed=73000, mode="true", success=True,
        force=4., lifted=True):
    return dict(checkpoint_label=label, grid=grid, mass_kg=mass, friction=.5,
                seed=seed, mode=mode, episode_id=f"{label}_{grid}_{mass}_{seed}_{mode}",
                reset_fingerprint=f"physical_{grid}_{mass}_{seed}", final_success=success,
                sustained_success=success, grasp_lost_after_lift=False, lifted_ever=lifted,
                lifted_steps=10 if lifted else 0, lifted_force_n=force,
                lifted_motor_limit_n=6. if lifted else None, reward_sum=2.)


def complete_rows():
    return [row(label, grid, mass, seed, mode) for label in ("2001", "5001", "8501")
            for grid in ("training", "intermediate") for mass in (.1, .2)
            for seed in (73000, 73001) for mode in MODES]


def declared_plan():
    return normalize_plan(dict(checkpoints={label: {} for label in ("2001", "5001", "8501")},
                               grids={name: dict(masses_kg=[.1, .2], frictions=[.5])
                                      for name in ("training", "intermediate")},
                               seeds=[73000, 73001], modes=list(MODES), steps=50,
                               episodes_per_checkpoint=48))


class DesignTests(unittest.TestCase):
    def test_complete_design_and_numeric_checkpoint_order(self):
        indexed, labels, grids, seeds = validate_rows(complete_rows())
        self.assertEqual(labels, ["2001", "5001", "8501"])
        self.assertEqual(seeds, [73000, 73001])
        self.assertEqual(len(indexed), 144)
        self.assertEqual(len(grids["training"]), 2)

    def test_missing_intervention_duplicate_and_reset_drift_fail(self):
        rows = complete_rows()
        with self.assertRaisesRegex(ValueError, "Incomplete matched"):
            validate_rows(rows[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_rows(rows + [rows[0]])
        changed = deepcopy(rows)
        changed[-1]["reset_fingerprint"] = "different physical state"
        with self.assertRaisesRegex(ValueError, "Physical reset differs"):
            validate_rows(changed)

    def test_different_seed_sets_between_grids_fail(self):
        rows = [r for r in complete_rows() if not (r["grid"] == "intermediate" and r["seed"] == 73001)]
        with self.assertRaisesRegex(ValueError, "seed blocks differ"):
            validate_rows(rows)

    def test_declared_plan_detects_case_seed_or_checkpoint_omitted_everywhere(self):
        expected = declared_plan()
        for predicate in (lambda r: r["mass_kg"] != .2, lambda r: r["seed"] != 73001,
                          lambda r: r["checkpoint_label"] != "8501"):
            with self.assertRaisesRegex(ValueError, "declared experiment plan"):
                validate_rows([r for r in complete_rows() if predicate(r)], expected)

    def test_declared_horizon_and_force_denominators_are_enforced(self):
        changed = complete_rows()
        changed[0]["lifted_steps"] = 51
        with self.assertRaisesRegex(ValueError, "exceeds"):
            validate_rows(changed, declared_plan())
        changed[0]["lifted_steps"] = 10
        changed[0]["force_sample_count"] = 10
        with self.assertRaisesRegex(ValueError, "two-finger"):
            validate_rows(changed, declared_plan())

    def test_mixed_runner_sources_or_mislabeled_checkpoint_are_rejected(self):
        sources = [dict(checkpoint_labels=[label], manifest=dict(checkpoint_label=label,
                       checkpoint_sha256=f"hash{label}", source_sha256={"runner.py":"same-source"},
                       task_semantics="native-carry", checkpoint=f"/tmp/{label}.pt"))
                   for label in ("2001", "5001", "8501")]
        parent = dict(checkpoints={label: dict(sha256=f"hash{label}", path=f"/tmp/{label}.pt")
                                   for label in ("2001", "5001", "8501")})
        validate_manifest_consistency(sources, parent)
        changed = deepcopy(sources)
        changed[-1]["manifest"]["source_sha256"]["runner.py"] = "different-source"
        with self.assertRaisesRegex(ValueError, "source_sha256 differs"):
            validate_manifest_consistency(changed, parent)
        changed = deepcopy(sources)
        changed[-1]["manifest"]["checkpoint_sha256"] = "wrong-model"
        with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
            validate_manifest_consistency(changed, parent)


class PairedStatisticsTests(unittest.TestCase):
    def test_bootstrap_resamples_whole_shared_seed_blocks(self):
        pairs = []
        for seed, left_success in ((73000, False), (73001, True)):
            for index in range(20):
                left = row("8501", mass=.1 + index * .01, seed=seed, success=left_success)
                right = row("5001", mass=left["mass_kg"], seed=seed, success=not left_success)
                pairs.append((left, right))
        result = seed_block_difference(pairs, "final_success")
        self.assertEqual(result["pairs"], 40)
        self.assertEqual(result["seed_blocks"], 2)
        self.assertEqual(result["paired_seed_bootstrap_95_interval"], [-1., 1.])
        self.assertFalse(result["left_better_supported"])

    def test_common_lift_excludes_failed_pickup_from_force_benefit(self):
        pairs = [(row("8501", seed=73000, force=4.), row("5001", seed=73000, force=6.)),
                 (row("8501", seed=73001, force=None, lifted=False, success=False),
                  row("5001", seed=73001, force=20.))]
        result = seed_block_difference(pairs, "lifted_force_n")
        self.assertEqual(result["pairs"], 1)
        self.assertEqual(result["excluded_pairs"], 1)
        self.assertEqual(result["difference"], -2.)
        self.assertIsNone(result["paired_seed_bootstrap_95_interval"])
        self.assertTrue(result["conditional_on_common_lift"])

    def test_conditional_point_estimate_matches_equal_seed_bootstrap(self):
        pairs = [(row("8501", mass=.1 + i * .01, seed=73000, force=5.),
                  row("5001", mass=.1 + i * .01, seed=73000, force=4.)) for i in range(3)]
        pairs.append((row("8501", seed=73001, force=9.), row("5001", seed=73001, force=4.)))
        result = seed_block_difference(pairs, "lifted_force_n")
        self.assertEqual(result["difference"], 3.)
        self.assertEqual(result["pooled_episode_difference"], 2.)

    def test_wrong_physical_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "changed the physical"):
            seed_block_difference([(row(mass=.1), row(mass=.2))], "final_success")

    def test_report_separates_learning_from_input_use_without_changing_training(self):
        report = analyze(complete_rows())
        self.assertEqual(len(report["checkpoint_comparisons"]), 6)
        self.assertEqual(len(report["true_vs_altered_inputs"]), 30)
        self.assertFalse(report["budget_assessment"]["fifty_thousand_cycles_necessary_or_sufficient_demonstrated"])
        self.assertIn("no automatic stop", report["budget_assessment"]["training_action"])

    def test_positive_effect_with_uncertain_size_is_not_supported_practical_gain(self):
        comparisons = [dict(left_checkpoint="8501", right_checkpoint="5001", grid=grid,
                            metrics={"final_success":dict(difference=.06,
                                    paired_seed_bootstrap_95_interval=[.001, .12])})
                       for grid in ("training", "intermediate")]
        report = budget_interpretation(comparisons, ["2001", "5001", "8501"], ["training", "intermediate"])
        for value in report["grids"].values():
            self.assertEqual(value["state"], "supported recent improvement; practical magnitude uncertain")


if __name__ == "__main__":
    unittest.main()
