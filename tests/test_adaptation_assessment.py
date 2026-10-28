"""A successful run or sensitive actor must not turn into a capability claim."""
import unittest

from experiments.interaction_adaptation.assess import assess, markdown
from tests.test_checkpoint_metadata import manifest, stopped_record


class AssessmentTests(unittest.TestCase):
    def test_user_stop_reports_saved_and_executed_progress_without_50k_completion(self):
        report = assess(training=manifest(), early_stop=stopped_record())
        training = report["training"]
        self.assertEqual(training["saved_checkpoint_completed_cycles"], 19000)
        self.assertEqual(training["executed_rollout_cycles"], 19489)
        self.assertFalse(training["requested_50000_iterations_completed"])
        text = markdown(report)
        self.assertIn("19,489", text)
        self.assertIn("19,000", text)
        self.assertIn("budget was not completed", text)
        self.assertEqual(report["policy"]["status"], "pending")

    def test_cross_task_reports_and_wrong_stopped_checkpoint_fail(self):
        with self.assertRaisesRegex(ValueError, "same declared task"):
            assess(retry={"task":"vertical"}, transfer={"task":"native_carry"})
        early = stopped_record()
        early["latest_saved_checkpoint_sha256"] = "pinned-stopped-policy"
        with self.assertRaisesRegex(ValueError, "early-stop pinned"):
            assess(policy=dict(paired_comparisons={}, metrics={}, checkpoint="ckpt_19001.pt",
                               checkpoint_sha256="other-policy"), training=manifest(), early_stop=early)

    def test_friction_sensitivity_alone_does_not_establish_task_benefit(self):
        policy = dict(paired_comparisons={}, metrics={}, checkpoint="pilot.pt",
                      friction_input_sensitivity_mean_abs_action=99.)
        self.assertEqual(assess(policy=policy)["policy"]["status"], "not_demonstrated")

    def test_friction_benefit_alone_is_only_partial_two_parameter_support(self):
        policy = dict(paired_comparisons={
            "wrong_friction": {"paired_bootstrap_95_interval": [.02, .2]},
            "wrong_mass": {"paired_bootstrap_95_interval": [-.1, .1]},
            "fixed_both": {"paired_bootstrap_95_interval": [.01, .2]}},
            metrics={}, checkpoint="trained.pt")
        report = assess(policy=policy)["policy"]
        self.assertEqual(report["status"], "partially_supported")
        self.assertFalse(report["correct_inputs_benefit"]["wrong_mass"])

    def test_prepared_or_running_full_budget_is_not_completed_training(self):
        for status in ("prepared", "running", "failed"):
            report = assess(training=dict(status=status, planned_iterations=50000))
            self.assertFalse(report["training"]["requested_50000_iterations_completed"])
        report = assess(training=dict(status="complete", planned_iterations=200))
        self.assertFalse(report["training"]["requested_50000_iterations_completed"])

    def test_one_successful_estimator_target_does_not_pass_both(self):
        estimator = dict(attempts={"5": {"improvement_condition_bootstrap_95ci":
                                      [[.01, .1], [-.01, .2]]}},
                         held_conditions=9, prior_median_mae=[.1, .2], checkpoint="best.pt")
        report = assess(estimator=estimator)["estimator"]
        self.assertEqual(report["status"], "partially_supported")
        self.assertFalse(report["targets_improve_over_training_median"]["friction"])

    def test_historical_support_does_not_hide_failed_new_policy_transfer(self):
        historical=dict(attempts={"5":{"improvement_condition_bootstrap_95ci":[[.01,.1],[.01,.2]]}},
                        held_conditions=9,prior_median_mae=[.1,.2],checkpoint="best.pt")
        transfer=dict(attempts={"5":{"improvement_condition_bootstrap_95ci":[[.01,.1],[-.2,-.01]]}},
                      held_conditions=16,prior_median_mae=[.1,.2])
        report=assess(estimator=historical,transfer=transfer)
        self.assertEqual(report["estimator"]["status"],"supported_on_historical_held_out_conditions")
        self.assertEqual(report["estimator_transfer"]["status"],"partially_supported")
        self.assertFalse(report["estimator_transfer"]["targets_improve_over_training_median"]["friction"])


if __name__ == "__main__":
    unittest.main()
