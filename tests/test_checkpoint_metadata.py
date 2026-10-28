"""Saved checkpoint counters must never inherit an unfinished planned budget."""
import unittest

from robot_adaptation.policy.checkpoint_metadata import checkpoint_progress, stopped_training_progress


def manifest(status="stopped_by_user"):
    return dict(status=status, planned_iterations=50000,
                settings=dict(num_envs=60, num_steps=50, total_timesteps=150000000))


def stopped_record():
    return dict(status="stopped_by_user", requested_50000_completed=False,
                original_planned_iterations=50000, completed_rollouts=19489,
                collected_transitions=58467000, latest_saved_checkpoint="/tmp/ckpt_19001.pt",
                saved_checkpoint_completed_cycles=19000, saved_checkpoint_transitions=57000000)


class CheckpointMetadataTests(unittest.TestCase):
    def test_numbered_checkpoint_counts_saved_cycles_not_original_plan(self):
        value = checkpoint_progress("/tmp/ckpt_19001.pt", manifest())
        self.assertEqual(value["completed_learning_cycles"], 19000)
        self.assertEqual(value["checkpoint_transitions"], 57000000)
        self.assertEqual(value["original_planned_iterations"], 50000)
        self.assertEqual(value["original_planned_transitions"], 150000000)
        self.assertFalse(value["full_planned_run_completed"])

    def test_final_filename_requires_completed_manifest(self):
        self.assertIsNone(checkpoint_progress("final_ckpt.pt", manifest())["completed_learning_cycles"])
        value = checkpoint_progress("final_ckpt.pt", manifest("complete"))
        self.assertEqual(value["completed_learning_cycles"], 50000)
        self.assertTrue(value["full_planned_run_completed"])
        self.assertIsNone(checkpoint_progress("arbitrary_weights.pt", manifest("complete"))["completed_learning_cycles"])

    def test_executed_rollouts_and_saved_weights_stay_separate(self):
        value = stopped_training_progress(stopped_record(), manifest())
        self.assertEqual(value["executed_rollout_cycles"], 19489)
        self.assertEqual(value["saved_checkpoint_completed_cycles"], 19000)
        self.assertEqual(value["unsaved_rollout_cycles"], 489)
        self.assertFalse(value["requested_50000_iterations_completed"])
        self.assertFalse(value["training_restart_authorized"])

    def test_inconsistent_saved_counter_or_transition_accounting_fails(self):
        record = stopped_record()
        record["saved_checkpoint_completed_cycles"] = 19489
        with self.assertRaisesRegex(ValueError, "numbered checkpoint"):
            stopped_training_progress(record, manifest())
        record = stopped_record()
        record["collected_transitions"] = 150000000
        with self.assertRaisesRegex(ValueError, "executed rollout"):
            stopped_training_progress(record, manifest())


if __name__ == "__main__":
    unittest.main()
