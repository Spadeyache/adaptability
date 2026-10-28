"""A user stop changes scope, never the meaning of saved/executed progress."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.interaction_adaptation.completion import AuditExpectations, Incomplete, digest
from experiments.interaction_adaptation.early_stop_completion import (
    audit, collection_raw, validate_inventory, validate_plan,
    validate_stopped_execution, validate_stopped_processes, validate_transfer,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class EarlyStopCompletionTests(unittest.TestCase):
    def guarded_fixture(self, root):
        (root / "trust_region.py").write_text("# preserved guard\n")
        manifest = dict(status="stopped_by_user", planned_iterations=50000, completed_rollouts=3,
                        optimization_semantics="fixed_original_rollout", trust_region=dict(enabled=True),
                        trust_region_sha256=digest(root / "trust_region.py"),
                        settings=dict(num_envs=60, num_steps=50, total_timesteps=150000000,
                                      guard_policy_updates=True, max_exact_kl=.05, learning_rate=1e-5))
        stop = dict(status="stopped_by_user", requested_50000_completed=False,
                    original_planned_iterations=50000, completed_rollouts=3, last_guard_iteration=3,
                    latest_saved_checkpoint=str(root / "ckpt_3.pt"), saved_checkpoint_completed_cycles=2,
                    saved_checkpoint_transitions=6000, collected_transitions=9000)
        rows = [dict(iteration=i, global_step=i*3000, rollout_observations=3000, effective_actions=5,
                     optimization_semantics="fixed_original_rollout", max_accepted_exact_kl=.04,
                     rollout_exact_kl=.03, accepted_updates=2) for i in range(1, 4)]
        (root / "trust_region.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        return manifest, stop, rows

    def test_executed_cycles_and_saved_policy_are_distinct_without_weakening_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, stop, _ = self.guarded_fixture(root)
            result = validate_stopped_execution(root, manifest, stop, require_guarded=True)
            self.assertEqual(result["executed_rollout_cycles"], 3)
            self.assertEqual(result["saved_checkpoint_completed_cycles"], 2)
            self.assertEqual(result["unsaved_rollout_cycles"], 1)
            self.assertFalse(result["requested_50000_iterations_completed"])
            self.assertEqual(AuditExpectations().minimum_iterations, 50000)

    def test_sequential_guard_evidence_and_transition_counts_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, stop, rows = self.guarded_fixture(root)
            rows[1]["iteration"] = 1
            (root / "trust_region.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "missing, duplicated"):
                validate_stopped_execution(root, manifest, stop)
            manifest, stop, _ = self.guarded_fixture(root)
            stop["collected_transitions"] = 12000
            with self.assertRaisesRegex(ValueError, "transitions"):
                validate_stopped_execution(root, manifest, stop)

    def test_loosening_bound_or_mislabeling_checkpoint_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, stop, _ = self.guarded_fixture(root)
            manifest["settings"]["max_exact_kl"] = .5
            with self.assertRaisesRegex(ValueError, "predeclared"):
                validate_stopped_execution(root, manifest, stop)
            manifest, stop, _ = self.guarded_fixture(root)
            stop["saved_checkpoint_completed_cycles"] = 3
            with self.assertRaisesRegex(ValueError, "numbered checkpoint"):
                validate_stopped_execution(root, manifest, stop)

    def proc_fixture(self, root):
        boot = root / "sys/kernel/random/boot_id"
        boot.parent.mkdir(parents=True)
        boot.write_text("host-boot")
        (root / "self/ns").mkdir(parents=True)
        (root / "self/ns/pid").symlink_to("pid:[42]")
        return [dict(role=role, pid=pid, start_ticks=100, boot_id="host-boot", pid_namespace="pid:[42]")
                for role, pid in (("supervisor", 10), ("worker", 20))]

    def test_process_identity_accepts_absent_or_reused_but_rejects_live_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            handles = self.proc_fixture(root)
            self.assertTrue(all(h["observation"] == "absent" for h in validate_stopped_processes(handles, proc_root=root)["handles"]))
            path = root / "20/stat"
            path.parent.mkdir()
            # state=field3; starttime=field22. comm contains spaces and ')'.
            path.write_text("20 (a worker ) name) S " + "0 "*18 + "101 0\n")
            result = validate_stopped_processes(handles, proc_root=root)
            self.assertEqual(result["handles"][1]["observation"], "pid_reused")
            path.write_text("20 (worker) S " + "0 "*18 + "100 0\n")
            with self.assertRaisesRegex(ValueError, "still alive"):
                validate_stopped_processes(handles, proc_root=root)

    def test_boot_or_pid_namespace_mismatch_is_incomplete_not_exit_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            handles = self.proc_fixture(root)
            handles[0]["boot_id"] = "another-boot"
            with self.assertRaises(Incomplete):
                validate_stopped_processes(handles, proc_root=root)
            handles[0]["boot_id"] = "host-boot"
            handles[0]["pid_namespace"] = "pid:[99]"
            with self.assertRaisesRegex(Incomplete, "PID namespace"):
                validate_stopped_processes(handles, proc_root=root)

    def test_source_inventory_rejects_mutation_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "one.py").write_text("original")
            inventory = root / "hashes.json"
            write_json(inventory, {"one.py": digest(source / "one.py")})
            self.assertEqual(validate_inventory(source, inventory)["files"], 1)
            (source / "one.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_inventory(source, inventory)
            write_json(inventory, {"../hashes.json": "anything"})
            with self.assertRaisesRegex(ValueError, "escapes"):
                validate_inventory(source, inventory)

    def raw_fixture(self, root):
        from robot_adaptation.collection.schema import INPUT_LAYOUTS
        rows = [dict(episode_id="one", seed=54000, success=True, actual_mass_kg=.1,
                     actual_friction=.4, supplied_mass_kg=.1, method="true", condition_index=0,
                     sequence_index=0, attempt=1)]
        lengths = dict(state=(51,), transition=(50,), physics=(100,), constant=())
        data = {}
        for key, (timeline, shape) in INPUT_LAYOUTS.items():
            if key in ("input__action_raw", "input__action_applied"):
                shape = (5,)
            data[key] = np.zeros(lengths[timeline]+shape)
        for key in ("input__object_pose_world_m_wxyz", "input__object_pose_tcp_m_wxyz", "input__tcp_pose_world_m_wxyz"):
            data[key][:, 3] = 1
        reset = json.dumps({"physical": [1, 2]}, sort_keys=True)
        data.update(meta__schema_version=np.asarray("pick_failure_v3"),
                    meta__physics_time_s=np.arange(100)/40., meta__fresh_scene_reset=np.asarray(True),
                    sim__contact_point_offsets=np.zeros(101, dtype=int),
                    sim__contact_position_world_m=np.zeros((0, 3)), sim__task_success=np.ones(51, dtype=bool),
                    label__success_final=np.asarray(True), label__held_goal_success_final=np.asarray(True),
                    label__seed=np.asarray(54000), label__mass_kg=np.asarray(.1),
                    label__object_static_friction=np.asarray(.4), label__object_dynamic_friction=np.asarray(.4),
                    label__finger_static_friction=np.asarray([.2, .2]), label__finger_dynamic_friction=np.asarray([.2, .2]),
                    label__table_static_friction=np.asarray(.3), label__table_dynamic_friction=np.asarray(.3),
                    label__size_xyz_m=np.asarray([.04]*3), label__reset_state_json=np.asarray(reset),
                    label__reset_state_hash=np.asarray(hashlib.sha256(reset.encode()).hexdigest()),
                    label__initial_object_pose_world_m_wxyz=np.asarray([0, 0, .02, 1, 0, 0, 0]),
                    input__goal_position_world_m=np.asarray([0, 0, .22]), input__reported_mass_kg=np.asarray(.1))
        (root / "episodes").mkdir()
        for name in ("manifest.json", "summary.json", "attempts.jsonl"):
            (root / name).write_text("{}")
        np.savez_compressed(root / "episodes/one.npz", **data)
        return rows, data

    def test_raw_sensor_schema_reset_hash_and_declared_task_are_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, data = self.raw_fixture(root)
            self.assertEqual(collection_raw(root, rows, retries=True, task="vertical")["raw_episodes"], 1)
            data["label__reset_state_json"] = np.asarray('{"physical": [3]}')
            np.savez_compressed(root / "episodes/one.npz", **data)
            with self.assertRaisesRegex(ValueError, "does not hash"):
                collection_raw(root, rows)
            data["label__reset_state_hash"] = np.asarray(hashlib.sha256(str(data["label__reset_state_json"]).encode()).hexdigest())
            data["input__goal_position_world_m"] = np.asarray([.1, 0, .22])
            np.savez_compressed(root / "episodes/one.npz", **data)
            with self.assertRaisesRegex(ValueError, "vertical task geometry"):
                collection_raw(root, rows, task="vertical")

    def test_missing_post_stop_outputs_cannot_be_reported_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = audit(root, evaluation_root=root / "post", require_guarded_updates=True)
            self.assertFalse(result["overall_ready"])
            self.assertFalse(result["requested_50000_iterations_completed"])
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(len(result["requirements"]), 15)

    def test_assessment_negative_findings_are_valid_but_stale_recorded_source_hash_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            post, train = root / "post", root / "train"
            stop = dict(status="stopped_by_user", completed_rollouts=3, saved_checkpoint_completed_cycles=2,
                        latest_saved_checkpoint_sha256="policy-hash", requested_50000_completed=False,
                        latest_saved_checkpoint=str(train / "ckpt_3.pt"), original_planned_iterations=50000,
                        collected_transitions=9000, saved_checkpoint_transitions=6000)
            write_json(root / "early_stop.json", stop)
            write_json(root / "status.json", dict(training_run=str(train)))
            write_json(train / "manifest.json", dict(status="stopped_by_user", planned_iterations=50000,
                       settings=dict(num_envs=60, num_steps=50, total_timesteps=150000000)))
            write_json(post / "plan.json", dict(policy_sha256="policy-hash", estimator_sha256="estimator-hash"))
            write_json(post / "source_sha256.json", {"experiments/interaction_adaptation/assess.py": "frozen-assess-code"})
            targets = dict(policy_evaluation=post / "policy/summary.json", estimator_evaluation=post / "estimator_evaluation.json",
                           retry_evaluation=post / "retries_vertical/summary.json", transfer_evaluation=post / "transfer_vertical/summary.json",
                           training_manifest=train / "manifest.json", early_stop_record=root / "early_stop.json")
            policy = dict(checkpoint="policy", checkpoint_sha256="policy-hash", metrics={},
                          paired_comparisons={name: dict(paired_bootstrap_95_interval=[-1, 1])
                                              for name in ("fixed_both", "wrong_mass", "wrong_friction")})
            estimator = dict(checkpoint="estimator", held_conditions=9, prior_median_mae=[.1, .2],
                             attempts={"5": dict(improvement_condition_bootstrap_95ci=[[-1, 1], [-1, 1]])})
            retry = dict(task="vertical", attempt_budget=3, methods=dict(adaptive_vs_fixed=dict(supported_on_this_sample=False)))
            transfer = dict(task="vertical", held_conditions=16, prior_median_mae=[.1, .2], attempts=estimator["attempts"])
            for name, value in (("policy_evaluation", policy), ("estimator_evaluation", estimator),
                                ("retry_evaluation", retry), ("transfer_evaluation", transfer)):
                write_json(targets[name], value)
            from experiments.interaction_adaptation.assess import assess
            report = assess(policy, estimator, retry, transfer=transfer)
            report.update(training=dict(status="stopped_by_user", requested_50000_iterations_completed=False,
                                        executed_rollout_cycles=3, saved_checkpoint_completed_cycles=2,
                                        latest_saved_checkpoint_sha256="policy-hash"),
                          sources={name: str(path) for name, path in targets.items()},
                          source_sha256={name: digest(path) for name, path in targets.items()},
                          assessment_source_sha256="frozen-assess-code")
            path = post / "assessment_vertical/assessment.json"
            write_json(path, report)
            path.with_suffix(".md").write_text("No benefit demonstrated; user stopped training.")
            result = audit(root, evaluation_root=post)
            row = next(r for r in result["requirements"] if r["id"] == "objective_assessment_vertical")
            self.assertEqual(row["status"], "proved", row)
            write_json(targets["policy_evaluation"], dict(changed_after_assessment=True))
            result = audit(root, evaluation_root=post)
            row = next(r for r in result["requirements"] if r["id"] == "objective_assessment_vertical")
            self.assertEqual(row["status"], "contradicted")
            self.assertIn("stale", row["reason"])

    def test_transfer_recomputation_matches_saved_numeric_prefix_keys_and_detects_error_tampering(self):
        from experiments.interaction_adaptation.transfer import summarize_transfer
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = dict(masses_kg=[.08, .16, .32, .48], frictions=[.25, .45, .65, .85],
                            attempts=5, seeds=list(range(64000, 64005)), horizon=50,
                            supplied_mass_kg=.1, supplied_friction=.4)
            rows = []
            for mi, mass in enumerate(protocol["masses_kg"]):
                for fi, friction in enumerate(protocol["frictions"]):
                    for attempt in range(1, 6):
                        rows.append(dict(condition=f"m{mi}f{fi}", attempt=attempt, seed=64000+attempt-1,
                            actual_mass_kg=mass, actual_friction=friction, supplied_mass_kg=.1, supplied_friction=.4,
                            estimate=dict(mass_kg=mass+.01, friction=friction+.1, informative_attempts=attempt,
                                          out_of_training_range=dict(mass_kg=False, friction=False))))
            hashes = dict(policy="policy", estimator="estimator")
            report = json.loads(json.dumps(summarize_transfer(rows, [.1, .4])))
            report.update(policy_checkpoint_sha256="policy", checkpoint_sha256="estimator", episodes=80)
            manifest = dict(completed_episodes=80, settings=dict(masses=protocol["masses_kg"], frictions=protocol["frictions"],
                            attempts=5, seed=64000, steps=50, supplied_mass=.1, supplied_friction=.4))
            (root / "attempts.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            target = "experiments.interaction_adaptation.early_stop_completion."
            with patch(target+"validate_collection_identity", return_value=(manifest, report, {})), patch(target+"collection_raw", return_value={}):
                self.assertEqual(validate_transfer(root, "vertical", protocol, hashes, {})["task"], "vertical")
                report["attempts"]["5"]["mae"][0] += .2
                with self.assertRaisesRegex(ValueError, "differs from recorded"):
                    validate_transfer(root, "vertical", protocol, hashes, {})


if __name__ == "__main__":
    unittest.main()
