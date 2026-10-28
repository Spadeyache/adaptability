"""Completion requires actual counters and matching artifacts, never good news."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.interaction_adaptation.completion import (
    AuditExpectations, POLICY_MODES, RETRY_MODES, audit, digest,
    validate_guarded_execution,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class CompletionTests(unittest.TestCase):
    def test_guard_metadata_cannot_hide_a_violating_accepted_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'trust_region.py').write_text('# audited helper')
            manifest=dict(status='complete',planned_iterations=2,optimization_semantics='test_guard',
                          trust_region=dict(enabled=True),trust_region_sha256=digest(root/'trust_region.py'),
                          settings=dict(guard_policy_updates=True,num_envs=60,num_steps=50,max_exact_kl=.05))
            rows=[dict(iteration=i,global_step=3000*i,rollout_observations=3000,effective_actions=5,
                       optimization_semantics='test_guard',max_accepted_exact_kl=.04,rollout_exact_kl=.02,
                       accepted_updates=1) for i in (1,2)]
            write_jsonl(root/'trust_region.jsonl',rows)
            self.assertEqual(validate_guarded_execution(root,manifest)['accepted_updates'],2)
            rows[0]['max_accepted_exact_kl']=.3
            write_jsonl(root/'trust_region.jsonl',rows)
            with self.assertRaisesRegex(ValueError,'exceeds the exact KL bound'):
                validate_guarded_execution(root,manifest)

    def fixture(self, root):
        expected = AuditExpectations(policy_seed_count=2, retry_conditions=1, retry_seed_blocks=2,
                                     transfer_masses=1, transfer_frictions=2, transfer_attempts=2,
                                     transfer_mass_values=(.16,),transfer_friction_values=(.25,.75))
        train, policy_eval, retry = root / "training", root / "final_evaluation", root / "retry_evaluation"
        train.mkdir(parents=True)
        weights = {name: torch.zeros((2, 46)) for name in
                   ("actor_mean.0.weight", "strength_mean.0.weight", "critic.0.weight")}
        policy = train / "final_ckpt.pt"
        torch.save(weights, policy)
        estimator = root / "estimator.pt"
        torch.save(dict(feature_names=["input__finger_width_m[0]"],
                        state_dict={"encoder.weight": torch.ones(2)}), estimator)
        write_json(train / "manifest.json", dict(status="complete", exit_code=0, observations=46,
            planned_iterations=50000, settings=dict(num_envs=60, num_steps=50, total_timesteps=150000000,
            condition_on_friction=True, friction_profile="independent_grid", masses_kg=[.1],
            cube_friction_coefficients=[.4])))
        (train / "console.log").write_text("Epoch: 50000, global_step=149997000\nSPS: 700\nmodel saved to " + str(policy) + "\n")
        write_json(root / "status.json", dict(status="complete", planned_iterations=50000,
            training_run=str(train), estimator_checkpoint=str(estimator), final_evaluation=str(policy_eval),
            retry_evaluation=str(retry), assessment=str(root / "assessment")))
        source = root / "source" / "example.py"
        source.parent.mkdir()
        source.write_text("# frozen source\n")
        write_json(root / "source_sha256.json", {"example.py": digest(source)})
        historical = dict(split="test", held_conditions=1, checkpoint_sha256=digest(estimator),
            conditions=[dict(target=[.1, .4])],
            attempts={str(k): dict(mae=[.1, .3]) for k in (1, 2, 3, 5)})
        write_json(root / "estimator_evaluation.json", historical)
        seeds = [91000, 91001]
        rows = [dict(mode=mode, seed=seed, mass_kg=.1, friction=.4, final_success=False)
                for mode in POLICY_MODES for seed in seeds]
        policy_report = dict(checkpoint_sha256=digest(policy), seeds=seeds,
            reset_fingerprints={str(seed): "fresh" for seed in seeds},
            masses_kg=[.1], frictions=[.4], horizon=50,
            metrics={mode: dict(final_success=0.) for mode in POLICY_MODES},
            paired_comparisons={mode: dict(metric="final_success", n_pairs=2, n_seed_blocks=2,
                                          paired_bootstrap_95_interval=[0., 0.]) for mode in POLICY_MODES[1:]})
        write_json(policy_eval / "summary.json", policy_report)
        write_json(policy_eval / "episodes.json", rows)
        traces = [dict(mode=mode, seed=seeds[0], step=step,
                       observation=[[0.] * 46], supplied=[[0.] * 46], action=[[0.] * 5])
                  for mode in ("true", "wrong_friction") for step in range(50)]
        write_jsonl(policy_eval / "traces.jsonl", traces)
        retry_rows = []
        for method in RETRY_MODES:
            for sequence in range(2):
                for attempt in range(1, 4):
                    identifier = f"{method}_{sequence}_{attempt}"
                    seed = 54000 + sequence * 3 + attempt - 1
                    row = dict(episode_id=identifier, method=method, condition_index=0,
                               sequence_index=sequence, attempt=attempt, seed=seed, success=False,
                               actual_mass_kg=.1, actual_friction=.4)
                    retry_rows.append(row)
                    self.raw_episode(retry / "episodes" / (identifier + ".npz"), row)
        write_jsonl(retry / "attempts.jsonl", retry_rows)
        write_json(retry / "manifest.json", dict(status="complete", completed_attempts=len(retry_rows),
            checkpoint_sha256=dict(policy=digest(policy), estimator=digest(estimator)),
            estimator_input_boundary="input__ fields and timestamp/frequency metadata only",
            settings=dict(seed=54000, sequences=2, conditions=[dict(mass_kg=.1, friction=.4)])))
        write_json(retry / "summary.json", dict(attempt_budget=3, methods={
            mode: dict(sequences=2, successes_within_budget=0) for mode in RETRY_MODES}))
        assessment = dict(policy=dict(status="not_demonstrated"), estimator=dict(status="partially_supported"),
                          closed_loop=dict(status="not_demonstrated"),
                          training=dict(requested_50000_iterations_completed=True, status="complete", planned_iterations=50000),
                          sources=dict(policy_evaluation=str(policy_eval / "summary.json"),
                              estimator_evaluation=str(root / "estimator_evaluation.json"),
                              retry_evaluation=str(retry / "summary.json"), training_manifest=str(train / "manifest.json")))
        write_json(root / "assessment" / "assessment.json", assessment)
        (root / "assessment" / "assessment.md").write_text("The evidence does not demonstrate a policy benefit.\n")
        return expected

    def raw_episode(self, path, row, fingerprint=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta__fresh_scene_reset=np.asarray(True),
            label__seed=np.asarray(row["seed"]), label__held_goal_success_final=np.asarray(row["success"]),
            label__mass_kg=np.asarray(row["actual_mass_kg"]),
            label__object_static_friction=np.asarray(row["actual_friction"]),
            label__object_dynamic_friction=np.asarray(row["actual_friction"]),
            label__reset_state_hash=np.asarray(fingerprint or str(row["seed"])))

    def requirement(self, result, identifier):
        return next(row for row in result["requirements"] if row["id"] == identifier)

    def test_negative_capability_results_allow_completed_research(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.fixture(root)
            result = audit(root, expectations=expected)
            self.assertTrue(result["overall_ready"], result)
            self.assertTrue(all(row["status"] == "proved" for row in result["requirements"]))
            conclusions = self.requirement(result, "final_objective_assessment")["evidence"]["conclusions"]
            self.assertEqual(conclusions["policy"], "not_demonstrated")

    def test_green_manifest_does_not_hide_missing_final_learning_cycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.fixture(root)
            (root / "training" / "console.log").write_text(
                "Epoch: 49999, global_step=149994000\nmodel saved to final_ckpt.pt\n")
            result = audit(root, expectations=expected)
            self.assertFalse(result["overall_ready"])
            self.assertEqual(self.requirement(result, "actual_training_execution")["status"], "contradicted")

    def test_evaluation_hash_mismatch_cannot_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.fixture(root)
            path = root / "final_evaluation" / "summary.json"
            report = json.loads(path.read_text())
            report["checkpoint_sha256"] = "another checkpoint"
            write_json(path, report)
            result = audit(root, expectations=expected)
            self.assertFalse(result["overall_ready"])
            self.assertEqual(self.requirement(result, "matched_final_policy_evaluation")["status"], "contradicted")

    def test_raw_retry_coverage_and_matched_initial_states_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.fixture(root)
            path = root / "retry_evaluation" / "episodes" / "adaptive_0_1.npz"
            row = next(row for row in read_rows(root / "retry_evaluation" / "attempts.jsonl")
                       if row["episode_id"] == "adaptive_0_1")
            self.raw_episode(path, row, fingerprint="different physics")
            result = audit(root, expectations=expected)
            self.assertFalse(result["overall_ready"])
            self.assertIn("different physical states", self.requirement(result, "actual_retry_evaluation")["reason"])
            path.unlink()
            result = audit(root, expectations=expected)
            self.assertIn("Raw episode files", self.requirement(result, "actual_retry_evaluation")["reason"])

    def test_running_training_is_incomplete_even_after_entering_final_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.fixture(root)
            path = root / "training" / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest.update(status="running", exit_code=None)
            write_json(path, manifest)
            result = audit(root, expectations=expected)
            self.assertFalse(result["overall_ready"])
            self.assertEqual(self.requirement(result, "actual_training_execution")["status"], "incomplete")

    def test_nonfinite_policy_cannot_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.fixture(root)
            path = root / "training" / "final_ckpt.pt"
            weights = torch.load(path, weights_only=True)
            weights["critic.0.weight"][0, 0] = float("nan")
            torch.save(weights, path)
            result = audit(root, expectations=expected)
            self.assertFalse(result["overall_ready"])
            self.assertEqual(self.requirement(result, "finite_46_input_final_policy")["status"], "contradicted")

    def test_stale_assessment_source_cannot_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.fixture(root)
            path = root / "assessment" / "assessment.json"
            report = json.loads(path.read_text())
            write_json(root / "pilot_summary.json", dict(checkpoint_sha256="old pilot"))
            report["sources"]["policy_evaluation"] = str(root / "pilot_summary.json")
            write_json(path, report)
            result = audit(root, expectations=expected)
            self.assertFalse(result["overall_ready"])
            self.assertIn("Assessment source policy_evaluation", self.requirement(result, "final_objective_assessment")["reason"])

    def test_missing_optional_combined_assessment_is_required_when_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.fixture(root)
            result = audit(root, combined_assessment=root / "combined", expectations=expected)
            self.assertFalse(result["overall_ready"])
            self.assertEqual(self.requirement(result, "combined_objective_assessment")["status"], "incomplete")

    def test_optional_transfer_adds_required_grid_hash_and_raw_evidence_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.fixture(root)
            directory = root / "transfer"
            hashes = dict(policy=digest(root / "training" / "final_ckpt.pt"), estimator=digest(root / "estimator.pt"))
            records = []
            for mu in (.25, .75):
                for attempt in (1, 2):
                    row = dict(episode_id=f"mu{mu}_{attempt}", seed=64000 + attempt - 1,
                        attempt=attempt, actual_mass_kg=.16, actual_friction=mu, success=False,
                        supplied_mass_kg=.1, supplied_friction=.4)
                    records.append(row)
                    self.raw_episode(directory / "episodes" / (row["episode_id"] + ".npz"), row)
            write_jsonl(directory / "attempts.jsonl", records)
            write_json(directory / "manifest.json", dict(status="complete", checkpoint_sha256=hashes,
                estimator_input_boundary="input__ plus timestamp/frequency metadata only", completed_episodes=4,
                settings=dict(masses=[.16], frictions=[.25, .75], attempts=2, seed=64000,
                              supplied_mass=.1, supplied_friction=.4)))
            path = directory / "summary.json"
            report = dict(policy_checkpoint_sha256=hashes["policy"], checkpoint_sha256=hashes["estimator"],
                          episodes=4, physical_conditions=2, attempts={"2": dict(mae=[.3, .3])})
            write_json(path, report)
            self.assertTrue(audit(root, transfer=directory, expectations=expected)["overall_ready"])
            report["checkpoint_sha256"] = "wrong estimator"
            write_json(path, report)
            result = audit(root, transfer=directory, expectations=expected)
            self.assertFalse(result["overall_ready"])
            self.assertEqual(self.requirement(result, "controlled_final_policy_estimator_transfer")["status"], "contradicted")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


if __name__ == "__main__":
    unittest.main()
