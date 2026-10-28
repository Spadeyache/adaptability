"""Exercise long-run stage accounting without launching simulation or learning."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.physics_conditioning import run_long


class LongPipelineTests(unittest.TestCase):
    def guarded_fixture(self,root):
        pilot=root/'final_ckpt.pt'
        pilot.write_bytes(b'guarded pilot')
        source=run_long.ROOT/'robot_adaptation/policy/trust_region.py'
        (root/'trust_region.py').write_bytes(source.read_bytes())
        manifest=dict(planned_iterations=100,optimization_semantics='guarded_test',
            trust_region=dict(enabled=True),trust_region_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            settings=dict(guard_policy_updates=True,num_envs=60,num_steps=50,max_exact_kl=.05))
        rows=[dict(iteration=i,global_step=i*3000,rollout_observations=3000,effective_actions=5,
                   optimization_semantics='guarded_test',max_accepted_exact_kl=.04,rollout_exact_kl=.03,
                   accepted_updates=1) for i in range(1,101)]
        path=root/'trust_region.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return pilot,manifest,rows,path

    def test_corrective_pilot_requires_actual_bounded_learning_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            pilot,manifest,rows,path=self.guarded_fixture(Path(temporary))
            self.assertEqual(run_long.validate_guarded_pilot(pilot,manifest)['accepted_updates'],100)
            rows[20]['max_accepted_exact_kl']=.2
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'beyond the exact KL bound'):
                run_long.validate_guarded_pilot(pilot,manifest)

    def test_corrective_pilot_cannot_pass_with_no_updates_or_changed_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            pilot,manifest,rows,path=self.guarded_fixture(Path(temporary))
            for row in rows:row['accepted_updates']=0
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'no policy learning'):
                run_long.validate_guarded_pilot(pilot,manifest)
            (pilot.parent/'trust_region.py').write_text('# changed implementation')
            with self.assertRaisesRegex(ValueError,'implementation changed'):
                run_long.validate_guarded_pilot(pilot,manifest)

    def fixture(self, root):
        pilot = root / "pilot"
        pilot.mkdir()
        checkpoint = pilot / "final_ckpt.pt"
        checkpoint.write_bytes(b"pinned pilot checkpoint")
        (pilot / "manifest.json").write_text(json.dumps(dict(status="complete", observations=46,
            settings=dict(num_envs=60, num_steps=50))))
        diagnostic = root / "pilot_evaluation.json"
        diagnostic.write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            reset_fingerprints={"81000": "verified"}, seeds=[81000])))
        estimator = root / "best.pt"
        estimator.write_bytes(b"pinned estimator")
        metrics = root / "estimator_metrics.json"
        metrics.write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(estimator.read_bytes()).hexdigest(),
            conditions=[dict(target=[.1, .4])], checkpoint=str(estimator))))
        output = root / "pipeline"
        arguments = ["--pilot-checkpoint", str(checkpoint), "--pilot-evaluation", str(diagnostic),
                     "--estimator-checkpoint", str(estimator), "--estimator-evaluation", str(metrics),
                     "--output", str(output)]
        def prepare(settings, prepare_only):
            self.assertTrue(prepare_only)
            run = root / "training"
            run.mkdir()
            (run / "manifest.json").write_text(json.dumps(dict(status="prepared", settings=settings)))
            (run / "launch_command.json").write_text(json.dumps(dict(command=["fake-training"],
                cwd=str(root), environment_overrides={})))
            (run / "final_ckpt.pt").write_bytes(b"mock training result")
            return run
        return arguments, output, prepare

    def test_complete_status_requires_training_and_every_evaluation_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, output, prepare = self.fixture(root)
            def completed_stage(command,**kwargs):
                if command==["fake-training"]:
                    return
                stage=Path(command[command.index("--output")+1])
                stage.mkdir()
                if "experiments.physics_conditioning.evaluate" in command:
                    data=dict(checkpoint_sha256=hashlib.sha256(b"mock training result").hexdigest(),
                              metrics={},paired_comparisons={},reset_fingerprints={"verified": True})
                elif "experiments.interaction_adaptation" in command:
                    data=dict(attempt_budget=3,methods=dict(fixed={},adaptive={},true={}))
                else:
                    (stage/"assessment.md").write_text("Completed report")
                    (stage/"assessment.json").write_text(json.dumps(dict(policy={},estimator={},closed_loop={},training={})))
                    return
                (stage/"summary.json").write_text(json.dumps(data))
            with patch("robot_adaptation.policy.train.main", side_effect=prepare), \
                 patch.object(run_long, "run_command",side_effect=completed_stage) as commands:
                run_long.main(arguments)
            self.assertEqual(commands.call_count, 4)
            stages = [call.args[0] for call in commands.call_args_list]
            self.assertEqual(stages[0], ["fake-training"])
            self.assertIn("experiments.physics_conditioning.evaluate", stages[1])
            self.assertIn("experiments.interaction_adaptation", stages[2])
            self.assertIn("experiments.interaction_adaptation.assess", stages[3])
            self.assertEqual(stages[1][stages[1].index("--seed-start") + 1], "91000")
            self.assertTrue((output / "source" / "robot_adaptation" / "policy" / "model.py").exists())
            state = json.loads((output / "status.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["settings"]["total_timesteps"], 150_000_000)
            self.assertEqual(state["settings"]["eval_freq"], 500)

    def test_evaluation_failure_keeps_training_completion_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, output, prepare = self.fixture(root)
            with patch("robot_adaptation.policy.train.main", side_effect=prepare), \
                 patch.object(run_long, "run_command", side_effect=[None, RuntimeError("evaluation failure")]):
                with self.assertRaisesRegex(RuntimeError, "evaluation failure"):
                    run_long.main(arguments)
            state = json.loads((output / "status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["failed_stage"], "evaluating_policy")
            self.assertIsNone(state["assessment"])
            training = json.loads((root / "training" / "manifest.json").read_text())
            self.assertEqual(training["status"], "complete")

    def test_zero_exit_without_evaluation_artifacts_does_not_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            arguments,output,prepare=self.fixture(root)
            with patch("robot_adaptation.policy.train.main",side_effect=prepare), \
                 patch.object(run_long,"run_command"):
                with self.assertRaises(FileNotFoundError):
                    run_long.main(arguments)
            state=json.loads((output/"status.json").read_text())
            self.assertEqual(state["status"],"failed")
            self.assertEqual(state["failed_stage"],"evaluating_policy")


if __name__ == "__main__":
    unittest.main()
