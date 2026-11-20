"""Report safety checks use synthetic metadata and never write real results."""
import tempfile
import unittest
from pathlib import Path

from clean.report import generate_report, mean_std, ranked_changes, status, _reproduction_commands, _old_regime_commands


class ReportTests(unittest.TestCase):
    def test_unfinished_final_cannot_replace_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "RESULTS.md"
            output.write_text("Existing evidence must be retained.\n")
            with self.assertRaisesRegex(FileNotFoundError, "Final evaluation is not complete"):
                generate_report(root)
            self.assertEqual(output.read_text(), "Existing evidence must be retained.\n")

    def test_undefined_statistics_and_unestablished_checks_remain_explicit(self):
        self.assertEqual(mean_std({"mean": None, "std": None}), "undefined")
        self.assertEqual(mean_std({"mean": float("nan"), "std": 0}), "undefined")
        self.assertEqual(status(None), "NOT ESTABLISHED")
        self.assertEqual(status(False), "FAIL")

    def test_cross_distribution_scores_are_never_ranked_as_model_improvements(self):
        def saved(iteration, base, cap, audit, weight, error):
            return {"iteration": iteration, "base_iteration": base,
                "config": {"grip_force_max": cap, "moving_weight": weight},
                "audit_sha256": audit, "summary": {"fitted_z": {"one_step_nmse": {"mean": error}}},
                "runs": [{"seed": seed, "metrics": {"fitted_z": {"one_step_nmse": error}}} for seed in range(5)]}
        first = saved(1, None, 30, "old", 1, 1)
        second = saved(2, 1, 30, "old", 10, 2)
        narrowed = saved(3, 1, 8, "new", 1, .1)
        result = ranked_changes([first, second, narrowed])
        self.assertEqual([change["current"]["iteration"] for change in result], [2])

    def test_generated_python_command_blocks_parse_without_execution(self):
        for lines in (_reproduction_commands(Path("/tmp/project/clean"), 3), _old_regime_commands(Path("/tmp/project/clean"))):
            start = next(index for index, line in enumerate(lines) if line.endswith("<<'PY'"))
            end = lines.index("PY", start)
            compile("\n".join(lines[start + 1:end]), "<report-reproduction>", "exec")


if __name__ == "__main__":
    unittest.main()
