"""Pure synthetic decisions/schedules; no experiment files or episodes opened."""
import copy
import unittest

from clean.search import (OPTIONS, FALLBACK_OPTIONS, REQUIRED_MODELS, METRICS,
    all_success, choose_proposal, decide_candidate, eligibility, select_reference, _plan_marker)


CONFIG = dict(width=128, depth=2, z_dim=2, lr=.001, z_l2=.0001, fit_steps=200,
    fit_lr=.02, moving_weight=1.0, prediction_horizon=1, grip_force_max=8.0)


def saved(number, error, *, f4=False, f5=True, success=False):
    score = {"mean": error, "std": .01, "values": [error] * 5}
    return {"iteration": number, "config": dict(CONFIG),
        "summary": {name: {metric: copy.deepcopy(score) for metric in METRICS} for name in REQUIRED_MODELS},
        "failure_checks": {f"F{i}": (f4 if i == 4 else f5 if i == 5 else True) for i in range(1, 10)},
        "success": {**{f"S{i}": success for i in range(1, 5)}, "plain_oracle_gap": -.1}}


class SearchTests(unittest.TestCase):
    def test_exactly_one_setting_changes_and_schedule_does_not_repeat_config(self):
        base = saved(3, 1.)
        tested = [base["config"]]
        used = []
        for number in range(4, 21):
            plan = choose_proposal(base, tested, used)
            changed = [key for key in base["config"] if plan["config"][key] != base["config"][key]]
            self.assertEqual(changed, [plan["changed_key"]])
            self.assertNotIn(plan["config"], tested)
            self.assertEqual(plan["config"]["grip_force_max"], 8.)
            tested.append(plan["config"])
            used.append(plan["option_id"])
            # Alternate hypothetical accepted/reverted proposals to ensure the
            # next option is always based on the actual retained configuration.
            if number % 2 == 0:
                base = {**base, "iteration": number, "config": plan["config"]}
        self.assertEqual(len(used), 17)

    def test_selection_uses_locked_nmse_after_validity_checks(self):
        base = saved(3, 1.)
        invalid = saved(4, .1, f5=False)
        improved = saved(5, .9)
        selected, _ = select_reference([base, invalid, improved])
        self.assertEqual(selected["iteration"], 5)
        retained, action, reason = decide_candidate(base, invalid)
        self.assertEqual(retained["iteration"], 3)
        self.assertEqual(action, "reverted")
        self.assertIn("F5", reason)
        retained, action, reason = decide_candidate(base, improved)
        self.assertEqual(action, "kept")
        self.assertIn("failed reference", reason)
        corrupted = saved(6, .01)
        corrupted["summary"]["history"]["rollout_error"]["mean"] = None
        self.assertFalse(eligibility(corrupted)[0])

    def test_success_stops_even_when_another_nmse_is_lower(self):
        base = saved(3, .5)
        successful = saved(4, .6, f4=True, success=True)
        retained, action, _ = decide_candidate(base, successful)
        self.assertEqual(retained["iteration"], 4)
        self.assertEqual(action, "success")
        self.assertTrue(all_success(retained))

    def test_required_f4_feasibility_precedes_nmse_reference_comparison(self):
        lower_error_failed = saved(4, .2)
        higher_error_feasible = saved(6, .4, f4=True)
        other_feasible = saved(8, .5, f4=True)
        f5_failed = saved(9, .1, f4=True, f5=False)
        selected, reason = select_reference([lower_error_failed, other_feasible, higher_error_feasible, f5_failed])
        self.assertEqual(selected["iteration"], 6)
        self.assertIn("fixed required constraint", reason)
        retained, action, reason = decide_candidate(lower_error_failed, higher_error_feasible)
        self.assertEqual((retained["iteration"], action), (6, "kept"))
        self.assertIn("not a metric change", reason)
        retained, action, reason = decide_candidate(higher_error_feasible, lower_error_failed)
        self.assertEqual((retained["iteration"], action), (6, "reverted"))
        self.assertIn("fails required F4", reason)
        improved_feasible = saved(10, .3, f4=True)
        retained, action, _ = decide_candidate(higher_error_feasible, improved_feasible)
        self.assertEqual((retained["iteration"], action), (10, "kept"))

    def test_failed_f4_reference_is_provisional_only_when_no_feasible_one_exists(self):
        higher = saved(3, .7)
        lower = saved(4, .6)
        selected, reason = select_reference([higher, lower])
        self.assertEqual(selected["iteration"], 4)
        self.assertIn("only as a failed reference", reason)
        self.assertFalse(selected["failure_checks"]["F4"])

    def test_superseded_untrained_plan_gets_a_distinct_command_log_marker(self):
        plan = {"audit_sha256": "same-audit", "base_iteration": 9,
            "option_id": "width=32", "config": {**CONFIG, "width": 32}}
        old = _plan_marker("command", 11, plan)
        corrected = {**plan, "base_iteration": 6, "config": {**plan["config"], "z_dim": 16}}
        new = _plan_marker("command", 11, corrected)
        self.assertNotEqual(old, new)
        self.assertNotEqual(new, "<!-- search-command-11 -->")
        self.assertIn("base-06", new)

    def test_failed_f5_fallback_is_explicit_and_never_allows_leakage(self):
        invalid = saved(3, .5, f5=False)
        with self.assertRaises(RuntimeError):
            select_reference([invalid])
        selected, reason = select_reference([invalid], allow_failed_f5=True)
        self.assertEqual(selected["iteration"], 3)
        self.assertIn("failed F5", reason)
        invalid["failure_checks"]["F1"] = False
        with self.assertRaises(RuntimeError):
            select_reference([invalid], allow_failed_f5=True)

    def test_explicit_fallbacks_are_used_after_primary_options(self):
        import json
        base = saved(3, 1.)
        used = [f"{key}={json.dumps(value)}" for key, value, _ in OPTIONS]
        proposal = choose_proposal(base, [base["config"]], used)
        self.assertTrue(proposal["fallback_option"])
        self.assertEqual(proposal["changed_key"], FALLBACK_OPTIONS[0][0])
        self.assertEqual(proposal["new_value"], FALLBACK_OPTIONS[0][1])


if __name__ == "__main__":
    unittest.main()
