"""Transfer assessment preserves conditions and compares equal evidence budgets."""
import unittest

from experiments.interaction_adaptation.transfer import summarize_transfer


class TransferSummaryTests(unittest.TestCase):
    def test_conditions_remain_units_and_more_evidence_has_paired_comparison(self):
        records = []
        for condition, target in (("a", [.1, .2]), ("b", [.3, .8])):
            for count in (1, 2, 3, 5):
                error = .1 if count == 1 else .02
                records.append(dict(condition=condition, attempt=count, actual_mass_kg=target[0],
                                    actual_friction=target[1], estimate=dict(
                                        mass_kg=target[0] + error, friction=target[1] + error,
                                        informative_attempts=count,
                                        out_of_training_range=dict(mass_kg=False, friction=False))))
        report = summarize_transfer(records, [.6, .9])
        self.assertEqual(report["held_conditions"], 2)
        self.assertEqual(len(report["conditions"]), 2)
        self.assertEqual(report["bootstrap"]["unit"], "physical condition")
        self.assertAlmostEqual(report["attempts"][5]["mae"][0], .02)
        self.assertAlmostEqual(report["attempts"][5]["improvement_over_one_attempt"][0], .08)
        self.assertTrue(all(report["attempts"][5]["supported_over_one_attempt"]))
        self.assertTrue(all(report["attempts"][5]["supported_over_prior"]))


if __name__ == "__main__":
    unittest.main()
