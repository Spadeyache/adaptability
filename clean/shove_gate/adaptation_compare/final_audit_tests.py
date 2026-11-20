"""Synthetic boundary checks for the independent report audit arithmetic."""
import unittest

from . import final_audit as audit


class AuditArithmeticTests(unittest.TestCase):
    def test_p1_requires_every_return(self):
        base = [1., 1., 1., 1., 1.]
        a = {n: base for n in audit.RETURNS}
        b = {n: [2.] * 5 for n in audit.RETURNS}
        self.assertTrue(audit.p1_arithmetic(base, a, base, b)['passed'])
        b[16] = base
        self.assertFalse(audit.p1_arithmetic(base, a, base, b)['passed'])

    def test_p1_equality_is_not_beyond_spread(self):
        base = [1.] * 5
        unchanged = {n: base for n in audit.RETURNS}
        result = audit.p1_arithmetic(base, unchanged, base, unchanged)
        self.assertFalse(result['passed'])
        self.assertTrue(result['by_return'][6]['a_nonrise'])
        self.assertFalse(result['by_return'][6]['b_rise_beyond_seed_spread'])

    def test_p1_a_tolerance_is_fixed_per_seed(self):
        base = [1.] * 5
        a = {n: [1. + 1e-11] * 5 for n in audit.RETURNS}
        b = {n: [2.] * 5 for n in audit.RETURNS}
        self.assertTrue(audit.p1_arithmetic(base, a, base, b)['passed'])
        a[30] = [1. + 1e-8] * 5
        self.assertFalse(audit.p1_arithmetic(base, a, base, b)['passed'])

    def test_p2_earliest_prefix_and_exact_match(self):
        a = {n: [2. - .2*n] * 5 for n in audit.PREFIXES}
        self.assertEqual(audit.p2_arithmetic(a, [1.4] * 5)['earliest_prefix'], 3)
        self.assertEqual(audit.p2_arithmetic(a, [1.] * 5)['earliest_prefix'], 5)

    def test_p2_overlapping_spread_does_not_rescue_worse_mean(self):
        a = {n: [0., 0., 0., 0., 6.] for n in audit.PREFIXES}
        result = audit.p2_arithmetic(a, [1.] * 5)
        self.assertFalse(result['passed'])
        self.assertIsNone(result['earliest_prefix'])

    def test_missing_seed_is_not_silently_accepted(self):
        with self.assertRaises(AssertionError):
            audit.moments([1.] * 4)

    def test_all_requested_returns_have_unambiguous_visit_indices(self):
        self.assertEqual([x - 1 for x in audit.RETURNS], [5, 15, 29])
        self.assertTrue(set(audit.RETURNS) <= set(audit.SNAPSHOTS))


if __name__ == '__main__':
    unittest.main()
