"""Check that the force oracle uses full XY error and physical rest for success."""
import unittest

import numpy as np

from experiments.lift_slide_belief.oracle_diagnostics import oracle_case


class OracleDiagnosticTest(unittest.TestCase):
    def test_lateral_error_changes_the_best_force(self):
        episode = dict(cube_id=200, episode_id=0, initial_state=np.zeros(43), direction=np.array([1., 0.]))
        variants = [dict(endpoint_x_m=x, endpoint_y_m=y, actual_stopped=stopped,
                         signed_distance_m=x, lateral_distance_m=y)
                    for x, y, stopped in ((.010, .004, True), (.008, 0., True))]
        result = oracle_case(episode, variants, .010)
        self.assertEqual(result['oracle_candidate_index'], 1)
        self.assertAlmostEqual(result['oracle_error_m'], .002)
        self.assertTrue(result['oracle_success'])

    def test_an_unstopped_exact_hit_does_not_establish_success(self):
        episode = dict(cube_id=200, episode_id=0, initial_state=np.zeros(43), direction=np.array([1., 0.]))
        variants = [dict(endpoint_x_m=x, endpoint_y_m=0., actual_stopped=stopped,
                         signed_distance_m=x, lateral_distance_m=0.)
                    for x, stopped in ((.010, False), (.005, True))]
        result = oracle_case(episode, variants, .010)
        self.assertEqual(result['oracle_error_m'], 0.)
        self.assertFalse(result['oracle_success'])
        self.assertFalse(result['oracle_selected_stopped'])
        self.assertAlmostEqual(result['best_stopped_error_m'], .005)
