"""Regression checks for measured gripper-relative motion and transition alignment."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.lift_slide_belief.slip_data import (
    FIXED_GRIP_CAP_N, LoggedGripMotion, classify_lift, project_rows,
)


def _pose(position, rotation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return SimpleNamespace(p=np.asarray(position), q=np.r_[xyzw[3], xyzw[:3]],
                           to_transformation_matrix=lambda: transform)


class SlipMeasurementTests(unittest.TestCase):
    def test_corotating_cube_has_zero_relative_velocity_and_scalar_finger_errors(self):
        rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        tcp_position = np.array([.2, .1, .3])
        offset = np.array([.03, .04, .01])
        tcp_velocity = np.array([.2, -.1, .0])
        omega = np.array([0., 0., 2.])
        local_twist = np.r_[rotation.T @ tcp_velocity, rotation.T @ omega]
        joints = np.zeros(9)
        joints[-2:] = [.015, .017]
        motion = object.__new__(LoggedGripMotion)
        motion.tcp = SimpleNamespace(entity_pose=_pose(tcp_position, rotation))
        motion.robot = SimpleNamespace(qpos=joints, qvel=np.r_[local_twist, 0., 0., 0.])
        motion.ik = SimpleNamespace(end_link_idx=0, pmodel=SimpleNamespace(
            compute_single_link_local_jacobian=lambda q, i: np.eye(6, 9)))
        motion.body = SimpleNamespace(entity_pose=_pose(tcp_position+offset, rotation),
                                      linear_velocity=tcp_velocity+np.cross(omega, offset))
        motion.fingers = [SimpleNamespace(drive_target=np.array([.01]), force_limit=2.),
                          SimpleNamespace(drive_target=np.array([.02]), force_limit=2.)]
        links = [SimpleNamespace(_bodies=[SimpleNamespace(entity_pose=_pose(np.zeros(3), np.eye(3)))])
                 for _ in range(2)]
        contact = {id(links[0]): np.array([[0., 1., 0.]]), id(links[1]): np.array([[0., -2., 0.]])}
        motion.env = SimpleNamespace(agent=SimpleNamespace(finger1_link=links[0], finger2_link=links[1]),
                                     scene=SimpleNamespace(), control_freq=20)
        # SAPIEN returns a tensor; this small adapter supplies the tensor's row.numpy interface.
        class Contact:
            def __init__(self, values):
                self.values = values

            def __getitem__(self, index):
                return SimpleNamespace(numpy=lambda: self.values[index])

        motion.env.scene.get_pairwise_contact_forces = lambda link, obj: Contact(contact[id(link)])
        motion.obj = object()
        motion.sensor_states = []
        with patch("experiments.lift_slide_belief.slip_data.measure", return_value=np.zeros(43)):
            motion.measured_state(motion, None, None, False)
            measured = motion.sensor_states[-1]
            np.testing.assert_allclose(measured["relative_velocity"], 0., atol=1e-12)
            np.testing.assert_allclose(measured["relative_velocity_instantaneous"], 0., atol=1e-12)
            np.testing.assert_allclose(measured["relative_position"], rotation.T @ offset, atol=1e-12)
            self.assertEqual(measured["finger_target_position"].shape, (2,))
            self.assertEqual(measured["finger_position_error"].shape, (2,))
            np.testing.assert_allclose(measured["finger_position_error"], [-.005, .003], atol=1e-12)
            np.testing.assert_allclose(measured["finger_contact_normal_n"], [1., 2.])
            expected_local_slip = np.array([.01, .02, -.03])
            motion.body.linear_velocity += rotation @ expected_local_slip
            motion.measured_state(motion, None, None, False)
            np.testing.assert_allclose(motion.sensor_states[-1]["relative_velocity_instantaneous"],
                                       expected_local_slip, atol=1e-12)
            # Endpoint velocity changes alone do not invent interval displacement.
            np.testing.assert_array_equal(motion.sensor_states[-1]["relative_velocity"], np.zeros(3))
            local_displacement = np.array([.001, -.002, .003])
            motion.body.entity_pose.p += rotation @ local_displacement
            motion.measured_state(motion, None, None, False)
            np.testing.assert_allclose(motion.sensor_states[-1]["relative_velocity"],
                                       local_displacement * 20, atol=1e-12)
            np.testing.assert_array_equal(motion.sensor_states[0]["relative_velocity"], np.zeros(3))
            np.testing.assert_array_equal(motion.sensor_states[1]["relative_velocity"], np.zeros(3))

    def test_projection_keeps_measured_pre_and_post_values_in_their_own_steps(self):
        state = np.zeros((4, 43), dtype=np.float32)
        state[:, 7:9] = np.arange(8).reshape(4, 2) * .001
        targets = np.arange(8, dtype=np.float32).reshape(4, 2) * .002
        timeline = dict(state=state, relative_position=np.arange(12).reshape(4, 3),
                        relative_velocity=np.arange(12, 24).reshape(4, 3),
                        finger_position_error=targets - state[:, 7:9])
        rows = {prefix+key: values[selection] for prefix, selection in (("", slice(None, -1)), ("next_", slice(1, None)))
                for key, values in timeline.items()}
        rows["true_mass"] = np.full(3, 999.)
        projected = project_rows(rows)
        self.assertEqual(projected["state"].shape, (3, 51))
        np.testing.assert_array_equal(projected["state"][:, -2:], (targets-state[:, 7:9])[:-1])
        np.testing.assert_array_equal(projected["next_state"][:, -2:], (targets-state[:, 7:9])[1:])
        np.testing.assert_array_equal(projected["next_state"][:-1], projected["state"][1:])
        self.assertFalse(np.any(projected["state"] == 999.))

    def test_requested_cap_matches_action_and_realized_drive_before_and_after_switch(self):
        motion = object.__new__(LoggedGripMotion)
        motion.force_range = (.25, 40.)
        motion.grip_cap_n = .5
        motion.cap_switch_step = 1
        motion.step_index = 0
        motion.issued_caps = []
        motion.fingers = [SimpleNamespace(force_limit=40.), SimpleNamespace(force_limit=40.)]

        def apply_action(_, action):
            actual_cap = .25 + (float(action[4]) + 1) * .5 * 39.75
            for joint in motion.fingers:
                joint.force_limit = actual_cap
            return np.zeros((5, 3))

        with patch("experiments.lift_slide_belief.scripted.ScriptedMotion.step", new=apply_action):
            for requested in (FIXED_GRIP_CAP_N, .5):
                action = np.zeros(8, dtype=np.float32)
                motion.step(action)
                self.assertAlmostEqual(.25 + (float(action[4]) + 1) * .5 * 39.75, requested, places=5)
                np.testing.assert_allclose(motion.issued_caps[-1], requested, atol=2e-6, rtol=0)


class SlipEventTests(unittest.TestCase):
    @staticmethod
    def episode():
        states = np.zeros((8, 43), dtype=np.float32)
        states[:, 21] = [.02, .027, .03, .034, .04, .04, .035, .03]
        states[:, 22:28] = [1, 0, 0, 0, 1, 0]
        states[5:, 30] = -.02
        tcp = np.zeros((8, 7), dtype=np.float32)
        tcp[:, 2] = states[:, 21]
        tcp[5:, 2] = .05
        initial = np.vstack((states[:1], states[:-1]))
        return dict(state=initial, next_state=states, next_tcp_pose=tcp,
                    tcp_pose=np.vstack((tcp[:1], tcp[:-1])),
                    next_tcp_linear_velocity=np.zeros((8, 3)),
                    next_relative_velocity=np.zeros((8, 3)),
                    next_finger_contact_normal_n=np.ones((8, 2)),
                    next_finger_position_error=np.zeros((8, 2)))

    def test_failed_pickup_never_becomes_acquired_slip(self):
        rows = self.episode()
        rows["next_finger_contact_normal_n"][:, 1] = 0.
        outcome = classify_lift(rows)
        self.assertEqual(outcome["outcome"], "failed_pickup")
        self.assertEqual(outcome["acquisition_step"], -1)
        self.assertEqual(outcome["slip_step"], -1)

    def test_acquisition_requires_sustained_airborne_bilateral_contact_before_slip(self):
        outcome = classify_lift(self.episode())
        self.assertEqual(outcome["acquisition_step"], 3)
        self.assertEqual(outcome["slip_step"], 5)
        self.assertEqual(outcome["outcome"], "slip_after_acquisition")

    def test_rotation_involved_grasp_loss_has_its_own_label(self):
        rows = self.episode()
        rotation = Rotation.from_euler("x", 30, degrees=True).as_matrix()
        rows["next_state"][5:, 22:28] = rotation[:, :2].T.ravel()
        outcome = classify_lift(rows)
        self.assertEqual(outcome["outcome"], "rotating_loss_after_acquisition")
        self.assertAlmostEqual(outcome["cube_rotation_at_first_slip_deg"], 30., places=4)


if __name__ == "__main__":
    unittest.main()
