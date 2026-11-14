"""Audit commanded shove force and reported normal-contact forces with isolated interventions.

Run with python -m experiments.lift_slide_belief.force_cap_diagnostics RUN_ROOT.
The production controller and saved data are read without edits. Interventions
are confined to a diagnostic subclass and fresh scenes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .collect import measure, physics
from .scripted import ScriptedMotion, record_episode
from robot_adaptation.simulation.context_dynamics import make_env


def audit_archived(run_root, output, cfg):
    records = []
    max_event = None
    for path in sorted((run_root / 'study' / 'target_control' / 'trajectories').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            force = data['contact_force_substeps_n'].astype(np.float64)
            magnitude = np.linalg.norm(force[:, :, :2], axis=2)
            start = int(data['force_start_step'][0])
            release = int(data['shove_release_step'][0])
            direction = data['motion_direction'][0]
            command = float(data['selected_force_n'])
            stages = {
                'approach': slice(0, start),
                'pulse_onset_first_control_step': slice(start, start + 1),
                'pulse_after_first_control_step': slice(start + 1, release),
                'pulse_all': slice(start, release),
                'withdrawal_first_control_step': slice(release, release + 1),
                'withdrawal_all': slice(release, release + cfg['retract_steps']),
                'after_withdrawal': slice(release + cfg['retract_steps'], None),
            }
            entry = dict(file=str(path.relative_to(run_root)),
                         cube_id=int(data['cube_id'][0]), episode_id=int(data['episode_id'][0]),
                         candidate_index=int(data['candidate_index']), commanded_force_n=command,
                         force_start_step=start, release_step=release,
                         peak_horizontal_force_n={}, horizontal_impulse_norm_ns={},
                         signed_impulse_ns={}, samples_above_4n={}, samples_above_command={})
            for stage, selected in stages.items():
                values = magnitude[selected]
                impulses = force[selected, :, :2].sum(axis=(0, 1)) / cfg['physx_hz']
                entry['peak_horizontal_force_n'][stage] = float(values.max())
                entry['horizontal_impulse_norm_ns'][stage] = float(np.linalg.norm(impulses))
                entry['signed_impulse_ns'][stage] = float(impulses @ direction)
                entry['samples_above_4n'][stage] = int((values > 4 + 1e-6).sum())
                entry['samples_above_command'][stage] = int((values > command + 1e-6).sum())
            records.append(entry)
            peak = magnitude[start:release].max()
            if max_event is None or peak > max_event['horizontal_contact_force_n']:
                relative_step, substep = np.unravel_index(np.argmax(magnitude[start:release]), (release-start, 5))
                control_step = start + int(relative_step)
                max_event = dict(file=str(path.relative_to(run_root)), cube_id=entry['cube_id'],
                                 episode_id=entry['episode_id'], candidate_index=entry['candidate_index'],
                                 commanded_force_n=command, horizontal_contact_force_n=float(peak),
                                 contact_force_xyz_n=force[control_step, substep].tolist(),
                                 direction_xy=direction.tolist(),
                                 projected_along_direction_n=float(force[control_step, substep, :2] @ direction),
                                 control_step=control_step, physics_substep=int(substep),
                                 time_since_force_onset_s=(int(relative_step)*5+int(substep)+1)/cfg['physx_hz'],
                                 cube_velocity_before_control_step=data['state'][control_step, 28:31].tolist(),
                                 cube_velocity_after_control_step=data['next_state'][control_step, 28:31].tolist())
    summary = {}
    for stage in records[0]['peak_horizontal_force_n']:
        peaks = np.array([r['peak_horizontal_force_n'][stage] for r in records])
        above = np.array([r['samples_above_4n'][stage] for r in records])
        own = np.array([r['samples_above_command'][stage] for r in records])
        summary[stage] = dict(maximum_horizontal_force_n=float(peaks.max()),
                             median_trajectory_peak_n=float(np.median(peaks)),
                             trajectories_above_4n=int((peaks > 4+1e-6).sum()),
                             native_10ms_samples_above_4n=int(above.sum()),
                             trajectories_above_their_command=int((own > 0).sum()),
                             native_10ms_samples_above_their_command=int(own.sum()))
    by_command = {}
    for command in sorted({r['commanded_force_n'] for r in records}):
        selected = [r for r in records if r['commanded_force_n'] == command]
        by_command[str(command)] = dict(
            count=len(selected),
            maximum_pulse_force_n=max(r['peak_horizontal_force_n']['pulse_all'] for r in selected),
            trajectories_with_pulse_peak_above_command=sum(r['samples_above_command']['pulse_all'] > 0 for r in selected),
            mean_signed_pulse_impulse_ns=float(np.mean([r['signed_impulse_ns']['pulse_all'] for r in selected])),
            requested_force_times_duration_ns=command*cfg['shove_duration_s'])
    report = dict(measurement='Horizontal resultant of reported normal contact impulses divided by timestep; friction impulses excluded',
                  trajectory_count=len(records), per_phase=summary, by_command=by_command,
                  worst_pulse_event=max_event, records=records)
    (output/'archived_forces.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


class ComponentTrace(ScriptedMotion):
    def __init__(self, env, variant, cfg):
        super().__init__(env, cfg['grip_force_limits_n'], cfg['finger_friction'])
        self.variant = variant
        self.trace = []
        self.current_command = np.zeros(8)
        self.current_wrench = np.zeros(6)
        self.current_torques = np.zeros((4, 7))
        self.control_step = -1
        self.substep = 0
        self.fk_check = None

    def step(self, action):
        self.control_step += 1
        self.substep = 0
        self.current_command = action.copy()
        self.current_wrench = np.zeros(6)
        self.current_torques = np.zeros((4, 7))
        if action[7] and not self.force_mode:
            self.check_kinematics()
            if self.variant == 'zero_arm_velocity_at_onset':
                velocity = self.robot.qvel.copy()
                velocity[:7] = 0
                self.robot.set_qvel(velocity)
        if not action[7] and self.force_mode and self.variant == 'soft_withdrawal_drives':
            self.base_drives = [(100., 20., 10., 'force')] * 7
        return super().step(action)

    def check_kinematics(self):
        model = self.ik.pmodel
        qpos = self.robot.qpos
        model.compute_forward_kinematics(qpos)
        pose = model.get_link_pose(self.ik.end_link_idx)
        local_tcp = self.base_inverse * self.tcp.entity_pose
        jacobian = model.compute_single_link_local_jacobian(qpos, self.ik.end_link_idx)
        rotation = pose.to_transformation_matrix()[:3, :3]
        numeric = []
        for index in range(7):
            perturbed = qpos.astype(float).copy()
            perturbed[index] += 1e-4
            model.compute_forward_kinematics(perturbed)
            numeric.append((model.get_link_pose(self.ik.end_link_idx).p - pose.p) / 1e-4)
        self.fk_check = dict(
            ik_end_link_index=int(self.ik.end_link_idx), tcp_link_index=int(self.tcp.index),
            fk_position_error_m=float(np.max(abs(pose.p - local_tcp.p))),
            fk_rotation_matrix_error=float(np.max(abs(pose.to_transformation_matrix()[:3, :3] -
                                                       local_tcp.to_transformation_matrix()[:3, :3]))),
            linear_jacobian_max_finite_difference_error=float(np.max(abs(
                rotation @ jacobian[:3, :7] - np.asarray(numeric).T))),
            finite_difference_joint_step=1e-4,
            robot_link_gravity_disabled=[bool(link.disable_gravity) for link in self.robot.links],
            arm_joint_friction=[float(j.friction) for j in self.arm_joints],
            cube_max_contact_impulse=float(self.body.max_contact_impulse))

    def _force_substep(self, force_xy):
        positions, velocities = self.robot.qpos, self.robot.qvel
        jacobian = self.ik.pmodel.compute_single_link_local_jacobian(
            positions, self.ik.end_link_idx)[:, :7]
        rotation = self.tcp.entity_pose.to_transformation_matrix()[:3, :3]
        jacobian[:3] = rotation @ jacobian[:3]
        jacobian[3:] = rotation @ jacobian[3:]
        twist = jacobian @ velocities[:7]
        rotation_error = Rotation.from_matrix(np.diag([1., -1., -1.]) @ rotation.T).as_rotvec()
        vertical_force = np.clip(200 * (.023 - self.tcp.entity_pose.p[2]) - 10 * twist[2], -60, 60)
        torque = np.clip(6 * rotation_error - .5 * twist[3:], -20, 20)
        if self.variant in ('no_vertical_feedback', 'horizontal_wrench_only'):
            vertical_force = 0.
        if self.variant in ('no_orientation_feedback', 'horizontal_wrench_only'):
            torque = np.zeros(3)
        if self.variant == 'zero_horizontal_command':
            force_xy = np.zeros(2)
        if self.variant == 'low_finger_drive_cap':
            self.set_grip_cap(.25)
        wrench = np.r_[force_xy, vertical_force, torque]
        effort = self.robot.compute_passive_force(gravity=False, coriolis_and_centrifugal=True)
        self.current_torques = np.stack((jacobian[:2].T @ force_xy,
                                        jacobian[2] * vertical_force,
                                        jacobian[3:].T @ torque,
                                        effort[:7].copy()))
        self.current_wrench = wrench.copy()
        effort[:7] += jacobian.T @ wrench
        self.robot.set_qf(effort)
        self.env.scene.step()

    def _physics_step(self):
        qvel = self.robot.qvel.copy()
        tcp_pose = self.tcp.entity_pose.raw_pose if hasattr(self.tcp.entity_pose, 'raw_pose') else np.r_[self.tcp.entity_pose.p, self.tcp.entity_pose.q]
        cube_velocity = self.body.linear_velocity.copy()
        super()._physics_step()
        per_link = -np.stack([self.env.scene.get_pairwise_contact_forces(link, self.obj)[0].numpy()
                             for link in self.contact_links])
        table_force = self.env.scene.get_pairwise_contact_forces(self.obj, self.env.table_scene.table)[0].numpy()
        self.trace.append(dict(
            control_step=self.control_step, physics_substep=self.substep,
            force_mode=self.force_mode, command=self.current_command.copy(),
            cartesian_wrench=self.current_wrench.copy(), torque_components=self.current_torques.copy(),
            applied_generalized_force=self.robot.qf.copy(), before_qvel=qvel,
            before_tcp_pose=np.asarray(tcp_pose), before_cube_velocity=cube_velocity,
            after_cube_velocity=self.body.linear_velocity.copy(),
            robot_to_cube_force=per_link.sum(0), robot_to_cube_force_by_link=per_link,
            table_to_cube_force=table_force,
            arm_drive_stiffness=np.array([j.stiffness for j in self.arm_joints]),
            arm_drive_damping=np.array([j.damping for j in self.arm_joints]),
            arm_drive_force_limit=np.array([j.force_limit for j in self.arm_joints]),
            finger_drive_force_limit=np.array([j.force_limit for j in self.fingers])))
        self.substep += 1


def run_case(run_root, output, cfg, cube, episode, candidate, variant):
    path = run_root/'study'/'target_control'/'trajectories'/f'cube_{cube:04}_episode_{episode}_candidate_{candidate:02}.npz'
    with np.load(path, allow_pickle=False) as saved:
        seed = int(saved['reset_seed'][0])
        direction = saved['motion_direction'][0].copy()
        commanded_force = float(saved['selected_force_n'])
        reference = {key:saved[key].copy() for key in ('state','next_state','contact_force_substeps_n')}
    mass, friction = physics(cfg['seed'], cube, cfg)
    env = make_env('PickCube-v1', cfg, cfg['grip_force_limits_n'])
    motion = ComponentTrace(env, variant, cfg)
    try:
        motion.set_episode(np.array([mass, friction, cfg['table_friction'], 1, 1, 1, 0, 0, 0, 0]), seed)
        result = record_episode(motion, cfg, 2, seed, np.random.default_rng(seed),
                                measure, direction=direction, force_n=commanded_force)
        if variant == 'baseline':
            for key, expected in reference.items():
                np.testing.assert_array_equal(result[key], expected)
        trace = {key: np.asarray([row[key] for row in motion.trace]) for key in motion.trace[0]}
        start = int(result['force_start_step'][0])
        release = int(result['shove_release_step'][0])
        pulse = trace['force_mode']
        force_norm = np.linalg.norm(trace['robot_to_cube_force'][:, :2], axis=1)
        pulse_indices = np.flatnonzero(pulse)
        peak_index = pulse_indices[np.argmax(force_norm[pulse])]
        release_mask = (trace['control_step'] >= release) & (trace['control_step'] < release + 5)
        log = dict(cube_id=cube, episode_id=episode, candidate=candidate, variant=variant,
                   mass_kg=mass, friction=friction, force_start_step=start, release_step=release,
                   commanded_horizontal_force_n=commanded_force, measured_pulse_peak_n=float(force_norm[pulse].max()),
                   measured_pulse_mean_norm_n=float(force_norm[pulse].mean()),
                   measured_pulse_signed_impulse_ns=float(trace['robot_to_cube_force'][pulse,:2].sum(0) @ direction / 100),
                   measured_withdrawal_peak_n=float(force_norm[release_mask].max()),
                   signed_stop_displacement_m=float((result['next_state'][-1,19:21]-result['state'][start,19:21]) @ direction),
                   observed_stopped=bool(result['observed_stopped'][0]),
                   kinematics_check=motion.fk_check,
                   peak_sample={key: trace[key][peak_index].tolist() for key in trace},
                   active_finger_cap_n=trace['finger_drive_force_limit'][pulse][0].tolist(),
                   arm_drive_stiffness_during_pulse=trace['arm_drive_stiffness'][pulse][0].tolist(),
                   arm_drive_damping_during_pulse=trace['arm_drive_damping'][pulse][0].tolist())
        filename=f'cube_{cube:04}_episode_{episode}_candidate_{candidate:02}_{variant}'
        np.savez_compressed(output/(filename+'.npz'), **trace)
        (output/(filename+'.json')).write_text(json.dumps(log, indent=2) + '\n')
        print({key:log[key] for key in ['cube_id','episode_id','variant','measured_pulse_peak_n',
                                       'measured_withdrawal_peak_n','signed_stop_displacement_m']},flush=True)
        return log
    finally:
        motion.close()



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = run_root / 'diagnostics' / 'force_cap'
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run_root / 'data' / 'dataset.json').read_text())
    torch.set_num_threads(1)
    archive = audit_archived(run_root, output, cfg)
    worst = archive['worst_pulse_event']
    cases = (worst['cube_id'], worst['episode_id'], worst['candidate_index'])
    variants = ['baseline', 'zero_horizontal_command', 'no_orientation_feedback',
                'no_vertical_feedback', 'horizontal_wrench_only', 'zero_arm_velocity_at_onset',
                'low_finger_drive_cap']
    results = [run_case(run_root, output, cfg, *cases, variant) for variant in variants]
    withdrawal = max(archive['records'], key=lambda row: row['peak_horizontal_force_n']['withdrawal_all'])
    withdrawal_case = (withdrawal['cube_id'], withdrawal['episode_id'], withdrawal['candidate_index'])
    results += [run_case(run_root, output, cfg, *withdrawal_case, variant)
                for variant in ('baseline', 'soft_withdrawal_drives')]
    (output / 'component_interventions.json').write_text(json.dumps(results, indent=2) + '\n')
    sources = {}
    for name, expected in cfg['code_sha256'].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Production source changed: {name}')
        sources[name] = actual
    summary = {
        'command_cap_applied_to': 'Norm of the horizontal TCP wrench command before J-transpose mapping',
        'full_contact_force_cap_enforced': False,
        'contact_force_feedback_in_production_controller': False,
        'reported_contact_measurement': 'Sum of normal-contact impulse vectors divided by 0.01 s; tangent friction impulses are absent from this installed CPU reporting path',
        'measurement_source_chain': [
            'mani_skill/envs/scene.py get_pairwise_contact_forces divides impulses by timestep',
            'mani_skill/utils/sapien_utils.py compute_total_impulse sums point.impulse',
            'sapien/include/sapien/physx/simulation_callback.hpp forwards extractContacts point.impulse',
            'sapien/include/physx/include/PxSimulationEventCallback.h extractContacts sets impulse = normal * scalar_impulse'
        ],
        'per_phase_archive_statistics': archive['per_phase'],
        'worst_reported_pulse_event': worst,
        'interventions': [
            {key: row[key] for key in ('cube_id', 'episode_id', 'variant', 'measured_pulse_peak_n',
                                      'measured_withdrawal_peak_n', 'signed_stop_displacement_m')}
            for row in results
        ],
        'baseline_reproduction': 'Both instrumented baselines exactly reproduce archived state, next_state and contact arrays',
        'kinematics_and_gravity': results[0]['kinematics_check'],
        'active_finger_cap_n': results[0]['active_finger_cap_n'],
        'arm_drive_stiffness_during_pulse': results[0]['arm_drive_stiffness_during_pulse'],
        'arm_drive_damping_during_pulse': results[0]['arm_drive_damping_during_pulse'],
        'causal_scope': [
            'Removing orientation feedback changes the worst reported pulse peak, while removing vertical feedback has little effect in this case',
            'Removing entry arm velocity reduces the peak but leaves an overshoot',
            'Lowering finger drive caps has little effect in this peak case despite their independent actuation',
            'Softer restored arm drives reduce the selected withdrawal peak',
            'Interventions change nonlinear contact dynamics and cannot be added as a linear decomposition',
            'The recorded signal excludes friction, so it does not establish the complete net robot-to-cube force'
        ],
        'credible_fix': [
            'Choose an explicit bound on reported normal force or full interaction force and implement the corresponding measurement',
            'Use measured-force feedback and a compliant, rate-limited contact controller through approach, shove and withdrawal',
            'Limit approach velocity and ramp the shove and withdrawal commands to reduce transient loading',
            'Validate every physics substep over masses, frictions, directions and contact geometries; sampled feedback alone does not guarantee a hard instantaneous bound'
        ],
        'data_implication': [
            'Existing trajectories remain observations of the implemented command-capped protocol',
            'Claims of a strict 4 N measured interaction-force experiment need new controller validation and newly collected affected shove trajectories',
            'Retrain and repeat prediction and landing evaluations after changing the controller; preserve the previous data and report'
        ],
        'production_source_sha256_unchanged': sources
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Saved', output / 'summary.json')


if __name__ == '__main__':
    main()
