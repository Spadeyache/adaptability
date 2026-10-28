"""Physics-input interventions and post-action, conditional-force accounting."""
import unittest

import numpy as np
import torch

from experiments.physics_conditioning.checkpoint_probe import (
    MODES,extract_metrics,policy_observation,supplied_parameters,
)


def episode(count=12):
    return {'input__action_applied':np.zeros((count,5)),
            'sim__is_grasped':np.zeros(count+1,dtype=bool),
            'sim__object_clearance_m':np.zeros(count+1),
            'sim__task_success':np.zeros(count+1,dtype=bool),
            'sim__policy_observation':np.zeros((count+1,45)),
            'input__grip_motor_limit_n':np.zeros((count,2)),
            'sim__reward':np.arange(count,dtype=float),
            'label__held_goal_success_final':np.asarray(False),
            'meta__control_frequency_hz':np.asarray(20),
            'input__finger_compression_mean_n':np.zeros((count,2))}


class CheckpointProbeTests(unittest.TestCase):
    def test_six_modes_preserve_the_existing_gpu_intervention_definitions(self):
        expected={'true':(.25,.2),'fixed_friction':(.25,.5),
                  'wrong_friction':(.25,1.),'fixed_mass':(.04,.2),
                  'wrong_mass':(.29,.2),'fixed_both':(.04,.5)}
        self.assertEqual(set(expected),set(MODES))
        for mode,wanted in expected.items():
            with self.subTest(mode=mode):np.testing.assert_allclose(supplied_parameters(mode,.25,.2),wanted)
        with self.assertRaises(ValueError):supplied_parameters('other',.25,.2)
        with self.assertRaises(ValueError):supplied_parameters('wrong_mass',.6,.2)

    def test_every_mode_changes_only_supplied_weight_and_appended_friction(self):
        original=torch.arange(45,dtype=torch.float32)[None]
        saved=original.clone()
        keep=[i for i in range(45) if i!=42]
        for mode in MODES:
            mass,mu=supplied_parameters(mode,.25,.2)
            supplied=policy_observation(original,mass,mu,1.62)
            self.assertEqual(tuple(supplied.shape),(1,46))
            torch.testing.assert_close(supplied[:,keep],original[:,keep])
            self.assertAlmostEqual(float(supplied[0,42]),mass*1.62,places=6)
            self.assertAlmostEqual(float(supplied[0,45]),mu,places=6)
            torch.testing.assert_close(original,saved)

    def test_force_and_caps_use_correct_post_action_lift_indices_and_two_fingers(self):
        data=episode()
        data['sim__is_grasped'][[2,4]]=True
        data['sim__object_clearance_m'][[2,4]]=[.04,.06]
        data['sim__policy_observation'][:,43:45]=1000.
        data['sim__policy_observation'][2,43:45]=[2.,4.]
        data['sim__policy_observation'][4,43:45]=[6.,8.]
        data['input__grip_motor_limit_n'][1]=[10.,14.]
        data['input__grip_motor_limit_n'][3]=[20.,24.]
        data['input__finger_compression_mean_n'][[1,3]]=[[1.,3.],[5.,7.]]
        result=extract_metrics(data)
        self.assertEqual(result['lifted_steps'],2)
        self.assertEqual(result['force_step_count'],2)
        self.assertEqual(result['force_sample_count'],4)
        self.assertEqual(result['lifted_force_n'],5.)
        self.assertEqual(result['lifted_force_sum_n'],20.)
        self.assertEqual(result['lifted_motor_limit_n'],17.)
        self.assertEqual(result['lifted_interval_force_n'],4.)
        self.assertEqual(result['lifted_exposure_s'],.1)
        self.assertEqual(result['reward_sum'],66.)
        self.assertTrue(result['grasp_lost_after_lift'])

    def test_no_lift_has_missing_conditional_force_not_a_zero_efficiency_claim(self):
        data=episode()
        data['sim__is_grasped'][2]=True
        data['sim__object_clearance_m'][2]=.03 # Strictly greater than3cm, matching GPU.
        data['sim__policy_observation'][:,43:45]=500.
        result=extract_metrics(data)
        self.assertFalse(result['lifted_ever'])
        self.assertFalse(result['grasp_lost_after_lift'])
        self.assertEqual(result['force_sample_count'],0)
        self.assertIsNone(result['lifted_force_n'])
        self.assertIsNone(result['lifted_motor_limit_n'])
        self.assertIsNone(result['lifted_interval_force_n'])

    def test_grasp_loss_before_first_lift_does_not_count_as_post_lift_loss(self):
        data=episode()
        data['sim__is_grasped'][1]=True
        data['sim__is_grasped'][3:]=True
        data['sim__object_clearance_m'][3:]=.04
        result=extract_metrics(data)
        self.assertTrue(result['lifted_ever'])
        self.assertFalse(result['grasp_lost_after_lift'])

    def test_sustained_success_is_final_ten_post_action_samples_with_correct_exposure(self):
        data=episode()
        data['sim__is_grasped'][3:]=True
        data['sim__task_success'][3:]=True
        data['label__held_goal_success_final']=np.asarray(True)
        result=extract_metrics(data)
        self.assertTrue(result['final_success'])
        self.assertTrue(result['sustained_success'])
        self.assertEqual(result['sustained_post_action_steps'],10)
        self.assertEqual(result['sustained_exposure_s'],.5)
        data['sim__task_success'][7]=False
        result=extract_metrics(data)
        self.assertTrue(result['final_success'])
        self.assertFalse(result['sustained_success'])

    def test_misaligned_metrics_and_inconsistent_final_label_are_rejected(self):
        data=episode();data['input__grip_motor_limit_n']=np.zeros((11,2))
        with self.assertRaises(ValueError):extract_metrics(data)
        data=episode();data['label__held_goal_success_final']=np.asarray(True)
        with self.assertRaisesRegex(ValueError,'label disagrees'):extract_metrics(data)
        data=episode();data['sim__policy_observation'][1,43]=-1.
        with self.assertRaises(ValueError):extract_metrics(data)
        data=episode();data['meta__control_frequency_hz']=np.asarray(0.)
        with self.assertRaisesRegex(ValueError,'frequency'):extract_metrics(data)


if __name__=='__main__':unittest.main()
