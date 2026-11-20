import unittest
import numpy as np
from experiments.property_probe.data import observed_features,check_cube_correlations,InputCorrelationError

class GuardTests(unittest.TestCase):
    def test_negative_label_encoding_is_blocked(self):
        y=np.column_stack((np.arange(8),np.arange(8)**2))
        with self.assertRaises(InputCorrelationError):
            check_cube_correlations(-y[:,:1],y,['hidden_mass'],dataset='unit',split='train',task=0)
    def test_constant_channel_is_not_infinite_correlation(self):
        y=np.column_stack((np.arange(8),np.arange(8)**2))
        self.assertTrue(check_cube_correlations(np.ones((8,1)),y,['constant'],dataset='unit',split='train',task=0)['passed'])
    def test_drop_both_states_and_velocity_descendant(self):
        z=dict(state=np.zeros((2,43)),next_state=np.zeros((2,43)),action=np.zeros((2,8)))
        for prefix in ('','next_'):
            for k,n in [('relative_position',3),('relative_velocity',3),('finger_position_error',2)]:z[prefix+k]=np.zeros((2,n))
        x,names,dropped=observed_features(z)
        for name in ['state[42]','next_state[42]','relative_position[2]','next_relative_position[2]','relative_velocity[2]','next_relative_velocity[2]']:
            self.assertIn(name,dropped);self.assertNotIn(name,names)
        self.assertEqual(x.shape[1],len(names))

if __name__=='__main__':unittest.main()
