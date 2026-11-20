"""Property-blind direct-force program; only the applied vector is an action."""
import numpy as np
PHYSICS_DT=.001
MODEL_DT=.005
COAST_S=1.5
SETTLE_S=.1
TRIM_AFTER_STOP_STEPS=5
HALF_SIDE=.02
ACTION_COLUMNS=['applied_force_x_N','applied_force_y_N','applied_force_z_N']

def action(plan,phase):
    if phase=='coast':return np.zeros(3,dtype=np.float32)
    angle=float(plan['direction_rad'])
    return (float(plan['force_n'])*np.array([np.cos(angle),np.sin(angle),0.])).astype(np.float32)
