"""Property-blind randomized commands and initial contact geometry."""
import numpy as np

PHYSICS_DT=.001
MODEL_DT=.005
APPROACH_S=.8
PRELOAD_S=.1
PUSH_S=.08
COAST_S=1.
PUSHER_MASS=.005
PUSHER_RADIUS=.008
HALF_SIDE=.02


def geometry(plan):
    angle=float(plan['direction_rad']); yaw=float(plan['yaw_rad'])
    direction=np.array([np.cos(angle),np.sin(angle),0.])
    rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    local_direction=rotation.T@direction[:2]
    axis=int(np.argmax(np.abs(local_direction)))
    normal_local=np.zeros(2);normal_local[axis]=-np.sign(local_direction[axis])
    point_local=HALF_SIDE*normal_local
    point_local[1-axis]=float(plan['contact_offset_m'])
    center=np.array([*plan['start_xy'],HALF_SIDE])
    point=center+np.r_[rotation@point_local,0.]
    normal=np.r_[rotation@normal_local,0.]
    start=point+PUSHER_RADIUS*normal-.003*direction
    # The release target depends on initial commands only, never observed motion.
    park=center-1.5*direction
    return dict(direction=direction,cube_center=center,contact_point=point,
                pusher_start=start,park=park)


def action(plan,phase):
    g=geometry(plan)
    force=.95*float(plan['cap_n'])*g['direction'] if phase=='push' else np.zeros(3)
    return np.r_[force,plan['cap_n'],float(phase=='push'),g['park']].astype(np.float32)


ACTION_COLUMNS=['commanded_force_x_N','commanded_force_y_N','commanded_force_z_N',
                'force_cap_N','pusher_engaged','release_target_x_m','release_target_y_m','release_target_z_m']
