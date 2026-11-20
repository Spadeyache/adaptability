"""Open-loop reach/grasp/lift adapted from installed ManiSkill PickCube solver.

Only episode inputs: initial observed cube pose and independently sampled settings.
All plans start from fixed nominal joint positions and concatenate planned endpoints.
No simulator/env handle, physical labels, or executed robot feedback is available here.
"""
import numpy as np
import sapien
import mplib
from mani_skill import PACKAGE_ASSET_DIR

INITIAL_QPOS=np.array([0,np.pi/8,0,-5*np.pi/8,0,3*np.pi/4,np.pi/4,.04,.04])
DT=.05

def sample_settings(episode):
    # Common randomized settings across cubes: balanced inputs, matched physics comparisons.
    rng=np.random.default_rng([2026111507,episode,1])
    return dict(episode_id=int(episode),approach_speed_scale=float(rng.uniform(.4,.8)),
        lift_speed_scale=float(rng.uniform(.15,.5)),lift_height_m=float(rng.uniform(.07,.15)),
        grip_force_N=float(np.exp(rng.uniform(np.log(.5),np.log(30)))),
        grasp_offset_m=rng.uniform(-.002,.002,3).tolist(),scene_seed=int(820000+episode))

def plan_episode(cube_pose,settings):
    """Cube pose is seven observed numbers; settings contain no object physics."""
    if set(settings)!={'episode_id','approach_speed_scale','lift_speed_scale','lift_height_m','grip_force_N','grasp_offset_m','scene_seed'}:
        raise ValueError('Unexpected script input')
    urdf=str(PACKAGE_ASSET_DIR/'robots/panda/panda_v2.urdf')
    planner=mplib.Planner(urdf=urdf,srdf=urdf.replace('.urdf','.srdf'),move_group='panda_hand_tcp')
    planner.set_base_pose(np.array([-.615,0,0,1,0,0,0]))
    velocity=np.asarray(planner.joint_vel_limits).copy();acceleration=np.asarray(planner.joint_acc_limits).copy()
    pose=sapien.Pose(cube_pose[:3],cube_pose[3:])
    approaching=np.array([0.,0.,-1.])
    # Same symmetry choice as installed compute_grasp_info_by_obb: choose the
    # cube horizontal axis nearest the nominal gripper closing direction.
    # Nominal FK depends only on fixed robot geometry/joints, never cube physics.
    planner.pinocchio_model.compute_forward_kinematics(INITIAL_QPOS)
    nominal=planner.pinocchio_model.get_link_pose(planner.move_group_link_id)
    target_closing=sapien.Pose(nominal[:3],nominal[3:]).to_transformation_matrix()[:3,1]
    axes=pose.to_transformation_matrix()[:3,:2]
    closing=axes[:,np.argmax(np.abs(target_closing@axes))].copy()
    if target_closing@closing<0:closing=-closing
    closing[2]=0;closing/=np.linalg.norm(closing)
    matrix=np.eye(4);matrix[:3,:3]=np.stack([np.cross(closing,approaching),closing,approaching],axis=1)
    matrix[:3,3]=pose.p+np.asarray(settings['grasp_offset_m'])
    grasp=sapien.Pose(matrix)
    reach=grasp*sapien.Pose([0,0,-.05])
    goal=sapien.Pose(grasp.p+np.array([0,0,settings['lift_height_m']]),grasp.q)
    q=INITIAL_QPOS.copy();commands=[];phases=[]
    def move(target,scale,opening,phase):
        nonlocal q
        planner.joint_vel_limits=velocity*scale;planner.joint_acc_limits=acceleration*scale
        result=planner.plan_screw(np.r_[target.p,target.q],q,time_step=DT,use_point_cloud=False)
        if result['status']!='Success':raise RuntimeError('Motion plan failed: '+str(result['status']))
        for position in result['position']:
            commands.append(np.r_[position[:7],opening,settings['grip_force_N']]);phases.append(phase)
        q[:7]=result['position'][-1][:7];q[7:]=.04 if opening==1 else 0
    move(reach,settings['approach_speed_scale'],1,'approach')
    move(grasp,settings['approach_speed_scale'],1,'grasp_approach')
    for _ in range(20):commands.append(np.r_[q[:7],-1,settings['grip_force_N']]);phases.append('close')
    q[7:]=0
    move(goal,settings['lift_speed_scale'],-1,'lift')
    for _ in range(40):commands.append(np.r_[q[:7],-1,settings['grip_force_N']]);phases.append('hold')
    return np.asarray(commands,dtype=np.float32),phases
