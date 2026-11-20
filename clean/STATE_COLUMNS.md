# State and action columns

The state is the explicit physical measurement whitelist in `clean/schema.py:observe`. It contains 48 coordinates. No mass, friction, inertia, reward, success flag, cube identity, or episode identity is an input coordinate. True per-cube properties are stored only in `clean/data/cubes.json` and installed only by `clean/native.py:set_cube_properties`.

Physical sensor readings can depend on the hidden properties through the simulator dynamics. This is permitted physical evidence. No state coordinate is analytically computed from a physical property label.

| Zero-based index | State column | Unit | Physical source within observe() |
|---|---|---|---|
| 0 | `joint_position_0` | rad (0–6), m (7–8) | robot.qpos |
| 1 | `joint_position_1` | rad (0–6), m (7–8) | robot.qpos |
| 2 | `joint_position_2` | rad (0–6), m (7–8) | robot.qpos |
| 3 | `joint_position_3` | rad (0–6), m (7–8) | robot.qpos |
| 4 | `joint_position_4` | rad (0–6), m (7–8) | robot.qpos |
| 5 | `joint_position_5` | rad (0–6), m (7–8) | robot.qpos |
| 6 | `joint_position_6` | rad (0–6), m (7–8) | robot.qpos |
| 7 | `joint_position_7` | rad (0–6), m (7–8) | robot.qpos |
| 8 | `joint_position_8` | rad (0–6), m (7–8) | robot.qpos |
| 9 | `joint_velocity_0` | rad/s (0–6), m/s (7–8) | robot.qvel |
| 10 | `joint_velocity_1` | rad/s (0–6), m/s (7–8) | robot.qvel |
| 11 | `joint_velocity_2` | rad/s (0–6), m/s (7–8) | robot.qvel |
| 12 | `joint_velocity_3` | rad/s (0–6), m/s (7–8) | robot.qvel |
| 13 | `joint_velocity_4` | rad/s (0–6), m/s (7–8) | robot.qvel |
| 14 | `joint_velocity_5` | rad/s (0–6), m/s (7–8) | robot.qvel |
| 15 | `joint_velocity_6` | rad/s (0–6), m/s (7–8) | robot.qvel |
| 16 | `joint_velocity_7` | rad/s (0–6), m/s (7–8) | robot.qvel |
| 17 | `joint_velocity_8` | rad/s (0–6), m/s (7–8) | robot.qvel |
| 18 | `joint_effort_0` | N m (0–6), N (7–8) | incoming child-link joint wrench: torque X for revolute, force X for prismatic |
| 19 | `joint_effort_1` | N m (0–6), N (7–8) | incoming child-link joint wrench: torque X for revolute, force X for prismatic |
| 20 | `joint_effort_2` | N m (0–6), N (7–8) | incoming child-link joint wrench: torque X for revolute, force X for prismatic |
| 21 | `joint_effort_3` | N m (0–6), N (7–8) | incoming child-link joint wrench: torque X for revolute, force X for prismatic |
| 22 | `joint_effort_4` | N m (0–6), N (7–8) | incoming child-link joint wrench: torque X for revolute, force X for prismatic |
| 23 | `joint_effort_5` | N m (0–6), N (7–8) | incoming child-link joint wrench: torque X for revolute, force X for prismatic |
| 24 | `joint_effort_6` | N m (0–6), N (7–8) | incoming child-link joint wrench: torque X for revolute, force X for prismatic |
| 25 | `joint_effort_7` | N m (0–6), N (7–8) | incoming child-link joint wrench: torque X for revolute, force X for prismatic |
| 26 | `joint_effort_8` | N m (0–6), N (7–8) | incoming child-link joint wrench: torque X for revolute, force X for prismatic |
| 27 | `gripper_opening_0` | m | sum of measured finger joint positions |
| 28 | `finger_position_error_0` | m | issued finger drive target minus measured finger position |
| 29 | `finger_position_error_1` | m | issued finger drive target minus measured finger position |
| 30 | `cube_position_0` | m | cube entity world pose position |
| 31 | `cube_position_1` | m | cube entity world pose position |
| 32 | `cube_position_2` | m | cube entity world pose position |
| 33 | `cube_rotation_0` | unitless | first two columns of cube pose rotation matrix |
| 34 | `cube_rotation_1` | unitless | first two columns of cube pose rotation matrix |
| 35 | `cube_rotation_2` | unitless | first two columns of cube pose rotation matrix |
| 36 | `cube_rotation_3` | unitless | first two columns of cube pose rotation matrix |
| 37 | `cube_rotation_4` | unitless | first two columns of cube pose rotation matrix |
| 38 | `cube_rotation_5` | unitless | first two columns of cube pose rotation matrix |
| 39 | `cube_linear_velocity_0` | m/s | cube rigid-body linear velocity |
| 40 | `cube_linear_velocity_1` | m/s | cube rigid-body linear velocity |
| 41 | `cube_linear_velocity_2` | m/s | cube rigid-body linear velocity |
| 42 | `cube_angular_velocity_0` | rad/s | cube rigid-body angular velocity |
| 43 | `cube_angular_velocity_1` | rad/s | cube rigid-body angular velocity |
| 44 | `cube_angular_velocity_2` | rad/s | cube rigid-body angular velocity |
| 45 | `cube_relative_position_0` | m | TCP rotation transpose times (cube position minus TCP position) |
| 46 | `cube_relative_position_1` | m | TCP rotation transpose times (cube position minus TCP position) |
| 47 | `cube_relative_position_2` | m | TCP rotation transpose times (cube position minus TCP position) |

The six world-orientation coordinates are the first two columns of the measured cube rotation matrix, stored column by column. Cube position relative to the gripper is measured in the TCP frame. Relative cube orientation is absent from this fixed 48-coordinate state.

Initial joint effort is zero before the first simulation step. Subsequent effort uses measured incoming child-link reaction wrench: the local X torque component for revolute joints and local X force component for the prismatic finger joints. Finger position error is the issued drive target minus measured position; the target is an exogenous controller command.

## Actions

All nine commands come from `clean/script.py:plan_episode(cube_pose, settings)`. The planner receives only the measured initial cube pose and independently sampled episode settings. It receives no simulator environment, physical labels, realized tracking feedback, success flag, or reward. Planning uses fixed nominal robot joints and planned endpoints. The complete sequence is fixed before executing the episode.

| Zero-based index | Action column | Code path / meaning |
|---|---|---|
| 0 | `arm_joint_target_0_rad` | Planned arm joint position target from fixed robot kinematics |
| 1 | `arm_joint_target_1_rad` | Planned arm joint position target from fixed robot kinematics |
| 2 | `arm_joint_target_2_rad` | Planned arm joint position target from fixed robot kinematics |
| 3 | `arm_joint_target_3_rad` | Planned arm joint position target from fixed robot kinematics |
| 4 | `arm_joint_target_4_rad` | Planned arm joint position target from fixed robot kinematics |
| 5 | `arm_joint_target_5_rad` | Planned arm joint position target from fixed robot kinematics |
| 6 | `arm_joint_target_6_rad` | Planned arm joint position target from fixed robot kinematics |
| 7 | `gripper_command_normalized` | Independently scheduled open/close command |
| 8 | `grip_force_cap_per_finger_N` | Randomly sampled actuator force cap per finger; issued command rather than a measured property |

The same randomized episode setting and action plan is reused across cubes at each episode index. Approach speed, lift speed, lift height, grip cap, and grasp offset vary across episodes independently of object properties. Action identity across cubes and episode-data hashes are checked by `clean/audit.py`; per-column correlations are computed using training cubes only.

Every collection receipt records the SHA-256 of the data file and its collection sources. The audit requires the current `schema.py`, `script.py`, and `native.py` hashes to match those receipts before certifying provenance.
