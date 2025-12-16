# API Contract: Humanoid Robot Control Interface

## Overview
This document defines the ROS 2 interface contracts for the 28-DoF humanoid robot system. These interfaces enable consistent communication between different system components and external applications.

## Joint State Interface

### Topic: `/joint_states`
- **Type**: `sensor_msgs/msg/JointState`
- **QoS**: Reliable, sensor data, 500 Hz publishing rate
- **Purpose**: Publish current joint positions, velocities, and efforts

**Message Fields**:
- `header`: Standard ROS header with timestamp
- `name`: Array of joint names (28 DOF)
- `position`: Array of joint positions in radians
- `velocity`: Array of joint velocities in rad/s
- `effort`: Array of joint efforts in Nm

## Joint Trajectory Interface

### Topic: `/joint_trajectory_controller/joint_trajectory`
- **Type**: `trajectory_msgs/msg/JointTrajectory`
- **QoS**: Reliable, command data, best-effort delivery
- **Purpose**: Send joint trajectory commands to the robot

**Message Fields**:
- `header`: Standard ROS header
- `joint_names`: Array of joint names (28 DOF)
- `points`: Array of trajectory points with positions, velocities, accelerations
- `time_from_start`: Duration from start for each point

## Teleoperation Interface

### Topic: `/joy`
- **Type**: `sensor_msgs/msg/Joy`
- **QoS**: Best-effort, real-time input
- **Purpose**: Receive gamepad input for teleoperation

**Message Fields**:
- `header`: Standard ROS header
- `axes`: Array of joystick axis values (-1.0 to 1.0)
- `buttons`: Array of button states (0 or 1)

## Robot State Interface

### Service: `/get_robot_state`
- **Type**: `std_srvs/srv/Trigger`
- **Purpose**: Request current robot state information

**Request**: Empty
**Response**:
- `success`: Boolean indicating success
- `message`: Status message

## Action Interface: Joint Trajectory Execution

### Action: `/joint_trajectory_controller/follow_joint_trajectory`
- **Type**: `control_msgs/action/FollowJointTrajectory`
- **Purpose**: Execute complex joint trajectories with feedback

**Goal**:
- `trajectory`: JointTrajectory message
- `path_tolerance`: Array of tolerances for path following
- `goal_tolerance`: Array of tolerances for goal achievement

**Result**:
- `error_code`: Integer error code
- `error_string`: Human-readable error description

## Parameter Interface

### Parameters for `humanoid_controller`:
- `control_frequency` (int, default: 500): Control loop frequency in Hz
- `real_time_factor` (double, default: 1.0): Real-time simulation factor
- `safety_limits_enabled` (bool, default: true): Enable safety limit checking
- `max_joint_velocity` (double, default: 5.0): Maximum joint velocity limit

## Quality of Service (QoS) Profiles

### Sensor Data (Joint States)
- Reliability: Reliable
- Durability: Volatile
- History: Keep Last (10)
- Rate: 500 Hz

### Command Data (Joint Trajectories)
- Reliability: Reliable
- Durability: Volatile
- History: Keep Last (1)
- Rate: As needed (up to 100 Hz)

### Real-time Input (Joy)
- Reliability: Best Effort
- Durability: Volatile
- History: Keep Last (1)
- Rate: 50 Hz minimum