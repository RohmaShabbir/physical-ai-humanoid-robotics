# Data Model: Vision-Language-Action System

**Feature**: 001-vla-complete-brain
**Date**: 2025-12-12
**Status**: Complete

## Entity: VLAActionRequest

**Description**: Represents a request to the Vision-Language-Action system with voice command and visual context

**Fields**:
- `id` (string): Unique identifier for the request
- `voice_command` (string): The original voice command from user
- `processed_text` (string): Text after speech recognition processing
- `visual_context` (dict): Image data and object detection results
- `timestamp` (datetime): When the request was created
- `status` (enum): ['pending', 'processing', 'completed', 'failed']
- `target_object` (string): Object identified in the command (e.g., "red cup")
- `target_location` (string): Location mentioned in the command (e.g., "kitchen sink")

**Validation Rules**:
- `voice_command` must not be empty
- `timestamp` must be within the last 5 minutes
- `status` must be one of the allowed values

## Entity: RobotSkill

**Description**: Represents a reusable robot behavior that can be executed

**Fields**:
- `id` (string): Unique identifier for the skill
- `name` (string): Name of the skill (e.g., "grasp", "navigate", "place")
- `description` (string): Human-readable description of the skill
- `parameters` (dict): Required parameters for the skill execution
- `preconditions` (list): Conditions that must be true before execution
- `postconditions` (list): Conditions that will be true after execution
- `timeout` (int): Maximum time allowed for skill execution in seconds

**Validation Rules**:
- `name` must be unique
- `parameters` must match expected schema for the skill type
- `timeout` must be between 1 and 300 seconds

## Entity: SkillChain

**Description**: Represents a sequence of skills to be executed in order

**Fields**:
- `id` (string): Unique identifier for the skill chain
- `name` (string): Name of the skill chain
- `skills` (list): Ordered list of RobotSkill IDs
- `current_index` (int): Index of the currently executing skill
- `status` (enum): ['pending', 'executing', 'completed', 'failed', 'interrupted']
- `start_time` (datetime): When execution started
- `end_time` (datetime): When execution completed or failed
- `error_info` (dict): Details about any errors that occurred

**Validation Rules**:
- `skills` list must not be empty
- `current_index` must be between 0 and length of skills list
- `start_time` must be before `end_time` if both are set

## Entity: SafetyMonitor

**Description**: Tracks safety-related information and states

**Fields**:
- `id` (string): Unique identifier for the safety monitor
- `heartbeat_interval` (int): Expected heartbeat interval in seconds
- `last_heartbeat` (datetime): Time of last received heartbeat
- `emergency_stop_active` (bool): Whether emergency stop is currently active
- `autonomy_level` (enum): ['shadow', 'shared_control', 'full_autonomy']
- `safety_violations` (list): Log of safety violations
- `is_operational` (bool): Whether the system is currently operational

**Validation Rules**:
- `heartbeat_interval` must be between 1 and 10 seconds
- `autonomy_level` must be one of the allowed values
- If `emergency_stop_active` is true, system must not execute new commands

## Entity: ROS2Goal

**Description**: Represents a ROS 2 action goal for robot execution

**Fields**:
- `id` (string): Unique identifier for the goal
- `action_type` (string): Type of action (e.g., "GraspAction", "NavigateAction")
- `parameters` (dict): Parameters specific to the action type
- `priority` (int): Priority level (0-10, higher is more urgent)
- `creation_time` (datetime): When the goal was created
- `execution_status` (enum): ['pending', 'active', 'succeeded', 'aborted', 'canceled']
- `result` (dict): Result data returned after execution

**Validation Rules**:
- `action_type` must be a valid ROS 2 action type
- `priority` must be between 0 and 10
- `execution_status` must be one of the allowed values

## Entity: PerformanceMetrics

**Description**: Tracks system performance metrics

**Fields**:
- `id` (string): Unique identifier for the metrics record
- `timestamp` (datetime): When metrics were recorded
- `speech_to_action_latency` (float): Time from speech input to action start (seconds)
- `vla_inference_time` (float): Time for VLA model inference (seconds)
- `task_completion_time` (float): Total time for task completion (seconds)
- `success_rate` (float): Success rate for the task (0.0 to 1.0)
- `tokens_per_second` (float): Processing speed of language model (tokens/sec)
- `memory_usage` (float): VRAM usage percentage (0.0 to 1.0)

**Validation Rules**:
- All time values must be positive
- `success_rate` and `memory_usage` must be between 0.0 and 1.0
- `tokens_per_second` must be positive

## Entity Relationships

- **SkillChain** contains multiple **RobotSkill** entities (1-to-many)
- **VLAActionRequest** generates **ROS2Goal** entities (1-to-many)
- **SkillChain** executes **ROS2Goal** entities (1-to-many)
- **SafetyMonitor** monitors all active **ROS2Goal** entities (1-to-many)
- **PerformanceMetrics** records metrics for **VLAActionRequest** entities (1-to-many)