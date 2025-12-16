---
id: 1
title: ros2-nervous-system-plan
stage: plan
date: 2025-12-12
surface: agent
model: claude-sonnet-4-5-20250929
feature: 001-ros2-humanoid-system
branch: 001-ros2-humanoid-system
user: user
command: /sp.plan Module 1 – The Robotic Nervous System (ROS 2)
labels: ["ros2", "humanoid", "planning", "architecture"]
links:
  spec: specs/001-ros2-humanoid-system/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-ros2-humanoid-system/plan.md
  - history/prompts/001-ros2-humanoid-system/1-ros2-nervous-system-plan.plan.prompt.md
tests:
  - null
---

## Prompt

/sp.plan Module 1 – The Robotic Nervous System (ROS 2)
Weeks 3–5 · Standalone Technical Plan
Create: architecture sketch, section structure, research approach, quality validation.
Architecture Sketch
Standalone ROS 2 workspace that becomes the permanent "nervous system" for all later modules:
text/physical_ai_ws/
├── src/
│   ├── humanoid_description/     ← URDF + Xacro + meshes
│   ├── humanoid_control/         ← ros2_control + hardware_interface
│   ├── humanoid_bringup/         ← launch files, params, RViz config
│   ├── humanoid_teleop/          ← gamepad → JointTrajectory action
│   └── python_ai_bridge/         ← rclpy template used by Modules 3–4
Section Structure (exact 5 chapters + project)
1.1 Why ROS 2 Won (2025 Perspective)
1.2 Core Concepts You Cannot Skip
1.3 Modeling a 28-DoF Humanoid from Scratch
1.4 Python-First rclpy Mastery for AI Engineers
1.5 Launch, Debug & Observe Like a Pro
→ Project M1 "Hello Humanoid" (fully working RViz + teleop demo)
Research Approach
Research-concurrent:

While writing 1.1 → web_search "ROS 2 adoption statistics 2025 industry" + X posts from @rosorg
While writing 1.3 → browse_page latest Unitree H1/G1 open-source URDF on GitHub (Dec 2025)
While writing 1.4 → code_execution test every rclpy pattern on ROS 2 Humble in Docker

Quality Validation

All code runs with colcon build --symlink-install && ros2 launch humanoid_bringup demo.launch.py
RViz shows full 28-DoF humanoid with zero warnings
500 Hz joint state publishing proven with ros2 topic hz
Gamepad teleop achieves <10 ms end-to-end latency

Decisions needing documentation

ROS 2 Distro → Humble (LTS) vs Jazzy → Chose Humble (100 % Isaac Sim 2024.2 compatibility)
Python vs C++ nodes → Pure Python (AI engineers hate CMake)
Real-time kernel → Optional but documented (PREEMPT_RT gives 1 kHz guaranteed)

Testing strategy

CI in GitHub Actions: Ubuntu 22.04 + ROS Humble + colcon test
Success = robot spawns, joints move smoothly with gamepad, no dropped messages at 500 Hz

## Response snapshot

# Implementation Plan: Module 1 – The Robotic Nervous System (ROS 2)

**Feature**: `001-ros2-humanoid-system` | **Date**: 2025-12-12 | **Spec**: [specs/001-ros2-humanoid-system/spec.md](specs/001-ros2-humanoid-system/spec.md)

## Summary

This implementation plan outlines the creation of a standalone ROS 2 workspace that serves as the permanent "nervous system" for all subsequent modules. The plan covers a comprehensive 5-chapter educational module focusing on ROS 2 fundamentals for humanoid robotics, including proper URDF modeling for a 28-DoF humanoid, ros2_control configuration, teleoperation capabilities, and Python-first rclpy patterns optimized for AI integration.

## 1.1 Why ROS 2 Won (2025 Perspective)

### Objectives
- Explain the three technical factors that led to ROS 2 adoption over ROS 1
- Document the advantages of DDS, managed nodes, lifecycle management, and security
- Compare ROS 2 distributions (Humble vs Iron vs Jazzy) for humanoid applications
- Map the current ROS 2 ecosystem for humanoid robotics

### Implementation Tasks
- Research and document the technical limitations of ROS 1 (single-master, no real-time, platform limitations)
- Analyze DDS (Data Distribution Service) benefits for distributed humanoid systems
- Document managed nodes and lifecycle state patterns for safe robot startup/shutdown
- Create comparison chart of ROS 2 distributions focusing on long-term support and humanoid compatibility
- Document key ecosystem components: Nav2, MoveIt 2, ros2_control, Isaac ROS, Foxglove, Webots

### Research Approach
- Web search: "ROS 2 adoption statistics 2025 industry" and posts from @rosorg
- Analyze official ROS 2 annual report 2025
- Document industry survey results from major robotics conferences
- Compare feature sets of different ROS 2 distributions

## 1.2 Core Concepts You Cannot Skip

### Objectives
- Explain fundamental ROS 2 concepts: Nodes, Topics, Services, Actions, Parameters
- Document critical QoS settings for humanoid applications
- Cover managed nodes and lifecycle states for safe operations
- Provide real-time Linux setup guidance

### Implementation Tasks
- Document Node implementation patterns in Python and C++
- Explain Topic communication with emphasis on QoS profiles
- Detail Service and Action patterns for synchronous and asynchronous operations
- Create QoS configuration guide with specific focus on:
  - Reliable vs Best-Effort for sensor data vs commands
  - Transient Local for safety-critical data
- Document lifecycle state management for safe robot startup
- Create real-time Linux setup checklist (PREEMPT_RT, isolcpus, 1kHz control loop)

### Research Approach
- Reference official ROS 2 documentation on QoS settings
- Analyze best practices for real-time configuration in humanoid systems
- Document practical examples of lifecycle state management

## 1.3 Modeling a 28-DoF Humanoid from Scratch

### Objectives
- Create complete URDF/Xacro model of a modern humanoid
- Configure ros2_control with proper hardware interfaces
- Implement safety limits and transmission configurations
- Set up SRDF for MoveIt 2 integration

### Implementation Tasks
- Design URDF structure for 28-DoF humanoid (based on Unitree H1/G1 reference)
- Implement Xacro macros for parameterized robot description
- Configure inertial parameters based on realistic values
- Set up ros2_control with JointStateInterface and PositionJointInterface
- Implement transmission tags and safety limits
- Create mimic joints for hand configurations
- Document SRDF configuration for MoveIt 2 compatibility
- Include sensor placements (IMU, force/torque sensors)

### Research Approach
- Browse latest Unitree H1/G1 open-source URDF on GitHub (Dec 2025)
- Research realistic inertial parameters for humanoid links
- Analyze best practices for ros2_control configuration
- Document proper mass and inertia tensor calculations

## 1.4 Python-First rclpy Mastery for AI Engineers

### Objectives
- Create Python package structure for ROS 2 nodes
- Document advanced rclpy patterns for AI integration
- Implement non-blocking callbacks for AI inference
- Optimize intra-process communication

### Implementation Tasks
- Create template Python ROS 2 package structure
- Document composition vs inheritance patterns
- Explain executor types (SingleThreaded vs MultiThreaded)
- Implement async services and actions with non-blocking callbacks
- Configure intra-process communication for zero-copy performance
- Document memory management to avoid garbage collection pauses
- Create rate vs timer vs wall timer usage guide
- Implement Python launch file patterns

### Research Approach
- Code execution testing of rclpy patterns on ROS 2 Humble in Docker
- Analyze best practices for AI/ML node integration
- Document memory management strategies for AI applications
- Test different executor patterns for AI workloads

## 1.5 Launch, Debug & Observe Like a Pro

### Objectives
- Create Python launch file templates
- Implement parameter management strategies
- Document debugging and monitoring techniques
- Set up visualization tools

### Implementation Tasks
- Create Python launch file templates with LaunchConfiguration
- Implement OpaqueFunction and GroupAction patterns
- Document remapping strategies and parameter management
- Create environment variable configuration patterns
- Set up RViz2 configuration for humanoid visualization
- Implement ros2 topic/param/service monitoring tools
- Document debugging workflows and best practices
- Create logging and introspection utilities

### Research Approach
- Test launch file patterns with complex robot configurations
- Document best practices for parameter management
- Analyze debugging tools and workflows for humanoid systems

## Architecture Sketch

```
physical_ai_ws/
├── src/
│   ├── humanoid_description/     ← URDF + Xacro + meshes
│   │   ├── urdf/
│   │   ├── meshes/
│   │   ├── launch/
│   │   └── config/
│   ├── humanoid_control/         ← ros2_control + hardware_interface
│   │   ├── controllers/
│   │   ├── hardware_interface/
│   │   ├── launch/
│   │   └── config/
│   ├── humanoid_bringup/         ← launch files, params, RViz config
│   │   ├── launch/
│   │   ├── config/
│   │   ├── params/
│   │   └── rviz/
│   ├── humanoid_teleop/          ← gamepad → JointTrajectory action
│   │   ├── src/
│   │   ├── launch/
│   │   └── config/
│   └── python_ai_bridge/         ← rclpy template used by Modules 3–4
│       ├── src/
│       ├── test/
│       └── launch/
```

## Quality Validation Criteria

### Performance Requirements
- **500 Hz joint state publishing**: Verified with `ros2 topic hz /joint_states`
- **<10 ms end-to-end gamepad latency**: Measured from input to robot response
- **Zero warnings in RViz**: Complete 28-DoF humanoid model loads without errors
- **Sub-millisecond control loop timing**: Achieved with real-time kernel configuration

### Functional Validation
- Complete `colcon build --symlink-install` without errors
- Successful execution of `ros2 launch humanoid_bringup demo.launch.py`
- Joint trajectory execution with proper feedback
- Gamepad teleoperation with smooth control
- Proper parameter management and configuration

### Code Quality Standards
- All code examples tested and verified for accuracy
- Proper QoS configurations for all communication patterns
- Memory management optimized to avoid GC pauses during AI inference
- Proper error handling and safety limits implemented

## Architectural Decisions Documented

### Decision: ROS 2 Distribution → Humble (LTS) vs Jazzy
- **Rationale**: 100% Isaac Sim 2024.2 compatibility with 5-year long-term support
- **Impact**: Ensures stability and compatibility for the duration of the project

### Decision: Python vs C++ nodes → Pure Python
- **Rationale**: AI engineers prefer Python; rclpy provides full ROS 2 capabilities
- **Impact**: Faster development for AI integration while maintaining performance

### Decision: Real-time kernel → Optional but documented
- **Rationale**: PREEMPT_RT provides 1kHz guaranteed control loops for safety
- **Impact**: Enables deterministic timing critical for humanoid balance and safety

## Testing Strategy

### CI Pipeline
- GitHub Actions with Ubuntu 22.04 + ROS Humble + colcon test
- Unit tests for individual nodes using pytest and rostest
- Integration tests with Gazebo simulation
- Performance validation for timing requirements

### Success Criteria
- Robot spawns correctly with complete URDF model
- Joints move smoothly with gamepad teleoperation
- No dropped messages at 500 Hz joint state publishing
- Sub-10ms latency for gamepad-to-robot command execution
- All safety limits and constraints properly enforced

### Validation Tests
- Joint state publishing frequency verification
- Teleoperation latency measurement
- RViz visualization without warnings
- Parameter loading and configuration validation
- Controller performance under load

## Outcome

- ✅ Impact: Created comprehensive implementation plan for Module 1 - The Robotic Nervous System (ROS 2) with 5 detailed chapters covering all aspects of ROS 2 for humanoid robotics
- 🧪 Tests: null
- 📁 Files: Created plan documentation with architecture sketch, research approach, quality validation criteria, architectural decisions, and testing strategy
- 🔁 Next prompts: Implementation of the individual components as per the plan
- 🧠 Reflection: The plan provides a complete roadmap for developing the ROS 2 humanoid system with clear objectives, implementation tasks, and validation criteria for each chapter

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin implementation of the humanoid_description package as outlined in Chapter 1.3