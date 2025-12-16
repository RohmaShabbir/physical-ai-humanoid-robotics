---
id: 1
title: digital-twin-spec
stage: spec
date: 2025-12-12
surface: agent
model: claude-sonnet-4-5-20250929
feature: 001-digital-twin
branch: 001-digital-twin
user: user
command: /sp.specify MODULE 2 – The Digital Twin (Gazebo & Unity)
labels: ["digital-twin", "gazebo", "unity", "simulation", "specification"]
links:
  spec: specs/001-digital-twin/spec.md
  checklist: specs/001-digital-twin/checklists/requirements.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-digital-twin/spec.md
  - specs/001-digital-twin/checklists/requirements.md
  - history/prompts/001-digital-twin/1-digital-twin-spec.spec.prompt.md
tests:
  - null
---

## Prompt

MODULE 2 – The Digital Twin (Gazebo & Unity)
Weeks 6–7 · ~90 pages · 5 chapters
Chapter 2.1 – Gazebo Ignition 2025: What Actually Works

Gazebo Harmonic (2025 default) vs classic – migration checklist
Installing via NVIDIA Isaac Sim container (the only way that doesn't break)
SDF 1.11 features you need: //contact/bitmasked, //inertial/recalculate

Chapter 2.2 – From URDF to Perfect SDF

Automated conversion script (gz sdf -p) and the 7 things it still gets wrong
Adding collision bitmasks, surface friction, self-collide-without-contact for hands
Realistic actuator modeling (effort, velocity limits, damping)

Chapter 2.3 – Sensor Simulation That Transfers to Real Robots

Depth camera noise models (RealSense D435i profile)
IMU simulation with realistic bias & noise (Allan variance numbers)
Ray sensor (LIDAR) with dropout and angular noise
Adding camera lens distortion and rolling shutter

Chapter 2.4 – Unity for When Gazebo Looks Too 2015

Unity Robotics Hub 2025 workflow
Exporting USD from Isaac Sim → Unity with animations
ROS-TCP-Connector vs rosbridge vs Unity ROS2 package benchmark

Chapter 2.5 – Building "Apartment-01" – The Standard Test Environment

Procedurally generating realistic kitchens, living rooms, clutter
Adding semantic labels for VLA training
Project M2 – "Digital Apartment"
Deliverable: A single launch file that starts Gazebo Harmonic with a fully sensorized humanoid standing in a cluttered apartment containing cups, chairs, and doors.

## Response snapshot

# Feature Specification: Digital Twin (Gazebo & Unity)

**Feature Branch**: `001-digital-twin`
**Created**: 2025-12-12
**Status**: Draft
**Input**: User description: "MODULE 2 – The Digital Twin (Gazebo & Unity)
Weeks 6–7 · ~90 pages · 5 chapters
Chapter 2.1 – Gazebo Ignition 2025: What Actually Works

Gazebo Harmonic (2025 default) vs classic – migration checklist
Installing via NVIDIA Isaac Sim container (the only way that doesn't break)
SDF 1.11 features you need: //contact/bitmasked, //inertial/recalculate

Chapter 2.2 – From URDF to Perfect SDF

Automated conversion script (gz sdf -p) and the 7 things it still gets wrong
Adding collision bitmasks, surface friction, self-collide-without-contact for hands
Realistic actuator modeling (effort, velocity limits, damping)

Chapter 2.3 – Sensor Simulation That Transfers to Real Robots

Depth camera noise models (RealSense D435i profile)
IMU simulation with realistic bias & noise (Allan variance numbers)
Ray sensor (LIDAR) with dropout and angular noise
Adding camera lens distortion and rolling shutter

Chapter 2.4 – Unity for When Gazebo Looks Too 2015

Unity Robotics Hub 2025 workflow
Exporting USD from Isaac Sim → Unity with animations
ROS-TCP-Connector vs rosbridge vs Unity ROS2 package benchmark

Chapter 2.5 – Building "Apartment-01" – The Standard Test Environment

Procedurally generating realistic kitchens, living rooms, clutter
Adding semantic labels for VLA training
Project M2 – "Digital Apartment"
Deliverable: A single launch file that starts Gazebo Harmonic with a fully sensorized humanoid standing in a cluttered apartment containing cups, chairs, and doors."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Gazebo Simulation Environment Setup (Priority: P1)

As a robotics researcher, I need to set up a realistic Gazebo simulation environment that accurately represents my physical humanoid robot, so that I can test algorithms and behaviors in a safe, cost-effective environment before deploying to the real robot.

**Why this priority**: This provides the foundational simulation capability that all other testing and development depends on. Without an accurate simulation, researchers cannot validate their algorithms safely or cost-effectively.

**Independent Test**: Can be fully tested by launching the Gazebo environment with the humanoid model and verifying that joint dynamics, sensor outputs, and physical interactions match expected real-world behavior.

**Acceptance Scenarios**:

1. **Given** a need to test humanoid walking algorithms, **When** I launch the Gazebo simulation, **Then** the humanoid model behaves with realistic physics and joint constraints
2. **Given** a need to validate sensor data, **When** I run the simulation, **Then** the simulated sensors produce data that closely matches the real robot's sensors
3. **Given** a need to test collision handling, **When** the humanoid interacts with objects in simulation, **Then** the physics accurately reflect real-world contact dynamics

---

### User Story 2 - Sensor Simulation with Realistic Noise Models (Priority: P2)

As a perception engineer, I need to simulate realistic sensor data including noise, bias, and environmental effects, so that I can develop robust perception algorithms that will work reliably on the physical robot.

**Why this priority**: Realistic sensor simulation is critical for developing perception algorithms that transfer from simulation to reality. Without proper noise modeling, algorithms may work in simulation but fail on the real robot.

**Independent Test**: Can be tested by comparing simulated sensor outputs with real robot sensor data to verify that noise characteristics and error patterns match.

**Acceptance Scenarios**:

1. **Given** a depth camera simulation, **When** capturing data in the simulated environment, **Then** the depth data includes realistic noise patterns similar to RealSense D435i
2. **Given** an IMU simulation, **When** the humanoid moves in simulation, **Then** the IMU data includes appropriate bias and noise based on Allan variance models
3. **Given** a LIDAR simulation, **When** scanning the environment, **Then** the data includes dropout and angular noise that matches real sensor characteristics

---

### User Story 3 - Digital Apartment Environment Creation (Priority: P3)

As a robotics developer, I need to create a standardized test environment with realistic household objects and layouts, so that I can evaluate robot capabilities in scenarios that reflect real-world deployment conditions.

**Why this priority**: Having a standardized test environment is essential for consistent evaluation of robot capabilities and comparing different algorithms. The apartment environment provides a realistic setting for testing navigation, manipulation, and interaction tasks.

**Independent Test**: Can be tested by launching the apartment environment and verifying that it contains the specified objects (cups, chairs, doors) with proper physical and visual properties.

**Acceptance Scenarios**:

1. **Given** a need to test navigation algorithms, **When** launching the apartment environment, **Then** the space contains realistic rooms (kitchen, living room) with navigable pathways
2. **Given** a need to test object manipulation, **When** the environment loads, **Then** it includes various household objects like cups and chairs positioned for interaction tasks
3. **Given** a need for VLA (Vision-Language-Action) training, **When** using the environment, **Then** objects have semantic labels for training perception models

---

### User Story 4 - Unity Simulation Alternative (Priority: P4)

As a developer who needs photorealistic simulation, I need access to Unity-based simulation that can connect to ROS, so that I can generate synthetic data for training perception models with realistic visual rendering.

**Why this priority**: Unity provides superior visual rendering capabilities compared to Gazebo, which is valuable for generating synthetic training data and testing vision-based algorithms under realistic lighting conditions.

**Independent Test**: Can be tested by exporting a scene from Isaac Sim to Unity and verifying that the robot model and environment render correctly with proper physics and ROS connectivity.

**Acceptance Scenarios**:

1. **Given** a need for photorealistic rendering, **When** using Unity simulation, **Then** the visual output matches real-world appearance with accurate lighting and materials
2. **Given** a need to connect to ROS, **When** running Unity simulation, **Then** the Unity ROS connector properly communicates with ROS nodes
3. **Given** a need to export from Isaac Sim, **When** converting USD files, **Then** the robot and environment maintain their properties and animations in Unity

---

### Edge Cases

- What happens when simulation physics diverge significantly from real robot behavior due to modeling inaccuracies?
- How does the system handle sensor simulation failures or extreme environmental conditions?
- What occurs when the humanoid robot experiences joint limits or actuator saturation in simulation?
- How does the system handle large numbers of objects in the apartment environment that could impact simulation performance?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support Gazebo Harmonic 2025 with proper installation via NVIDIA Isaac Sim container
- **FR-002**: System MUST convert URDF models to SDF format with proper joint limits, inertial properties, and collision geometries
- **FR-003**: System MUST simulate realistic sensor outputs including depth cameras, IMU, and LIDAR with appropriate noise models
- **FR-004**: System MUST provide accurate actuator modeling with effort limits, velocity limits, and damping parameters
- **FR-005**: System MUST include collision bitmasks and proper contact handling for complex interactions like hand manipulation
- **FR-006**: System MUST generate procedurally realistic apartment environments with kitchens, living rooms, and household objects
- **FR-007**: System MUST provide semantic labeling for objects to support VLA training
- **FR-008**: System MUST support Unity integration via ROS connectors for photorealistic rendering
- **FR-009**: System MUST include a single launch file that starts the complete simulation environment with humanoid robot
- **FR-010**: Users MUST be able to access both Gazebo and Unity simulation environments for the same robot model
- **FR-011**: System MUST simulate realistic contact dynamics including friction, surface properties, and self-collision handling
- **FR-012**: System MUST provide camera simulation with lens distortion and rolling shutter effects

### Key Entities

- **Digital Twin Environment**: Represents the virtual simulation space that mirrors the physical robot's environment, including physics properties, lighting, and objects
- **Sensor Simulation Layer**: Represents the abstraction that provides realistic sensor data including noise models, biases, and environmental effects
- **Robot Model Converter**: Represents the process that transforms URDF robot descriptions into simulation-ready SDF formats with proper physical properties
- **Apartment Environment Generator**: Represents the procedural system that creates realistic household environments with appropriate object placement and semantic labeling

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can launch the complete simulation environment with a single command and have the humanoid robot properly positioned in the apartment setting within 30 seconds
- **SC-002**: Simulated sensor data matches real robot sensor characteristics with noise profiles within 10% of measured real-world values
- **SC-003**: Physics simulation maintains real-time performance (20+ Hz) with 28-DoF humanoid and complex apartment environment containing 50+ objects
- **SC-004**: Users can successfully transfer algorithms developed in simulation to the physical robot with at least 80% of performance metrics preserved
- **SC-005**: The system supports both Gazebo and Unity workflows with equivalent robot model fidelity and sensor simulation quality
- **SC-006**: Apartment environment contains at least 5 different room types with realistic furniture and objects for comprehensive testing scenarios

## Outcome

- ✅ Impact: Created comprehensive feature specification for Digital Twin (Gazebo & Unity) module with 4 prioritized user stories, functional requirements, and measurable success criteria
- 🧪 Tests: null
- 📁 Files: Created spec.md and requirements checklist in checklists/requirements.md
- 🔁 Next prompts: Ready for planning phase with /sp.plan
- 🧠 Reflection: The specification captures all essential elements of the digital twin system including Gazebo simulation, sensor modeling, Unity integration, and the apartment test environment

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Proceed with planning phase to implement the digital twin system