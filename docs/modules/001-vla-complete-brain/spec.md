# Feature Specification: Vision-Language-Action - The Complete Brain

**Feature Branch**: `001-vla-complete-brain`
**Created**: 2025-12-12
**Status**: Draft
**Input**: User description: "MODULE 4 – Vision-Language-Action – The Complete Brain
Weeks 11–13 · ~100 pages · 5 chapters + Capstone
Chapter 4.1 – VLA Models You Can Actually Run on Jetson in 2025

Open-VLA-7B (Prismatic), RT-2-X, Octo-Small, Gemini-Robotics-1.5
Quantized GGUF versions that fit in 16 GB VRAM
Performance table: tokens/sec vs success rate on common tasks

Chapter 4.2 – Voice → Action End-to-End Pipeline

Faster-Whisper (large-v3) running locally at 8× real-time on Orin
LLM prompt template that turns "Put the apple in the microwave" into structured ROS 2 goals
Skill library + chaining (grasp → navigate → place)

Chapter 4.3 – Making LLMs Reliable for Robotics

Chain-of-thought prompting, self-correction, tool calling (ReAct style)
When to fall back to classical planning or imitation

Chapter 4.4 – Running 7B Models on Jetson with TensorRT-LLM

Converting .safetensors → TensorRT engine step-by-step
Streaming tokens while robot is moving (no blocking)

Chapter 4.5 – Safety, Watchdogs & Real-Robot Deployment

Heartbeat monitor, emergency stop action server
Gradual autonomy levels (shadow mode → shared control → full autonomy)

Capstone Project – "Hey Robot, Tidy the Apartment"
Exact task (same as real 2025 robotics competitions):

Robot starts on charging dock
User says: "The red cup is on the table. Bring it to the kitchen sink."
Robot must perceive, plan, walk, grasp (or scoop), navigate back, place cup in sink
Success if completed in ≤3 minutes in simulation and ≤5 minutes on real hardware (Unitree Go2/G1 or proxy).

Deliverable repo contains:

Single ros2 launch file launch_capstone.py that works both in Isaac Sim and on real Jetson + robot
3 demo videos: simulation, Jetson-only dry run, real robot run"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Voice Command Processing and Action Execution (Priority: P1)

A robotics researcher or user gives a voice command to the robot (e.g., "The red cup is on the table. Bring it to the kitchen sink."), and the robot processes the command using speech recognition, vision-language-action models, and executes the required tasks (perceive, plan, walk, grasp, navigate, place).

**Why this priority**: This represents the core functionality of the VLA system and enables the complete end-to-end user experience from voice input to physical action.

**Independent Test**: Can be fully tested by providing a voice command to the robot and verifying that it correctly identifies objects, plans actions, and executes the task sequence successfully.

**Acceptance Scenarios**:

1. **Given** robot is on charging dock and listening for commands, **When** user says "Bring the red cup to the kitchen sink", **Then** robot processes the command, identifies the red cup, plans a path to grasp it, navigates to the sink, and places the cup there within the time limits.

2. **Given** robot has processed a complex multi-step command, **When** robot executes the skill chain (grasp → navigate → place), **Then** each step completes successfully with appropriate error handling if any step fails.

---

### User Story 2 - Real-time Vision-Language-Action Processing (Priority: P1)

A robotics system processes visual input and language commands simultaneously to perform complex manipulation tasks using VLA models that run efficiently on Jetson hardware with limited VRAM.

**Why this priority**: This is essential for the robot to understand its environment and execute tasks based on both visual perception and language understanding.

**Independent Test**: Can be tested by providing visual scenes with objects and language commands to verify that the VLA model correctly interprets both modalities and generates appropriate action sequences.

**Acceptance Scenarios**:

1. **Given** robot observes a scene with multiple objects, **When** user specifies a particular object to manipulate, **Then** robot correctly identifies the target object and executes the appropriate action.

---

### User Story 3 - LLM Reliability and Fallback Systems (Priority: P2)

A robotics system uses LLMs for planning and decision-making but has reliable fallback mechanisms to classical planning or imitation when the LLM fails or produces unreliable outputs.

**Why this priority**: This ensures the robot can operate safely and reliably even when the primary AI systems fail or produce unexpected results.

**Independent Test**: Can be tested by introducing scenarios where the LLM produces incorrect or unreliable outputs and verifying that the system correctly falls back to classical planning methods.

**Acceptance Scenarios**:

1. **Given** LLM produces an unreliable plan or gets stuck in a loop, **When** self-correction mechanisms detect the issue, **Then** system falls back to classical planning or imitation methods to complete the task.

---

### User Story 4 - Safe Real-Robot Deployment (Priority: P1)

A robotics system operates with safety mechanisms including heartbeat monitoring, emergency stop capabilities, and gradual autonomy levels that can transition from shadow mode to full autonomy as confidence increases.

**Why this priority**: Safety is critical for real-world robot deployment and ensures the system can be controlled and stopped if needed.

**Independent Test**: Can be tested by verifying safety mechanisms work correctly, including emergency stop functionality and heartbeat monitoring.

**Acceptance Scenarios**:

1. **Given** robot is operating autonomously, **When** heartbeat monitor fails to receive signals, **Then** robot enters safe state and stops movement.

2. **Given** user activates emergency stop, **When** emergency stop command is received, **Then** robot immediately stops all movement and enters safe state.

---

### Edge Cases

- What happens when the VLA model encounters objects not seen during training?
- How does the system handle ambiguous language commands (e.g., "the cup" when multiple cups are present)?
- What happens when the robot cannot successfully grasp an object after multiple attempts?
- How does the system handle partial task failures during the skill chain execution?
- What happens when the robot's hardware resources (battery, processing power) become limited during task execution?
- How does the system handle situations where the initial perception was incorrect and the plan needs to be adjusted mid-execution?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support VLA models (Open-VLA-7B, RT-2-X, Octo-Small, Gemini-Robotics-1.5) that can run on Jetson hardware with 16GB VRAM
- **FR-002**: System MUST include speech recognition using Faster-Whisper (large-v3) running at 8× real-time on Orin
- **FR-003**: System MUST convert voice commands into structured ROS 2 goals using appropriate LLM prompt templates
- **FR-004**: System MUST include a skill library with chaining capabilities (grasp → navigate → place) for complex task execution
- **FR-005**: System MUST implement chain-of-thought prompting and self-correction mechanisms for LLM reliability
- **FR-006**: System MUST support TensorRT-LLM conversion from .safetensors format for efficient inference on Jetson
- **FR-007**: System MUST stream tokens while the robot is moving without blocking execution
- **FR-008**: System MUST include heartbeat monitoring and emergency stop action server for safety
- **FR-009**: System MUST support gradual autonomy levels (shadow mode → shared control → full autonomy)
- **FR-010**: System MUST execute the "Hey Robot, Tidy the Apartment" capstone task within 3 minutes in simulation and 5 minutes on real hardware
- **FR-011**: System MUST provide a single ROS2 launch file (launch_capstone.py) that works both in Isaac Sim and on real Jetson + robot
- **FR-012**: System MUST include performance metrics comparing tokens/sec vs success rate on common tasks

### Key Entities

- **VLA Model**: Vision-Language-Action model that processes visual and language inputs to generate robot actions
- **Speech Recognition System**: Component that converts voice commands to text using Faster-Whisper
- **ROS 2 Goals**: Structured action commands that the robot executes in sequence
- **Skill Library**: Collection of reusable robot behaviors (grasp, navigate, place) that can be chained together
- **Safety Monitor**: System components including heartbeat monitor and emergency stop server for safe operation
- **Autonomy Levels**: Different operational modes (shadow, shared control, full autonomy) that control the level of human oversight

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Robot completes the "Hey Robot, Tidy the Apartment" capstone task in ≤3 minutes in simulation and ≤5 minutes on real hardware
- **SC-002**: VLA models achieve sufficient tokens/sec performance while maintaining acceptable success rates on common robotics tasks
- **SC-003**: Voice-to-action pipeline processes commands with end-to-end latency suitable for real-time interaction
- **SC-004**: System operates safely with zero critical safety failures during real robot deployment
- **SC-005**: Robot successfully executes skill chains (grasp → navigate → place) with >80% success rate on standard objects
- **SC-006**: Speech recognition system achieves 95% accuracy in converting voice commands to text in typical environments
- **SC-007**: LLM reliability mechanisms successfully detect and correct or fallback from unreliable outputs in >90% of cases
