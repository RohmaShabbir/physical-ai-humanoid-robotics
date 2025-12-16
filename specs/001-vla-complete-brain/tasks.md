# Implementation Tasks: Vision-Language-Action System

**Feature**: 001-vla-complete-brain
**Date**: 2025-12-12
**Spec**: [spec.md](./spec.md)
**Plan**: [plan.md](./plan.md)
**Input**: `/sp.task Module 4 – Vision-Language-Action (The Complete Brain + Capstone)`

## Task Checklist Format

All tasks follow the format:
```text
- [ ] T### [P?] [US#] Task description with file path
```

Where:
- `T###` = Sequential task number
- `[P]` = Parallelizable task (optional)
- `[US#]` = User Story reference (required for story tasks)

## Phase 1: Setup (Project initialization)

- [X] T001 Create project structure in src/ with all required subdirectories per plan.md
- [ ] T002 Set up Python virtual environment with Python 3.11
- [ ] T003 Install base dependencies: ROS 2 Humble, PyTorch, OpenCV, Transformers
- [X] T004 Create requirements.txt with all project dependencies
- [X] T005 Initialize ROS 2 package structure with package.xml and setup.py
- [X] T006 Create config directory and initialize base configuration files
- [X] T007 Create tests directory structure with subdirectories
- [X] T008 Set up documentation directory structure for Docusaurus

## Phase 2: Foundational (Blocking prerequisites)

- [X] T009 [P] Implement ROS 2 interfaces in src/ros_interfaces/action_interfaces.py
- [X] T010 [P] Implement message converters in src/ros_interfaces/message_converters.py
- [X] T011 [P] Create ROS 2 launch directory structure in src/capstone_bringup/launch/
- [X] T012 [P] Create base configuration files (vla_config.yaml, voice_config.yaml, safety_config.yaml, capstone_config.yaml)
- [X] T013 [P] Set up basic logging and monitoring utilities
- [X] T014 [P] Implement performance metrics collection framework

## Phase 3: User Story 1 - Voice Command Processing and Action Execution (Priority: P1)

**Goal**: Enable end-to-end voice command processing to robot action execution
**Independent Test**: Provide voice command to robot and verify object identification, action planning, and task execution

### Phase 3A: Voice Pipeline Implementation
- [X] T015 [P] [US1] Implement speech recognition module in src/voice_pipeline/speech_recognition.py
- [X] T016 [P] [US1] Create audio input handler in src/voice_pipeline/audio_input.py
- [X] T017 [P] [US1] Implement voice processor with command parsing in src/voice_pipeline/voice_processor.py
- [X] T018 [US1] Create VoiceCommandService interface implementation following contracts/voice_command_service.yaml

### Phase 3B: VLA Model Integration
- [X] T019 [P] [US1] Implement TensorRT engine utilities in src/vla_inference/tensorrt_engine.py
- [X] T020 [P] [US1] Create VLA model wrapper in src/vla_inference/vla_model.py
- [X] T021 [US1] Implement VLA inference service following contracts/vla_inference_action.yaml
- [X] T022 [US1] Create VLA action request handling in src/vla_inference/action_handler.py

### Phase 3C: Skill Library Implementation
- [X] T023 [P] [US1] Implement grasp skill in src/skill_library/grasp_skill.py
- [X] T024 [P] [US1] Implement navigation skill in src/skill_library/navigation_skill.py
- [X] T025 [P] [US1] Implement place skill in src/skill_library/place_skill.py
- [X] T026 [US1] Create skill chain executor in src/skill_library/skill_chain.py
- [X] T027 [US1] Create SkillExecutionAction interface implementation for skill execution

### Phase 3D: Integration and Testing
- [X] T028 [US1] Integrate voice pipeline with VLA inference service
- [X] T029 [US1] Connect VLA output to skill chain execution
- [X] T030 [US1] Implement LLM prompt template for converting voice to ROS 2 goals
- [X] T031 [US1] Test basic voice command "Bring the red cup to the kitchen sink" in simulation
- [X] T032 [US1] Measure end-to-end latency (speech → action start) < 2.0 seconds
- [X] T033 [US1] Verify task completion within 3 minutes in simulation (success criteria SC-001)

## Phase 4: User Story 2 - Real-time Vision-Language-Action Processing (Priority: P1)

**Goal**: Enable real-time VLA processing with visual and language inputs on Jetson hardware
**Independent Test**: Provide visual scenes with language commands and verify correct interpretation and action generation

### Phase 4A: VLA Model Optimization
- [ ] T034 [P] [US2] Create TensorRT-LLM engine build script for VLA models
- [ ] T035 [US2] Implement quantized VLA model loading (8GB VRAM compatibility)
- [ ] T036 [US2] Optimize VLA inference for streaming token processing (non-blocking)
- [ ] T037 [US2] Implement performance monitoring for tokens/sec measurement

### Phase 4B: Real-time Processing
- [ ] T038 [US2] Create real-time camera feed integration with VLA model
- [ ] T039 [US2] Implement streaming inference that works while robot moves
- [ ] T040 [US2] Optimize VRAM usage for continuous operation on Jetson
- [ ] T041 [US2] Test VLA model with multiple object scenarios and ambiguous commands

### Phase 4C: Performance Validation
- [ ] T042 [US2] Validate tokens/sec vs success rate performance metrics (FR-012)
- [ ] T043 [US2] Verify 80% success rate on standard tidy tasks in simulation (SC-002)
- [ ] T044 [US2] Test with various object types and environments for generalization

## Phase 5: User Story 4 - Safe Real-Robot Deployment (Priority: P1)

**Goal**: Implement safety mechanisms with heartbeat monitoring and emergency stop for real robot deployment
**Independent Test**: Verify safety mechanisms work including emergency stop and heartbeat monitoring

### Phase 5A: Safety System Implementation
- [ ] T045 [P] [US4] Implement heartbeat monitor in src/safety_system/heartbeat_monitor.py
- [ ] T046 [P] [US4] Create emergency stop server in src/safety_system/emergency_stop_server.py
- [ ] T047 [US4] Implement safety manager for autonomy level control in src/safety_system/safety_manager.py
- [ ] T048 [US4] Create SafetyMonitorService interface for safety operations

### Phase 5B: Autonomy Control
- [ ] T049 [US4] Implement shadow mode functionality
- [ ] T050 [US4] Implement shared control mode functionality
- [ ] T051 [US4] Implement full autonomy mode functionality
- [ ] T052 [US4] Create autonomy level transition logic and safety checks

### Phase 5C: Safety Validation
- [ ] T053 [US4] Test heartbeat monitoring with timeout scenarios
- [ ] T054 [US4] Test emergency stop functionality with various execution states
- [ ] T055 [US4] Verify zero critical safety failures during real robot deployment (SC-004)
- [ ] T056 [US4] Test gradual autonomy level transitions in safe environment

## Phase 6: User Story 3 - LLM Reliability and Fallback Systems (Priority: P2)

**Goal**: Implement LLM reliability mechanisms with fallback to classical planning
**Independent Test**: Introduce scenarios where LLM produces incorrect outputs and verify fallback to classical planning

### Phase 6A: Reliability Mechanisms
- [ ] T057 [P] [US3] Implement chain-of-thought prompting for VLA decisions
- [ ] T058 [P] [US3] Create self-correction mechanisms for LLM outputs
- [ ] T059 [US3] Implement tool calling (ReAct style) for VLA system
- [ ] T060 [US3] Create detection system for unreliable LLM outputs

### Phase 6B: Fallback Systems
- [ ] T061 [US3] Implement classical planning fallback mechanisms
- [ ] T062 [US3] Create imitation learning fallback for manipulation tasks
- [ ] T063 [US3] Integrate fallback decision logic with main VLA pipeline
- [ ] T064 [US3] Test fallback scenarios when LLM gets stuck in loops

### Phase 6C: Reliability Validation
- [ ] T065 [US3] Test LLM reliability mechanisms with edge cases
- [ ] T066 [US3] Verify 90% successful detection of unreliable outputs (SC-007)
- [ ] T067 [US3] Test success rate of fallback systems in failure scenarios

## Phase 7: Capstone Integration and Testing

**Goal**: Integrate all components for the "Hey Robot, Tidy the Apartment" capstone project
**Independent Test**: Complete the capstone task end-to-end with voice command to task completion

### Phase 7A: Capstone Launch System
- [ ] T068 [P] Create main system manager in src/capstone_bringup/system_manager.py
- [ ] T069 [P] Create capstone launch file launch_capstone.py that works in Isaac Sim and on real Jetson
- [ ] T070 [P] Create VLA system launch file vla_system.launch.py
- [ ] T071 [P] Integrate all components in main.py entry point

### Phase 7B: Capstone Testing
- [ ] T072 Execute "Hey Robot, Tidy the Apartment" task in simulation: "The red cup is on the table. Bring it to the kitchen sink."
- [ ] T073 Verify robot perceives, plans, walks, grasps, navigates, and places cup in sink
- [ ] T074 Validate task completes in ≤3 minutes in simulation (FR-010)
- [ ] T075 Validate task completes in ≤5 minutes on real hardware (FR-010)
- [ ] T076 Test skill chains (grasp → navigate → place) with >80% success rate (SC-005)

### Phase 7C: Performance Validation
- [ ] T077 Validate speech recognition accuracy ≥95% in typical environments (SC-006)
- [ ] T078 Run 100 recorded voice commands test set with ≥80% success rate
- [ ] T079 Measure and validate all performance metrics per success criteria

## Phase 8: Testing and Quality Assurance

### Phase 8A: Unit Tests
- [ ] T080 [P] Create unit tests for VLA inference components in tests/unit/test_vla_inference.py
- [ ] T081 [P] Create unit tests for voice pipeline components in tests/unit/test_voice_pipeline.py
- [ ] T082 [P] Create unit tests for skill library in tests/unit/test_skill_library.py
- [ ] T083 [P] Create unit tests for safety system in tests/unit/test_safety_system.py

### Phase 8B: Integration Tests
- [ ] T084 Create integration test for voice-to-action pipeline in tests/integration/test_voice_to_action.py
- [ ] T085 Create integration test for skill chaining in tests/integration/test_skill_chaining.py
- [ ] T086 Create integration test for safety integrity in tests/integration/test_safety_integrity.py
- [ ] T087 Create contract tests for ROS interfaces in tests/contract/test_ros_interfaces.py

## Phase 9: Documentation and Deliverables

### Phase 9A: Documentation
- [ ] T088 Create Docusaurus markdown documentation for VLA system architecture
- [ ] T089 Add WebGL embeds to documentation for 3D visualization
- [ ] T090 Create API documentation for all ROS 2 interfaces

### Phase 9B: Demo Videos and Final Delivery
- [ ] T091 Create simulation demo video (≤3 min task completion)
- [ ] T092 Create Jetson-only dry run demo video
- [ ] T093 Create real robot demo video (≤5 min task completion)
- [ ] T094 Package final deliverables: single launch_capstone.py file and 3 demo videos

## Phase 10: Polish & Cross-Cutting Concerns

- [ ] T095 Optimize system performance for Jetson hardware constraints
- [ ] T096 Implement error handling and graceful degradation mechanisms
- [ ] T097 Add comprehensive logging for debugging and monitoring
- [ ] T098 Final validation of all functional requirements (FR-001 through FR-012)
- [ ] T099 Final validation of all success criteria (SC-001 through SC-007)
- [ ] T100 Complete final system integration testing

## Dependencies

**User Story Completion Order**:
1. Phase 2 (Foundational) must complete before any user story phases
2. User Story 1, 2, and 4 (all P1) can proceed in parallel after foundational phase
3. User Story 3 (P2) can start after foundational but benefits from core components
4. Phase 7 (Capstone) requires completion of User Stories 1, 2, and 4

## Parallel Execution Examples

**Per Story Parallelization**:
- **US1**: Tasks T015-T017 (voice pipeline) can run in parallel with T019-T021 (VLA components) and T023-T025 (skills)
- **US2**: TensorRT optimization (T034-T036) can run in parallel with real-time processing (T038-T040)
- **US4**: Heartbeat monitor (T045) and emergency stop (T046) can be developed in parallel
- **US3**: Chain-of-thought (T057) and self-correction (T058) can run in parallel

## Implementation Strategy

**MVP Scope**: Just User Story 1 (core voice-to-action pipeline) with basic safety mechanisms for US4, sufficient for demonstrating basic functionality.

**Incremental Delivery**:
1. Complete Phase 1 (Setup) and Phase 2 (Foundational)
2. Complete US1 (Voice Command Processing) for core functionality
3. Complete US4 (Safety) for basic safety
4. Complete US2 (VLA Processing) for full capabilities
5. Complete US3 (Reliability) for robustness
6. Complete Phases 7-10 for final delivery

## Final Deliverable Validation

- ✅ Single ros2 launch file launch_capstone.py works in Isaac Sim and on real Jetson (FR-011)
- ✅ "Hey Robot, Tidy the Apartment" task completes ≤3 min simulation, ≤5 min real hardware (FR-010)
- ✅ 3 demo videos created: simulation, Jetson-only dry run, real robot run
- ✅ All functional requirements satisfied (FR-001 through FR-012)
- ✅ All success criteria validated (SC-001 through SC-007)
- ✅ End-to-end latency < 2.0 seconds (SC-003)
- ✅ 100 voice commands test set with ≥80% success rate