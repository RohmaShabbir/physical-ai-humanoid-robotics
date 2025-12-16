# Implementation Tasks: Digital Twin (Gazebo & Unity)

**Feature**: 001-digital-twin
**Branch**: 001-digital-twin
**Input**: Implementation plan from `/specs/001-digital-twin/plan.md`
**Output**: `ros2 launch humanoid_gazebo apartment.launch.py` starts in <15s with fully sensorized humanoid in Apartment-01

## Task Structure

### Phase 0: Environment Setup (Dependency: None)
- **T0.1**: Set up Isaac Sim container environment [4h]
  - T0.1.1: Install NVIDIA Container Toolkit [1h]
  - T0.1.2: Pull and configure Isaac Sim 2024.2 container [2h]
  - T0.1.3: Test basic Gazebo Harmonic functionality [1h]
  - **Deliverable**: Working Isaac Sim container with Gazebo Harmonic

- **T0.2**: Set up ROS 2 workspace structure [2h]
  - T0.2.1: Create src/humanoid_gazebo package [1h]
  - T0.2.2: Create src/apartment_worlds package [1h]
  - **Deliverable**: Basic workspace structure with required packages

### Phase 1: Robot Model Conversion (Dependency: T0.1, T0.2)
- **T1.1**: Convert URDF to SDF with automated tool [3h]
  - T1.1.1: Run gz sdf -p on existing URDF [1h]
  - T1.1.2: Identify the 7 common conversion issues [1h]
  - T1.1.3: Document differences between URDF and SDF [1h]
  - **Deliverable**: Initial SDF model with conversion report

- **T1.2**: Fix SDF conversion issues (the 7 known bugs) [6h]
  - T1.2.1: Add collision bitmasks for selective collision detection [1h]
  - T1.2.2: Implement surface friction parameters [1h]
  - T1.2.3: Fix self-collision handling for hands [1h]
  - T1.2.4: Recalculate accurate inertial properties [1h]
  - T1.2.5: Set proper joint limits, effort, and velocity constraints [1h]
  - T1.2.6: Add realistic actuator modeling (effort, velocity limits, damping) [1h]
  - **Deliverable**: Corrected SDF model with all 7 issues resolved

- **T1.3**: Validate physics properties [2h]
  - T1.3.1: Test joint dynamics and constraints [1h]
  - T1.3.2: Verify inertial properties and mass distribution [1h]
  - **Deliverable**: Physics-validated SDF model

### Phase 2: Sensor Simulation (Dependency: T1.2)
- **T2.1**: Implement depth camera with RealSense D435i profile [4h]
  - T2.1.1: Add depth camera sensor to robot model [1h]
  - T2.1.2: Configure RealSense D435i noise model [2h]
  - T2.1.3: Add lens distortion and depth noise parameters [1h]
  - **Deliverable**: Depth camera with realistic noise model

- **T2.2**: Implement IMU with realistic bias & noise [3h]
  - T2.2.1: Add IMU sensor to robot model [1h]
  - T2.2.2: Configure Allan variance-based noise model [1h]
  - T2.2.3: Add bias and drift parameters [1h]
  - **Deliverable**: IMU with realistic noise characteristics

- **T2.3**: Implement LIDAR with dropout and angular noise [3h]
  - T2.3.1: Add LIDAR sensor to robot model [1h]
  - T2.3.2: Configure dropout characteristics [1h]
  - T2.3.3: Add angular noise parameters [1h]
  - **Deliverable**: LIDAR with realistic noise model

- **T2.4**: Implement RGB camera with distortion and rolling shutter [3h]
  - T2.4.1: Add RGB camera sensor to robot model [1h]
  - T2.4.2: Configure lens distortion parameters [1h]
  - T2.4.3: Add rolling shutter effects [1h]
  - **Deliverable**: RGB camera with realistic effects

- **T2.5**: Validate sensor outputs against real data [4h]
  - T2.5.1: Compare depth camera output to real Realsense data [2h]
  - T2.5.2: Compare IMU output to real sensor data [1h]
  - T2.5.3: Compare LIDAR output to real sensor data [1h]
  - **Deliverable**: Sensor validation report with RMSE < 3% for depth

### Phase 3: Environment Creation (Dependency: T0.2)
- **T3.1**: Create basic apartment environment [4h]
  - T3.1.1: Design kitchen layout with basic furniture [2h]
  - T3.1.2: Design living room layout with basic furniture [2h]
  - **Deliverable**: Basic apartment SDF world file

- **T3.2**: Add household objects and clutter [5h]
  - T3.2.1: Create cup models with realistic properties [1h]
  - T3.2.2: Create chair models with realistic properties [1h]
  - T3.2.3: Create door models with realistic properties [1h]
  - T3.2.4: Add additional household objects [1h]
  - T3.2.5: Position objects realistically in environment [1h]
  - **Deliverable**: Apartment environment with household objects

- **T3.3**: Add semantic labels for VLA training [3h]
  - T3.3.1: Assign semantic labels to objects [1h]
  - T3.3.2: Configure visual tags for perception training [1h]
  - T3.3.3: Validate label accessibility [1h]
  - **Deliverable**: Apartment environment with semantic labels

- **T3.4**: Optimize environment for performance [3h]
  - T3.4.1: Test physics performance with 50+ objects [1h]
  - T3.4.2: Optimize collision meshes for performance [1h]
  - T3.4.3: Verify real-time performance (20+ Hz) [1h]
  - **Deliverable**: Optimized apartment environment

### Phase 4: Gazebo Plugins and Integration (Dependency: T1.2, T2.5)
- **T4.1**: Create custom Gazebo plugins for sensors [5h]
  - T4.1.1: Develop depth camera plugin with noise model [2h]
  - T4.1.2: Develop IMU plugin with realistic characteristics [1.5h]
  - T4.1.3: Develop LIDAR plugin with dropout simulation [1.5h]
  - **Deliverable**: Custom sensor plugins

- **T4.2**: Create physics validation plugin [3h]
  - T4.2.1: Develop plugin for physics stability testing [1.5h]
  - T4.2.2: Implement 10-minute stand test functionality [1.5h]
  - **Deliverable**: Physics validation plugin

- **T4.3**: Integrate ROS 2 control interfaces [4h]
  - T4.3.1: Set up joint state publisher [1h]
  - T4.3.2: Configure joint trajectory controllers [1h]
  - T4.3.3: Integrate sensor data publishers [1h]
  - T4.3.4: Test ROS 2 communication [1h]
  - **Deliverable**: ROS 2 integrated robot model

### Phase 5: Unity Integration Documentation (Dependency: T1.2)
- **T5.1**: Research USD export workflow [2h]
  - T5.1.1: Test USD export from Isaac Sim [1h]
  - T5.1.2: Document export process and limitations [1h]
  - **Deliverable**: USD export documentation

- **T5.2**: Document Unity ROS connection options [2h]
  - T5.2.1: Evaluate ROS-TCP-Connector vs rosbridge vs Unity ROS2 package [1h]
  - T5.2.2: Document benchmark results and recommendations [1h]
  - **Deliverable**: Unity connection documentation

- **T5.3**: Create Unity integration guide [2h]
  - T5.3.1: Document Unity import process for exported USD [1h]
  - T5.3.2: Document ROS connection setup [1h]
  - **Deliverable**: Unity integration guide

### Phase 6: Launch System and Testing (Dependency: T2.5, T3.4, T4.3)
- **T6.1**: Create unified launch system [4h]
  - T6.1.1: Create apartment.launch.py file [2h]
  - T6.1.2: Configure all necessary parameters [1h]
  - T6.1.3: Test basic launch functionality [1h]
  - **Deliverable**: Unified launch file

- **T6.2**: Performance testing [3h]
  - T6.2.1: Measure launch time (target <15s) [1h]
  - T6.2.2: Verify real-time performance with 28-DoF humanoid [1h]
  - T6.2.3: Test with 50+ objects in environment [1h]
  - **Deliverable**: Performance test results

- **T6.3**: Physics stability testing [3h]
  - T6.3.1: Run 10-minute stand test [1h]
  - T6.3.2: Test walking gait stability [1h]
  - T6.3.3: Verify contact dynamics with objects [1h]
  - **Deliverable**: Stability test results

- **T6.4**: Sensor accuracy validation [4h]
  - T6.4.1: Run depth sensor RMSE test vs real data [2h]
  - T6.4.2: Validate IMU noise characteristics [1h]
  - T6.4.3: Validate LIDAR dropout patterns [1h]
  - **Deliverable**: Sensor validation report

### Phase 7: Documentation and Finalization (Dependency: T6.4)
- **T7.1**: Create user documentation [3h]
  - T7.1.1: Write setup guide for Gazebo environment [1h]
  - T7.1.2: Write usage instructions for digital twin [1h]
  - T7.1.3: Document troubleshooting procedures [1h]
  - **Deliverable**: User documentation

- **T7.2**: Create technical documentation [3h]
  - T7.2.1: Document architecture decisions (Gazebo vs Classic fallback) [1h]
  - T7.2.2: Document SDF auto-conversion fixes [1h]
  - T7.2.3: Document Unity path decision (include or mention) [1h]
  - **Deliverable**: Technical documentation

- **T7.3**: Final integration test [2h]
  - T7.3.1: Complete end-to-end test of launch file [1h]
  - T7.3.2: Verify all requirements are met [1h]
  - **Deliverable**: Final test report

## Dependencies Summary
- T1.x depends on T0.x (Environment Setup)
- T2.x depends on T1.2 (Sensor Simulation depends on fixed SDF)
- T4.x depends on T1.2 and T2.5 (Integration depends on validated models)
- T6.x depends on T2.5, T3.4, T4.3 (Launch depends on all components)
- T7.x depends on T6.4 (Documentation depends on final validation)

## Total Estimated Time: 80 hours

## Final Deliverable
A single launch file `ros2 launch humanoid_gazebo apartment.launch.py` that:
1. Starts in <15 seconds on RTX 4070 Ti
2. Loads a fully sensorized humanoid robot in Apartment-01
3. Includes depth camera, IMU, LIDAR with realistic noise models
4. Provides real-time physics simulation (20+ Hz) with 28-DoF humanoid and 50+ objects
5. Passes physics stability test (10-minute stand without falling)
6. Achieves depth sensor RMSE < 3% vs real RealSense data