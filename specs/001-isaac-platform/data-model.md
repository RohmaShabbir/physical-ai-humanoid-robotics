# Data Model: Isaac Platform AI Robot Brain

## 1. Isaac Perception Stack

### Description
Represents the collection of Isaac ROS GEMs (cuVSLAM, stereo disparity, AprilTag, PeopleNet, Pose Estimation) that processes sensor data at high frame rates.

### Fields
- `id`: Unique identifier for the perception stack instance
- `cuVSLAM_enabled`: Boolean indicating if Visual SLAM is active
- `stereo_disparity_enabled`: Boolean indicating if stereo processing is active
- `apriltag_enabled`: Boolean indicating if AprilTag detection is active
- `peoplenet_enabled`: Boolean indicating if PeopleNet is active
- `pose_estimation_enabled`: Boolean indicating if pose estimation is active
- `frame_rate`: Current processing frame rate in Hz
- `processing_latency`: Average processing latency in milliseconds
- `sensor_data_input`: Reference to input sensor data streams
- `output_pose`: Estimated pose from SLAM
- `output_objects`: Detected objects and their properties
- `output_poses`: Estimated body poses from pose estimation

### Relationships
- Processes data from `SensorData` streams
- Outputs to `RobotState` and `NavigationSystem`
- Configured by `PerceptionConfig`

## 2. Nvblox 3D Mapper

### Description
Represents the dense 3D reconstruction system that creates environment maps for navigation planning.

### Fields
- `id`: Unique identifier for the Nvblox mapper instance
- `map_resolution`: Resolution of the 3D map in meters
- `map_bounds`: Bounding box defining the map extent
- `tsdf_integration_weight`: Weight for TSDF integration
- `max_integration_distance_m`: Maximum distance for integration
- `voxel_size`: Size of each voxel in meters
- `voxels_per_side`: Number of voxels per side of each block
- `truncation_distance`: TSDF truncation distance in meters
- `occupancy_threshold`: Threshold for occupancy determination
- `esdf_enabled`: Boolean indicating if ESDF computation is active
- `clearing_ray_length_m`: Length of clearing rays in meters
- `max_ray_length_m`: Maximum ray length in meters
- `map_accuracy`: Current map accuracy in meters

### Relationships
- Receives data from `IsaacPerceptionStack`
- Outputs 3D map to `NavigationSystem`
- Configured by `MappingConfig`

## 3. Legged Navigation System

### Description
Represents the combination of Nav2 SMAC planner, legged footprint costmap plugin, and footstep planning for legged robot locomotion.

### Fields
- `id`: Unique identifier for the navigation system instance
- `global_planner`: Name/type of global planner (SMAC)
- `local_planner`: Name/type of local planner (DWA/MPC)
- `footstep_planner_enabled`: Boolean indicating if footstep planning is active
- `legged_footprint`: Configuration for legged robot footprint
- `costmap_resolution`: Resolution of costmap in meters
- `costmap_width`: Width of costmap in meters
- `costmap_height`: Height of costmap in meters
- `robot_radius`: Radius of robot for collision checking
- `foot_separation`: Distance between feet for stability
- `step_height`: Maximum step height for footstep planning
- `step_length`: Maximum step length for footstep planning
- `navigation_goal`: Current navigation goal coordinates
- `navigation_status`: Current status of navigation (active, failed, complete)

### Relationships
- Uses maps from `Nvblox3DMapper`
- Receives sensor data from `IsaacPerceptionStack`
- Outputs commands to `RobotController`
- Configured by `NavigationConfig`

## 4. Domain Randomization Pipeline

### Description
Represents the Replicator-based system that applies lighting, texture, pose, camera, and physics randomization for sim-to-real transfer.

### Fields
- `id`: Unique identifier for the domain randomization pipeline
- `lighting_randomization`: Configuration for lighting variations
- `texture_randomization`: Configuration for texture variations
- `pose_randomization`: Configuration for object pose variations
- `camera_randomization`: Configuration for camera parameter variations
- `physics_randomization`: Configuration for physics parameter variations
- `curriculum_stage`: Current stage in curriculum learning (0-1 scale)
- `randomization_intensity`: Current intensity of randomization (0-1 scale)
- `texture_library_path`: Path to library of textures for randomization
- `material_properties_range`: Range of material property variations
- `lighting_properties_range`: Range of lighting parameter variations
- `physics_properties_range`: Range of physics parameter variations

### Relationships
- Applied to `IsaacSimulationEnvironment`
- Configures `SimulationParameters`
- Influences `TrainingDataGenerator`

## 5. Jetson Orin Platform

### Description
Represents the embedded computing platform that runs all perception, planning, and inference workloads with thermal and power constraints.

### Fields
- `id`: Unique identifier for the Jetson platform instance
- `platform_model`: Model of Jetson platform (Orin NX 16GB)
- `jetpack_version`: Version of JetPack installed
- `gpu_memory`: Total GPU memory in GB
- `cpu_cores`: Number of CPU cores
- `thermal_throttling_active`: Boolean indicating if thermal throttling is active
- `current_power_draw`: Current power consumption in watts
- `current_temperature`: Current temperature in Celsius
- `available_memory`: Available system memory in GB
- `available_gpu_memory`: Available GPU memory in GB
- `max_power_mode`: Boolean indicating if max power mode is active
- `gui_disabled`: Boolean indicating if GUI is disabled
- `tensorrt_optimized`: Boolean indicating if TensorRT optimization is applied
- `docker_containers_running`: Number of active Docker containers
- `thermal_margin`: Remaining thermal margin before throttling

### Relationships
- Runs `IsaacPerceptionStack`
- Runs `Nvblox3DMapper`
- Runs `LeggedNavigationSystem`
- Configured by `DeploymentConfig`

## 6. Robot State

### Description
Represents the current state of the robot including pose, velocity, and sensor data.

### Fields
- `id`: Unique identifier for the robot state
- `timestamp`: Timestamp of the state measurement
- `position`: 3D position vector (x, y, z)
- `orientation`: Orientation as quaternion (x, y, z, w)
- `linear_velocity`: Linear velocity vector (x, y, z)
- `angular_velocity`: Angular velocity vector (x, y, z)
- `joint_positions`: Array of joint positions
- `joint_velocities`: Array of joint velocities
- `joint_efforts`: Array of joint efforts
- `battery_level`: Current battery level (0-1)
- `robot_mode`: Current operational mode (navigation, idle, emergency)
- `safety_status`: Current safety status (safe, warning, error)

### Relationships
- Updated by `IsaacPerceptionStack`
- Used by `LeggedNavigationSystem`
- Monitored by `SafetySystem`

## 7. Navigation Goal

### Description
Represents a navigation goal with coordinates and constraints for the legged navigation system.

### Fields
- `id`: Unique identifier for the navigation goal
- `goal_x`: X coordinate of goal position
- `goal_y`: Y coordinate of goal position
- `goal_z`: Z coordinate of goal position
- `orientation_tolerance`: Tolerance for goal orientation in radians
- `position_tolerance`: Tolerance for goal position in meters
- `path_tolerance`: Tolerance for path following in meters
- `goal_priority`: Priority level of the goal (1-10)
- `goal_constraints`: Additional constraints for navigation
- `timeout`: Timeout for reaching the goal in seconds
- `recovery_behavior`: Recovery behavior if goal cannot be reached

### Relationships
- Used by `LeggedNavigationSystem`
- Created by `NavigationInterface`
- Monitored by `NavigationMonitor`

## 8. Training Data Sample

### Description
Represents a sample of training data generated by the domain randomization pipeline.

### Fields
- `id`: Unique identifier for the training sample
- `image_data`: Image data for perception training
- `depth_data`: Depth data for 3D understanding
- `semantic_labels`: Semantic segmentation labels
- `randomization_parameters`: Parameters used for domain randomization
- `simulation_metadata`: Metadata about simulation conditions
- `ground_truth_pose`: Ground truth pose for training
- `timestamp`: Timestamp of data generation
- `curriculum_level`: Curriculum learning level at generation
- `quality_score`: Quality score of the training sample
- `domain_gap_measure`: Measure of sim-to-real domain gap

### Relationships
- Generated by `DomainRandomizationPipeline`
- Used for training `PerceptionModels`
- Stored in `TrainingDataset`