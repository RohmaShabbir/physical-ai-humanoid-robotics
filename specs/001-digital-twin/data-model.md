# Data Model: Digital Twin (Gazebo & Unity)

## 1. Digital Twin Environment

### Description
Represents the virtual simulation space that mirrors the physical robot's environment, including physics properties, lighting, and objects.

### Fields
- `id`: Unique identifier for the environment
- `name`: Environment name (e.g., "Apartment-01", "Kitchen-Test")
- `type`: Environment type (e.g., "apartment", "laboratory", "outdoor")
- `description`: Human-readable description of the environment
- `sdf_path`: Path to the SDF file defining the environment
- `physics_properties`: Physics engine configuration parameters
- `lighting_settings`: Environmental lighting configuration
- `object_count`: Number of objects in the environment
- `creation_date`: Timestamp of environment creation
- `version`: Version of the environment definition

### Relationships
- Contains multiple `EnvironmentObject` instances
- Associated with one `RobotModel` instance
- Uses specific `PhysicsEngine` configuration

## 2. Sensor Simulation Layer

### Description
Represents the abstraction that provides realistic sensor data including noise models, biases, and environmental effects.

### Fields
- `id`: Unique identifier for the sensor simulation
- `sensor_type`: Type of sensor (e.g., "depth_camera", "imu", "lidar", "rgb_camera")
- `model_name`: Specific sensor model (e.g., "RealSense_D435i", "Hokuyo_URG-04LX")
- `noise_parameters`: Configuration for sensor noise models
- `bias_parameters`: Configuration for sensor bias models
- `calibration_data`: Intrinsic and extrinsic calibration parameters
- `update_rate`: Sensor data update frequency (Hz)
- `range_min`: Minimum sensing range
- `range_max`: Maximum sensing range
- `fov_horizontal`: Horizontal field of view (degrees)
- `fov_vertical`: Vertical field of view (degrees)
- `resolution`: Sensor resolution (width x height for cameras)

### Relationships
- Belongs to one `RobotModel` instance
- Associated with specific `NoiseModel` configuration
- Connected to ROS 2 topics for data publishing

## 3. Robot Model Converter

### Description
Represents the process that transforms URDF robot descriptions into simulation-ready SDF formats with proper physical properties.

### Fields
- `id`: Unique identifier for the conversion process
- `source_format`: Input format (e.g., "URDF", "XACRO")
- `target_format`: Output format (e.g., "SDF")
- `source_path`: Path to source URDF/XACRO file
- `target_path`: Path to output SDF file
- `conversion_date`: Timestamp of conversion
- `robot_name`: Name of the robot being converted
- `degrees_of_freedom`: Number of joints/degrees of freedom
- `conversion_options`: Specific options used during conversion
- `validation_status`: Whether the converted model passed validation tests

### Relationships
- Converts from `URDFModel` to `SDFModel`
- Uses specific `ConversionTool` (e.g., `gz sdf -p`)
- Associated with `PhysicsProperties` for the robot

## 4. Apartment Environment Generator

### Description
Represents the procedural system that creates realistic household environments with appropriate object placement and semantic labeling.

### Fields
- `id`: Unique identifier for the generator configuration
- `environment_type`: Type of environment to generate (e.g., "apartment", "house", "office")
- `room_count`: Number of rooms to generate
- `room_types`: List of room types to include (e.g., ["kitchen", "living_room", "bedroom"])
- `object_categories`: List of object categories to place (e.g., ["furniture", "kitchenware", "electronics"])
- `furniture_density`: Density of furniture placement (0.0 to 1.0)
- `object_placement_strategy`: Algorithm for object placement
- `semantic_labels`: Whether to include semantic labeling for objects
- `navigation_mesh`: Whether to generate navigation mesh for path planning

### Relationships
- Generates multiple `EnvironmentObject` instances
- Creates `DigitalTwinEnvironment` instances
- Uses `ObjectLibrary` for available objects

## 5. Environment Object

### Description
Represents a physical object within the simulation environment.

### Fields
- `id`: Unique identifier for the object
- `name`: Object name (e.g., "cup_001", "chair_002")
- `category`: Object category (e.g., "furniture", "kitchenware", "electronics")
- `model_path`: Path to the 3D model file
- `physical_properties`: Mass, friction, restitution, etc.
- `visual_properties`: Color, texture, material properties
- `position`: 3D position in the environment (x, y, z)
- `orientation`: 3D orientation in the environment (roll, pitch, yaw)
- `scale`: Scaling factor for the object
- `is_static`: Whether the object is fixed in place
- `collision_properties`: Collision detection settings
- `semantic_label`: Label for perception training (e.g., "cup", "chair")

### Relationships
- Belongs to one `DigitalTwinEnvironment` instance
- Referenced by `ApartmentEnvironmentGenerator` during placement

## 6. Physics Engine Configuration

### Description
Represents the physics simulation parameters and settings.

### Fields
- `id`: Unique identifier for the physics configuration
- `engine_name`: Name of the physics engine (e.g., "bullet", "ode", "dart")
- `gravity`: Gravity vector (x, y, z)
- `real_time_factor`: Target simulation speed relative to real time
- `max_step_size`: Maximum physics step size (seconds)
- `real_time_update_rate`: Target update rate (Hz)
- `solver_type`: Type of physics solver
- `iterations`: Number of solver iterations per step
- `contact_surface_layer`: Thickness of contact surface layer
- `cfm`: Constraint force mixing parameter
- `erp`: Error reduction parameter

### Relationships
- Used by `DigitalTwinEnvironment` instances
- Configures the underlying physics simulation

## 7. Noise Model

### Description
Represents the mathematical model for sensor noise and error characteristics.

### Fields
- `id`: Unique identifier for the noise model
- `model_type`: Type of noise model (e.g., "gaussian", "allan_variance", "realistic_profile")
- `sensor_type`: Type of sensor the model applies to
- `parameters`: Specific parameters for the noise model
- `reference_source`: Source of the noise model (e.g., manufacturer specs, empirical data)
- `validation_metrics`: Metrics for validating the noise model
- `bias_terms`: Bias parameters (offset, drift, etc.)

### Relationships
- Applied to `SensorSimulationLayer` instances
- Referenced by sensor plugins in simulation

## 8. Robot Model (SDF)

### Description
Represents the robot model in SDF format with all physical and kinematic properties.

### Fields
- `id`: Unique identifier for the robot model
- `name`: Robot name
- `sdf_path`: Path to the SDF file
- `degrees_of_freedom`: Number of controllable joints
- `links`: List of robot links with properties
- `joints`: List of robot joints with properties
- `materials`: List of materials used in the model
- `sensors`: List of sensors attached to the robot
- `actuators`: List of actuators with properties
- `collision_bitmasks`: Collision bitmask configurations
- `inertial_properties`: Mass, center of mass, and inertia tensor for each link
- `visual_properties`: Visual representation of each link

### Relationships
- Generated from `URDFModel` via `RobotModelConverter`
- Used in `DigitalTwinEnvironment` instances
- Connected to `SensorSimulationLayer` instances