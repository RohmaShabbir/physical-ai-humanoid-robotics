# Quickstart Guide: Digital Twin (Gazebo & Unity)

## Prerequisites

Before starting with the digital twin simulation, ensure you have the following prerequisites installed:

### System Requirements
- Ubuntu 22.04 LTS (recommended) or compatible Linux distribution
- NVIDIA GPU with CUDA support (for Isaac Sim container)
- At least 16GB RAM (32GB recommended for complex simulations)
- At least 50GB free disk space
- Multi-core processor (8+ cores recommended)

### Software Dependencies
- Docker and NVIDIA Container Toolkit
- ROS 2 Jazzy
- Git
- Python 3.11

## Installation

### 1. Install NVIDIA Isaac Sim Container

The recommended approach is to use the NVIDIA Isaac Sim container which includes Gazebo Harmonic:

```bash
# Pull the Isaac Sim container
docker pull nvcr.io/nvidia/isaac/isaac-sim:2024.2.0

# Run the container with GUI support
xhost +local:docker
docker run --name isaac_sim --gpus all -e "ACCEPT_EULA=Y" --rm -it \
  -p 5000:5000 -p 5555:5555 -p 50051:50051 \
  --network=host \
  --env "NVIDIA_VISIBLE_DEVICES=all" \
  --env "NVIDIA_DRIVER_CAPABILITIES=all" \
  nvcr.io/nvidia/isaac/isaac-sim:2024.2.0
```

### 2. Clone the Repository

```bash
git clone https://github.com/your-organization/physical-ai-humanoid-robotics.git
cd physical-ai-humanoid-robotics
```

### 3. Set Up ROS 2 Workspace

```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Create workspace if not already done
mkdir -p ~/digital-twin-ws/src
cd ~/digital-twin-ws/src

# Link the project
ln -s /path/to/physical-ai-humanoid-robotics ~/digital-twin-ws/src/
cd ~/digital-twin-ws

# Build the workspace
colcon build --packages-select humanoid_gazebo apartment_worlds
source install/setup.bash
```

## Launch the Digital Twin Environment

### 1. Start the Apartment Environment with Humanoid Robot

```bash
# Launch the digital twin environment
ros2 launch humanoid_gazebo apartment.launch.py
```

This command will:
- Start Gazebo Harmonic with the apartment environment
- Load the humanoid robot model with realistic physics
- Initialize all sensors (depth camera, IMU, LIDAR)
- Position the robot in a standard starting pose

### 2. Verify the Launch

The environment should start within 15 seconds. You can verify the launch by checking:

```bash
# List active ROS 2 topics
ros2 topic list

# Look for sensor topics
ros2 topic list | grep /sensor
```

Expected topics include:
- `/camera/depth/image_raw` - Depth camera data
- `/camera/rgb/image_raw` - RGB camera data
- `/imu/data` - IMU sensor data
- `/lidar/scan` - LIDAR scan data
- `/joint_states` - Robot joint states

## Basic Robot Control

### 1. Send Joint Commands

```bash
# Send a simple joint command using ROS 2
ros2 topic pub /joint_group_position_controller/commands std_msgs/Float64MultiArray "data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
```

### 2. Monitor Sensor Data

```bash
# Listen to depth camera data
ros2 topic echo /camera/depth/image_raw

# Listen to IMU data
ros2 topic echo /imu/data
```

## Unity Integration (Optional)

### 1. Export to Unity

If you want to use Unity for photorealistic rendering:

```bash
# Export the robot and environment as USD
python3 scripts/export_to_usd.py --robot-model humanoid --environment apartment-01 --output-path ./unity_assets
```

### 2. Import into Unity

1. Open Unity 2025 LTS
2. Import the exported USD files
3. Add the ROS-TCP-Connector package
4. Configure the connection to your ROS 2 network

## API Usage Examples

### 1. List Available Environments

```bash
curl -X GET http://localhost:8080/environments
```

### 2. Launch an Environment

```bash
curl -X POST http://localhost:8080/environments/apartment-01/launch \
  -H "Content-Type: application/json" \
  -d '{
    "robot_pose": {
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    },
    "simulation_speed": 1.0,
    "enable_gui": true
  }'
```

### 3. Get Robot Joint States

```bash
curl -X GET http://localhost:8080/robots/humanoid-robot/joints
```

### 4. Send Joint Commands

```bash
curl -X POST http://localhost:8080/robots/humanoid-robot/joints/commands \
  -H "Content-Type: application/json" \
  -d '{
    "names": ["joint1", "joint2", "joint3"],
    "positions": [0.1, -0.5, 0.3],
    "velocities": [0.0, 0.0, 0.0]
  }'
```

## Quality Validation

### 1. Performance Check

Verify the simulation runs in real-time:

```bash
# Monitor real-time factor
gz stats
```

Expected: Real-time factor ≥ 0.8 for smooth simulation.

### 2. Sensor Accuracy Check

Validate depth camera accuracy:

```bash
# Run sensor validation script
python3 scripts/validate_depth_camera.py --reference-data real_d435i_data --simulated-data gazebo_d435i_data
```

Expected: RMSE < 3% compared to real Realsense D435i data.

### 3. Physics Validation

Check physics accuracy:

```bash
# Run physics validation
python3 scripts/validate_physics.py --robot humanoid --motion walking
```

Expected: Sim-to-real transfer rate ≥ 80%.

## Troubleshooting

### Common Issues

1. **Simulation runs too slowly**:
   - Check real-time factor with `gz stats`
   - Reduce environment complexity or physics accuracy settings
   - Ensure GPU acceleration is working

2. **Sensors not publishing data**:
   - Verify Gazebo plugins are loaded: `gz topic -l`
   - Check for plugin errors in console output
   - Ensure sensor topics are connected to ROS bridge

3. **Robot model not loading**:
   - Check SDF file validity: `gz sdf -k model.sdf`
   - Verify all mesh files are accessible
   - Ensure URDF to SDF conversion completed successfully

### Debugging Commands

```bash
# Check Gazebo topics
gz topic -l

# Check ROS 2 topics
ros2 topic list

# Monitor Gazebo performance
gz stats

# Check ROS 2 node health
ros2 node list
```

## Next Steps

1. **Advanced Control**: Implement more complex robot behaviors using the control interfaces
2. **Environment Customization**: Create custom apartment layouts and object arrangements
3. **Algorithm Testing**: Test your robotics algorithms in the simulation before real-world deployment
4. **Unity Workflow**: Explore the Unity integration for photorealistic rendering and perception training