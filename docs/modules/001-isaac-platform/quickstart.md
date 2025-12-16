# Quickstart Guide: Isaac Platform AI Robot Brain

## Overview
This guide will help you set up and run the Isaac Platform AI Robot Brain that enables autonomous navigation in unknown apartment environments using Isaac Sim and Isaac ROS GEMs.

## Prerequisites

### Hardware Requirements
- NVIDIA Jetson Orin NX 16GB (for deployment)
- Compatible humanoid robot with appropriate sensors (RGB-D camera, IMU, etc.)
- Development machine with NVIDIA GPU for Isaac Sim

### Software Requirements
- JetPack 6.1 on Jetson Orin
- Isaac Sim 2025 (installed via Omniverse Launcher)
- ROS 2 Jazzy
- Docker
- TensorRT 8.6

## Installation

### 1. Set up Isaac Sim 2025
```bash
# Install Isaac Sim via Omniverse Launcher (the only supported method)
# Follow NVIDIA's installation guide for Omniverse Launcher
# Install Isaac Sim 2025 with Core, Replicator, and synthetic data pipeline
```

### 2. Set up ROS 2 Workspace
```bash
# Create and navigate to workspace
mkdir -p ~/isaac_ws/src
cd ~/isaac_ws

# Clone the Isaac Platform packages
git clone https://github.com/your-org/isaac_ros_visual_slam src/isaac_ros_visual_slam
git clone https://github.com/your-org/isaac_ros_nvblox src/isaac_ros_nvblox
git clone https://github.com/your-org/humanoid_nav2 src/humanoid_nav2
git clone https://github.com/your-org/jetson_deploy src/jetson_deploy

# Build the workspace
colcon build --packages-select isaac_ros_visual_slam isaac_ros_nvblox humanoid_nav2 jetson_deploy
source install/setup.bash
```

### 3. Install Isaac ROS GEMs
```bash
# Install Isaac ROS cuVSLAM
sudo apt install ros-jazzy-isaac-ros-cuvslam

# Install other required GEMs
sudo apt install ros-jazzy-isaac-ros-stereo-disparity
sudo apt install ros-jazzy-isaac-ros-apriltag
sudo apt install ros-jazzy-isaac-ros-peoplenet
sudo apt install ros-jazzy-isaac-ros-pose-estimation
sudo apt install ros-jazzy-isaac-ros-visual-slams
```

### 4. Configure Jetson Orin for Deployment
```bash
# Flash JetPack 6.1
# Disable GUI for performance
sudo systemctl set-default multi-user.target
sudo systemctl disable gdm3

# Set max power mode
sudo /usr/sbin/nvpmodel -m 0

# Install TensorRT optimization tools
sudo apt install tensorrt
```

## Running the System

### 1. Launch on Development Machine (Isaac Sim)
```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash
source ~/isaac_ws/install/setup.bash

# Launch Isaac Sim with the robot in Apartment-01
ros2 launch isaac_ros_visual_slam apartment_sim.launch.py
```

### 2. Launch Perception Stack
```bash
# Launch the complete perception stack
ros2 launch isaac_ros_visual_slam perception_pipeline.launch.py

# Verify 60 Hz performance
ros2 topic hz /visual_slam/tracking/pose_graph/pose
```

### 3. Launch Mapping System
```bash
# Launch Nvblox mapping
ros2 launch isaac_ros_nvblox mapping_pipeline.launch.py

# Verify map accuracy within 5cm
ros2 service call /nvblox_node/get_map_accuracy nvblox_msgs/srv/GetMapAccuracy
```

### 4. Launch Navigation System
```bash
# Launch legged navigation
ros2 launch humanoid_nav2 navigation.launch.py

# Set navigation goal
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{
  pose: {
    header: {frame_id: 'map'},
    pose: {
      position: {x: 5.0, y: -2.0, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    }
  }
}"
```

## Deployment to Jetson Orin

### 1. Build Docker Images
```bash
cd ~/isaac_ws/src/jetson_deploy/docker

# Build perception stack image
docker build -t isaac-perception:latest -f Dockerfile.perception .

# Build mapping image
docker build -t nvblox-mapping:latest -f Dockerfile.mapping .

# Build navigation image
docker build -t humanoid-navigation:latest -f Dockerfile.navigation .
```

### 2. Optimize with TensorRT
```bash
cd ~/isaac_ws/src/jetson_deploy/tensorrt

# Run optimization scripts for each Isaac ROS GEM
python3 optimize_cuvslam.py
python3 optimize_peoplenet.py
python3 optimize_pose_estimation.py
```

### 3. Deploy to Jetson
```bash
# Copy Docker images to Jetson
docker save -o isaac-perception.tar isaac-perception:latest
scp isaac-perception.tar jetson@<jetson-ip>:~/
docker load -i isaac-perception.tar

# Run the complete system on Jetson
docker-compose -f ~/isaac_ws/src/jetson_deploy/docker/docker-compose.yml up
```

## Quality Validation

### 1. Performance Tests
```bash
# Verify perception stack runs at 60 Hz
ros2 topic hz /camera/rgb/image_raw

# Check mapping accuracy
ros2 service call /nvblox_node/compare_with_ground_truth nvblox_msgs/srv/CompareMapsWithGroundTruth

# Test navigation success rate
ros2 run navigation_tests apartment_navigation_test.py
```

### 2. Hardware-in-the-Loop Tests
```bash
# Run tests on actual Jetson hardware
# Monitor for thermal throttling
watch -n 1 'cat /sys/devices/virtual/thermal/thermal_zone*/temp'

# Verify no throttling during 7B VLA inference
sudo tegrastats --interval 1000
```

### 3. Sim-to-Real Validation
```bash
# Place AprilTag board in environment
# Compare measurements between sim and real
ros2 run sim_real_comparison april_tag_validation.py
```

## Troubleshooting

### Common Issues

1. **Perception stack not reaching 60 Hz**:
   - Check Jetson power mode: `sudo /usr/sbin/nvpmodel -q`
   - Verify TensorRT optimization is applied
   - Reduce perception pipeline complexity if needed

2. **Thermal throttling occurs**:
   - Ensure proper cooling solution is installed
   - Check thermal paste application
   - Verify no other processes consuming GPU

3. **Navigation fails in unknown environments**:
   - Verify Nvblox mapping is working correctly
   - Check sensor calibration
   - Ensure sufficient visual features for SLAM

### Monitoring Commands
```bash
# Monitor perception performance
ros2 run isaac_ros_visual_slam perception_monitor.py

# Monitor mapping quality
ros2 run isaac_ros_nvblox mapping_quality_monitor.py

# Monitor navigation status
ros2 run humanoid_nav2 navigation_monitor.py
```

## Next Steps

1. **Customize for your specific robot** by adapting the URDF and sensor configurations
2. **Extend domain randomization** for your specific application scenarios
3. **Optimize perception models** for your specific environment conditions
4. **Implement additional safety checks** as needed for your deployment