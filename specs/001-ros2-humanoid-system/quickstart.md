# Quickstart Guide: ROS 2 Humanoid System

## Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill installed
- Python 3.10+
- Git
- colcon build tools
- Real-time kernel (optional but recommended for production)

## Installation Steps

### 1. Set up ROS 2 Environment

```bash
# Install ROS 2 Humble
sudo apt update
sudo apt install ros-humble-desktop ros-humble-cv-bridge ros-humble-tf2-tools ros-humble-rviz2 ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-gazebo-ros2-control

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Install colcon build tools
sudo apt install python3-colcon-common-extensions python3-rosdep
```

### 2. Create and Initialize the Workspace

```bash
# Create the ROS 2 workspace
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws

# Initialize rosdep
sudo rosdep init
rosdep update
```

### 3. Clone the Humanoid Packages

```bash
cd ~/physical_ai_ws/src

# Clone the humanoid packages (these would be your actual packages)
git clone [your-humanoid-description-repo]
git clone [your-humanoid-control-repo]
git clone [your-humanoid-bringup-repo]
git clone [your-humanoid-teleop-repo]
git clone [your-python-ai-bridge-repo]
```

### 4. Build the Workspace

```bash
cd ~/physical_ai_ws

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
colcon build --symlink-install

# Source the workspace
source install/setup.bash
```

### 5. Verify Installation

```bash
# Check that all packages built successfully
colcon list

# Run the basic launch file to test the system
ros2 launch humanoid_bringup demo.launch.py
```

## Running the Demo

### Launch the Humanoid System

```bash
# Source the workspace
source ~/physical_ai_ws/install/setup.bash

# Launch the full system
ros2 launch humanoid_bringup demo.launch.py
```

### View in RViz

The system will automatically launch RViz with the proper configuration to visualize the 28-DoF humanoid model.

### Test Joint States

```bash
# Monitor joint states at 500 Hz
ros2 topic echo /joint_states --field position

# Check the frequency of joint state publishing
ros2 topic hz /joint_states
```

## Next Steps

1. Explore the different packages in the `src/` directory
2. Review the URDF model in `humanoid_description/`
3. Configure ros2_control in `humanoid_control/`
4. Set up gamepad teleoperation in `humanoid_teleop/`
5. Test the Python AI bridge in `python_ai_bridge/`

## Troubleshooting

### Common Issues

- **Package not found**: Ensure you've sourced the workspace with `source install/setup.bash`
- **Permission errors**: Check that your user is in the `dialout` group for hardware access
- **Real-time issues**: Install and configure PREEMPT_RT kernel for deterministic control

### Verification Commands

```bash
# Check all running nodes
ros2 node list

# Check all active topics
ros2 topic list

# Check all available services
ros2 service list
```