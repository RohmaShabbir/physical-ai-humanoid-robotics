# Quickstart Guide: Vision-Language-Action System

**Feature**: 001-vla-complete-brain
**Date**: 2025-12-12
**Status**: Complete

## Overview

This guide provides instructions for setting up and running the Vision-Language-Action (VLA) system on NVIDIA Jetson hardware. The system enables voice command processing and execution for robotic tasks, with a primary focus on the "Hey Robot, Tidy the Apartment" capstone project.

## Prerequisites

- NVIDIA Jetson AGX Orin or Orin Nano with 8GB+ RAM
- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill installed
- Python 3.11
- NVIDIA JetPack SDK with CUDA 11.4+
- At least 16GB free storage space for models

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/vla-complete-brain.git
cd vla-complete-brain
```

### 2. Set up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

### 3. Install Dependencies

```bash
# Install ROS 2 dependencies
sudo apt update
sudo apt install python3-rosdep python3-colcon-common-extensions

# Install additional dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers faster-whisper tensorrt_llm
pip install opencv-python openvino
```

### 4. Download Models

```bash
# Download Open-VLA-7B model (or your chosen VLA model)
python scripts/download_vla_model.py --model openvla-7b

# Download Faster-Whisper large-v3 model
python -c "from faster_whisper import WhisperModel; model = WhisperModel('large-v3')"
```

### 5. Build TensorRT Engine (Optional but Recommended)

```bash
python scripts/build_tensorrt_engine.py --model openvla-7b --precision float16
```

## Configuration

### 1. Environment Setup

```bash
# Set up ROS 2 environment
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### 2. Configuration Files

Update the configuration files in `config/`:

- `vla_config.yaml`: VLA model settings and paths
- `voice_config.yaml`: Speech recognition parameters
- `safety_config.yaml`: Safety system parameters
- `capstone_config.yaml`: Capstone project settings

## Usage

### 1. Launch the Complete System

```bash
# For simulation with Isaac Sim
ros2 launch capstone_bringup capstone.launch.py simulation:=true

# For real hardware
ros2 launch capstone_bringup capstone.launch.py simulation:=false
```

### 2. Run Individual Components

```bash
# Launch only VLA inference
ros2 launch capstone_bringup vla_system.launch.py

# Run voice pipeline standalone
python src/voice_pipeline/voice_processor.py

# Test skill library
python src/skill_library/skill_chain.py --test
```

### 3. Send Voice Commands

Once the system is running, you can issue commands like:

- "The red cup is on the table. Bring it to the kitchen sink."
- "Pick up the book and place it on the shelf."
- "Move the bottle from the counter to the cabinet."

## Testing

### 1. Unit Tests

```bash
# Run all unit tests
python -m pytest tests/unit/

# Run specific test
python -m pytest tests/unit/test_vla_inference.py
```

### 2. Integration Tests

```bash
# Run integration tests
python -m pytest tests/integration/

# Run voice-to-action test
python -m pytest tests/integration/test_voice_to_action.py
```

### 3. Performance Tests

```bash
# Run performance benchmarks
python scripts/performance_test.py --duration 60
```

## Capstone Project: "Hey Robot, Tidy the Apartment"

### Setup

1. Ensure the robot starts on the charging dock
2. Verify the target object (e.g., red cup) is visible to the robot
3. Ensure the destination (kitchen sink) is accessible

### Execution

1. Start the complete system:
   ```bash
   ros2 launch capstone_bringup capstone.launch.py
   ```

2. Issue the command: "The red cup is on the table. Bring it to the kitchen sink."

3. Monitor the execution through the system logs and safety monitors

### Success Criteria

- Task completed in ≤3 minutes in simulation
- Task completed in ≤5 minutes on real hardware
- Robot successfully perceives, plans, grasps, navigates, and places the object

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure sufficient VRAM and correct model paths in config files
2. **Speech Recognition Not Responding**: Check audio input device and permissions
3. **ROS 2 Connection Issues**: Verify ROS_DOMAIN_ID and network configuration
4. **Performance Degradation**: Monitor VRAM usage and consider model quantization

### Safety System

- If emergency stop activates, reset with: `ros2 service call /emergency_stop_reset std_srvs/Trigger`
- Check safety logs in `/logs/safety/`
- Verify heartbeat intervals in `safety_config.yaml`

## Development

### Adding New Skills

1. Create a new skill file in `src/skill_library/`
2. Implement the skill following the RobotSkill interface
3. Register the skill in the skill chain manager
4. Add unit tests in `tests/unit/`

### Model Optimization

1. Use TensorRT for inference optimization
2. Apply quantization techniques for memory efficiency
3. Profile performance with `scripts/profile_performance.py`
4. Adjust batch sizes in configuration files