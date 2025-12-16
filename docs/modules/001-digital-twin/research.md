# Research Summary: Digital Twin (Gazebo & Unity)

## 1. Gazebo Harmonic 2025 and NVIDIA Isaac Sim 2024.2 Compatibility

### Decision: Use NVIDIA Isaac Sim container for Gazebo Harmonic installation
### Rationale: The containerized approach provides a stable, tested environment that avoids dependency conflicts common with native installation
### Alternatives considered:
- Native Ubuntu package installation: Often leads to dependency conflicts and broken installations
- Source compilation: Time-consuming and error-prone, especially for complex simulation environments
- ROS 2 ecosystem packages: May not include latest features and fixes

## 2. RealSense D435i Noise Model Implementation

### Decision: Implement realistic depth camera noise based on actual Realsense D435i characteristics
### Rationale: Accurate sensor simulation is critical for sim-to-real transfer learning
### Alternatives considered:
- Simple Gaussian noise: Insufficient to capture real sensor behavior
- Generic depth camera models: Don't reflect specific characteristics of target hardware
- No noise model: Would not prepare algorithms for real-world sensor limitations

### Key parameters for Realsense D435i simulation:
- Radial distortion: k1, k2, k3 coefficients
- Tangential distortion: p1, p2 coefficients
- Depth noise: Range-dependent variance following manufacturer specifications
- Angular noise: Direction-dependent errors based on viewing angle

## 3. URDF to SDF Conversion Process

### Decision: Use automated conversion with manual refinement
### Rationale: The `gz sdf -p` tool provides a solid foundation, but requires manual adjustments for accurate physics simulation
### Alternatives considered:
- Manual SDF creation from scratch: Extremely time-consuming and error-prone
- Direct SDF authoring: Not practical for complex humanoid robots with 28+ degrees of freedom
- Third-party conversion tools: Limited support for advanced SDF features

### Key refinements needed after conversion:
- Collision bitmasks for selective collision detection
- Surface friction parameters for realistic contact
- Self-collision handling for hands and complex linkages
- Accurate inertial properties recalculation
- Joint limits, effort, and velocity constraints

## 4. Physics Simulation Parameters

### Decision: Use Gazebo Harmonic with realistic actuator modeling
### Rationale: Proper actuator simulation is essential for accurate robot behavior and control
### Alternatives considered:
- Simplified physics models: Insufficient for accurate sim-to-real transfer
- Custom physics plugins: Would require significant development time
- External physics engines: Would complicate integration with ROS 2

### Key actuator parameters:
- Effort limits: Maximum torque/force constraints
- Velocity limits: Maximum joint speed constraints
- Damping coefficients: Energy dissipation modeling
- Friction parameters: Static and dynamic friction modeling

## 5. Unity Integration Approach

### Decision: Use USD export from Isaac Sim to Unity with ROS-TCP-Connector
### Rationale: USD provides standardized format for complex scene and robot data transfer
### Alternatives considered:
- Direct Unity robot import: Limited support for complex kinematics and physics
- Custom export formats: Would require significant custom development
- Unity Robotics Hub only: Limited to Unity native tools without Isaac Sim benefits

### Unity connection options evaluated:
- ROS-TCP-Connector: Most mature and widely used option
- rosbridge: Web-based but potentially slower for real-time simulation
- Unity ROS2 package: Newer, less tested in production environments

## 6. Sensor Simulation Requirements

### Decision: Implement comprehensive sensor stack with realistic noise models
### Rationale: Complete sensor simulation is necessary for robust algorithm development
### Alternatives considered:
- Minimal sensor set: Would limit testing capabilities
- Idealized sensors without noise: Would not prepare for real-world conditions
- Third-party sensor plugins: Limited customization options

### Sensors to simulate:
- Depth camera (RealSense D435i profile): Including noise, distortion, and dropout
- IMU: With realistic bias and noise based on Allan variance models
- LIDAR: With dropout and angular noise characteristics
- Camera: With lens distortion and rolling shutter effects

## 7. Apartment Environment Generation

### Decision: Procedural generation with standardized object sets
### Rationale: Procedural generation allows for varied, realistic environments while maintaining consistency
### Alternatives considered:
- Manual environment creation: Time-consuming and not scalable
- Pre-built environments only: Limited variation for testing
- Fully random generation: Could create unrealistic or untestable scenarios

### Key features for apartment environment:
- Multiple room types: Kitchen, living room, bedroom, bathroom
- Household objects: Cups, chairs, tables, doors with realistic properties
- Semantic labeling: For VLA training and perception tasks
- Navigation mesh: For path planning and movement validation