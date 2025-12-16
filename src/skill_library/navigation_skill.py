"""
Navigation Skill Implementation for the Vision-Language-Action System
Handles robot navigation to specified locations with obstacle avoidance
"""

import time
import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import get_system_logger
from ros_interfaces.message_converters import RobotSkill


class NavigationSkill:
    """
    Navigation skill implementation for moving the robot to specified locations
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the navigation skill

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("NavigationSkill")
        self.config = self._load_config(config_path)

        # Navigation parameters
        self.linear_velocity = self.config.get("linear_velocity", 0.3)  # m/s
        self.angular_velocity = self.config.get("angular_velocity", 0.5)  # rad/s
        self.min_distance_threshold = self.config.get("min_distance_threshold", 0.1)  # meters
        self.collision_threshold = self.config.get("collision_threshold", 0.5)  # meters
        self.max_navigation_time = self.config.get("max_navigation_time", 60.0)  # seconds
        self.path_planning_algorithm = self.config.get("path_planning_algorithm", "dijkstra")

        # Current navigation state
        self.is_executing = False
        self.current_destination = None
        self.last_navigation_success = False
        self.robot_position = [0.0, 0.0, 0.0]  # x, y, theta

        self.logger.info("Navigation Skill initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the navigation skill
        """
        default_config = {
            "linear_velocity": 0.3,  # m/s
            "angular_velocity": 0.5,  # rad/s
            "min_distance_threshold": 0.1,  # meters
            "collision_threshold": 0.5,  # meters
            "max_navigation_time": 60.0,  # seconds
            "path_planning_algorithm": "dijkstra",
            "enable_obstacle_avoidance": True,
            "enable_localization": True
        }

        if config_path:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(config)
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")

        return default_config

    def execute(self, parameters: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute the navigation skill with given parameters

        Args:
            parameters: Dictionary containing navigation parameters
                - target_location: Named location or coordinates
                - target_position: [x, y, theta] target position
                - path_type: Type of path ('shortest', 'safe', 'fast')
                - speed_profile: Speed profile ('normal', 'slow', 'fast')
                - avoid_dynamic_obstacles: Whether to avoid moving obstacles

        Returns:
            Tuple of (success, message, execution_info)
        """
        if self.is_executing:
            return False, "Navigation skill already executing", {}

        self.is_executing = True
        start_time = time.time()

        try:
            # Validate parameters
            validation_result = self._validate_parameters(parameters)
            if not validation_result[0]:
                return False, validation_result[1], {}

            # Extract parameters
            target_location = parameters.get('target_location', 'unknown')
            target_position = parameters.get('target_position', [0.0, 0.0, 0.0])
            path_type = parameters.get('path_type', 'shortest')
            speed_profile = parameters.get('speed_profile', 'normal')
            avoid_dynamic_obstacles = parameters.get('avoid_dynamic_obstacles', True)

            self.current_destination = target_location

            self.logger.info(f"Executing navigation to {target_location} at {target_position}")

            # Plan path to destination
            path_result = self._plan_path(target_position, path_type)
            if not path_result[0]:
                return False, f"Path planning failed: {path_result[1]}", {}

            path = path_result[1]

            # Execute navigation along the path
            navigation_result = self._execute_navigation(
                path, speed_profile, avoid_dynamic_obstacles, start_time
            )

            success = navigation_result[0]
            message = navigation_result[1]
            execution_info = navigation_result[2]

            execution_info.update({
                "target_location": target_location,
                "target_position": target_position,
                "path_type": path_type,
                "execution_time": time.time() - start_time
            })

            self.last_navigation_success = success
            self.logger.info(f"Navigation execution completed: {message}")
            return success, message, execution_info

        except Exception as e:
            self.logger.error(f"Error executing navigation skill: {e}")
            execution_info = {
                "execution_time": time.time() - start_time,
                "target_location": parameters.get('target_location', 'unknown'),
                "navigation_success": False,
                "error": str(e)
            }
            return False, f"Navigation execution failed: {e}", execution_info
        finally:
            self.is_executing = False

    def _validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate navigation parameters

        Args:
            parameters: Dictionary of navigation parameters

        Returns:
            Tuple of (is_valid, message)
        """
        target_position = parameters.get('target_position', [0.0, 0.0, 0.0])
        if len(target_position) != 3:
            return False, "Target position must be [x, y, theta] coordinates"

        x, y, theta = target_position
        # Check if target is within reasonable bounds (adjust based on your robot's workspace)
        if abs(x) > 10.0 or abs(y) > 10.0:
            return False, f"Target position ({x}, {y}) out of bounds"

        return True, "Parameters valid"

    def _plan_path(self, target_position: List[float], path_type: str) -> Tuple[bool, str, Optional[List[List[float]]]]:
        """
        Plan a path to the target position

        Args:
            target_position: [x, y, theta] target position
            path_type: Type of path to plan

        Returns:
            Tuple of (success, message, path)
        """
        try:
            # In a real implementation, this would use a path planning algorithm
            # like A*, Dijkstra, or RRT to plan a collision-free path
            # For simulation, we'll create a simple straight-line path

            start_pos = self.robot_position
            target_x, target_y, target_theta = target_position

            # Create a simple path (in reality, this would be more complex)
            path = []
            steps = 10  # Number of steps in the path
            for i in range(steps + 1):
                t = i / steps
                x = start_pos[0] + t * (target_x - start_pos[0])
                y = start_pos[1] + t * (target_y - start_pos[1])
                theta = start_pos[2] + t * (target_theta - start_pos[2])
                path.append([x, y, theta])

            return True, "Path planned successfully", path

        except Exception as e:
            return False, f"Path planning failed: {e}", None

    def _execute_navigation(self, path: List[List[float]], speed_profile: str,
                           avoid_dynamic_obstacles: bool, start_time: float) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute the navigation along the planned path

        Args:
            path: List of [x, y, theta] waypoints
            speed_profile: Speed profile to use
            avoid_dynamic_obstacles: Whether to avoid dynamic obstacles
            start_time: Start time for the navigation

        Returns:
            Tuple of (success, message, execution_info)
        """
        try:
            # Adjust velocity based on speed profile
            velocity_scale = 1.0
            if speed_profile == 'slow':
                velocity_scale = 0.5
            elif speed_profile == 'fast':
                velocity_scale = 1.5

            current_velocity = self.linear_velocity * velocity_scale
            current_angular_velocity = self.angular_velocity * velocity_scale

            # Execute navigation for each waypoint in the path
            for i, waypoint in enumerate(path):
                # Check for timeout
                if time.time() - start_time > self.max_navigation_time:
                    return False, "Navigation timeout", {"waypoint_reached": i, "timeout": True}

                # Check for obstacles
                if avoid_dynamic_obstacles:
                    obstacle_result = self._check_for_obstacles(waypoint)
                    if obstacle_result[0]:
                        # Handle obstacle
                        obstacle_avoidance_result = self._handle_obstacle(obstacle_result[1])
                        if not obstacle_avoidance_result[0]:
                            return False, f"Obstacle avoidance failed: {obstacle_avoidance_result[1]}", {}

                # Move to waypoint (simulated)
                self._move_to_waypoint(waypoint, current_velocity, current_angular_velocity)

                # Update robot position
                self.robot_position = waypoint

                # Log progress
                progress = (i + 1) / len(path)
                self.logger.info(f"Navigation progress: {progress:.1%}")

            # Check if we're close enough to the target
            target = path[-1]
            distance_to_target = math.sqrt(
                (self.robot_position[0] - target[0])**2 +
                (self.robot_position[1] - target[1])**2
            )

            if distance_to_target <= self.min_distance_threshold:
                return True, "Navigation completed successfully", {
                    "waypoints_traversed": len(path),
                    "final_distance_error": distance_to_target,
                    "navigation_success": True
                }
            else:
                return False, f"Failed to reach target (distance error: {distance_to_target:.3f}m)", {
                    "waypoints_traversed": len(path),
                    "final_distance_error": distance_to_target,
                    "navigation_success": False
                }

        except Exception as e:
            return False, f"Navigation execution failed: {e}", {"error": str(e)}

    def _check_for_obstacles(self, waypoint: List[float]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for obstacles near the waypoint

        Args:
            waypoint: [x, y, theta] waypoint to check

        Returns:
            Tuple of (has_obstacle, obstacle_info)
        """
        # In a real implementation, this would use sensor data (LIDAR, cameras, etc.)
        # to detect obstacles in the environment
        # For simulation, we'll randomly generate obstacles
        import random
        if random.random() < 0.1:  # 10% chance of obstacle
            obstacle_info = {
                "position": [waypoint[0] + random.uniform(-0.2, 0.2),
                            waypoint[1] + random.uniform(-0.2, 0.2)],
                "size": random.uniform(0.1, 0.5),
                "type": "static"  # or "dynamic"
            }
            return True, obstacle_info

        return False, {}

    def _handle_obstacle(self, obstacle_info: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Handle detected obstacle by replanning or avoiding

        Args:
            obstacle_info: Information about the detected obstacle

        Returns:
            Tuple of (success, message)
        """
        self.logger.warning(f"Obstacle detected at {obstacle_info['position']}, handling...")

        # In a real implementation, this would implement obstacle avoidance
        # For simulation, we'll just wait and pretend we avoided the obstacle
        time.sleep(0.5)  # Simulate obstacle avoidance time

        return True, "Obstacle avoided successfully"

    def _move_to_waypoint(self, waypoint: List[float], linear_velocity: float, angular_velocity: float):
        """
        Move the robot to the specified waypoint

        Args:
            waypoint: [x, y, theta] target waypoint
            linear_velocity: Linear velocity to use
            angular_velocity: Angular velocity to use
        """
        # In a real implementation, this would send commands to the robot's base controller
        # For simulation, we'll just sleep for a time proportional to the distance
        distance = math.sqrt(
            (waypoint[0] - self.robot_position[0])**2 +
            (waypoint[1] - self.robot_position[1])**2
        )

        # Calculate time needed to reach waypoint (simplified)
        move_time = distance / linear_velocity if linear_velocity > 0 else 0.1
        time.sleep(min(move_time, 1.0))  # Cap the sleep time for simulation

    def get_current_position(self) -> List[float]:
        """
        Get the current robot position

        Returns:
            Current position [x, y, theta]
        """
        return self.robot_position.copy()

    def set_current_position(self, position: List[float]):
        """
        Set the current robot position (for simulation/testing)

        Args:
            position: New position [x, y, theta]
        """
        if len(position) == 3:
            self.robot_position = position.copy()

    def get_skill_info(self) -> Dict[str, Any]:
        """
        Get information about the navigation skill

        Returns:
            Dictionary with skill information
        """
        return {
            "skill_name": "navigate",
            "is_executing": self.is_executing,
            "current_destination": self.current_destination,
            "last_success": self.last_navigation_success,
            "current_position": self.robot_position,
            "config": self.config
        }

    def cancel_execution(self) -> bool:
        """
        Cancel current navigation execution

        Returns:
            True if cancellation was successful, False otherwise
        """
        if not self.is_executing:
            return False

        self.logger.info("Cancelling navigation execution")
        # In a real implementation, this would stop the robot
        self.is_executing = False
        return True


class NavigationSkillAdapter:
    """
    Adapter to convert NavigationSkill to RobotSkill format for the skill chain
    """

    def __init__(self):
        self.navigation_skill = NavigationSkill()
        self.logger = get_system_logger("NavigationSkillAdapter")

    def execute_from_robot_skill(self, robot_skill: RobotSkill) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute navigation skill from RobotSkill format

        Args:
            robot_skill: RobotSkill object containing navigation parameters

        Returns:
            Tuple of (success, message, execution_info)
        """
        if robot_skill.skill_type != "navigate":
            return False, f"Skill type mismatch: expected 'navigate', got '{robot_skill.skill_type}'", {}

        return self.navigation_skill.execute(robot_skill.parameters)

    def create_robot_skill(self, target_location: str, target_position: Optional[List[float]] = None,
                          path_type: str = "shortest", speed_profile: str = "normal") -> RobotSkill:
        """
        Create a RobotSkill object for navigation

        Args:
            target_location: Named location to navigate to
            target_position: [x, y, theta] target position
            path_type: Type of path to take
            speed_profile: Speed profile to use

        Returns:
            RobotSkill object configured for navigation
        """
        if target_position is None:
            # Default positions for common locations (in a real system, these would come from a map)
            location_positions = {
                "kitchen": [2.0, 0.0, 0.0],
                "living room": [0.0, 2.0, 1.57],
                "bedroom": [-2.0, 0.0, 3.14],
                "bathroom": [0.0, -2.0, -1.57],
                "kitchen sink": [2.5, 0.5, 0.0],
                "table": [1.0, 1.0, 0.785]
            }
            target_position = location_positions.get(target_location.lower(), [0.0, 0.0, 0.0])

        parameters = {
            "target_location": target_location,
            "target_position": target_position,
            "path_type": path_type,
            "speed_profile": speed_profile,
            "avoid_dynamic_obstacles": True
        }

        return RobotSkill(
            skill_type="navigate",
            parameters=parameters,
            confidence=0.85,  # Default confidence for navigation
            execution_time=time.time(),
            status="pending"
        )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Navigation Skill...")

    # Create navigation skill
    nav_skill = NavigationSkill()

    # Test with different parameters
    test_params = [
        {
            "target_location": "kitchen sink",
            "target_position": [2.5, 0.5, 0.0],
            "path_type": "shortest",
            "speed_profile": "normal"
        },
        {
            "target_location": "bedroom",
            "target_position": [-2.0, 0.0, 3.14],
            "path_type": "safe",
            "speed_profile": "slow"
        }
    ]

    for i, params in enumerate(test_params):
        print(f"\nTest {i+1}: Navigating to {params['target_location']}")

        # Set initial position for testing
        nav_skill.set_current_position([0.0, 0.0, 0.0])

        success, message, info = nav_skill.execute(params)
        print(f"  Success: {success}")
        print(f"  Message: {message}")
        print(f"  Execution time: {info.get('execution_time', 0):.3f}s")
        print(f"  Final position: {nav_skill.get_current_position()}")

    # Test skill adapter
    print(f"\nTesting Navigation Skill Adapter...")
    adapter = NavigationSkillAdapter()

    robot_skill = adapter.create_robot_skill(
        target_location="kitchen",
        path_type="fast"
    )

    print(f"Created RobotSkill: {robot_skill.skill_type}")
    print(f"Parameters: {robot_skill.parameters}")

    # Execute through adapter
    success, message, info = adapter.execute_from_robot_skill(robot_skill)
    print(f"Adapter execution - Success: {success}, Message: {message}")

    print("\nNavigation Skill test completed.")