"""
Grasp Skill Implementation for the Vision-Language-Action System
Handles object grasping operations with various grasp types and safety checks
"""

import time
import math
from typing import Dict, Any, Optional, Tuple
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import get_system_logger
from ros_interfaces.message_converters import RobotSkill


class GraspSkill:
    """
    Grasp skill implementation for picking up objects
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the grasp skill

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("GraspSkill")
        self.config = self._load_config(config_path)

        # Grasp parameters
        self.min_grasp_width = self.config.get("min_grasp_width", 0.01)  # 1cm
        self.max_grasp_width = self.config.get("max_grasp_width", 0.10)  # 10cm
        self.approach_distance = self.config.get("approach_distance", 0.05)  # 5cm
        self.lift_distance = self.config.get("lift_distance", 0.05)  # 5cm
        self.grasp_force = self.config.get("grasp_force", 10.0)  # Newtons
        self.max_attempts = self.config.get("max_attempts", 3)

        # Current grasp state
        self.is_executing = False
        self.current_object = None
        self.last_grasp_success = False

        self.logger.info("Grasp Skill initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the grasp skill
        """
        default_config = {
            "min_grasp_width": 0.01,  # meters
            "max_grasp_width": 0.10,  # meters
            "approach_distance": 0.05,  # meters
            "lift_distance": 0.05,  # meters
            "grasp_force": 10.0,  # Newtons
            "max_attempts": 3,
            "enable_force_feedback": True,
            "enable_vision_guided": True
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
        Execute the grasp skill with given parameters

        Args:
            parameters: Dictionary containing grasp parameters
                - target_object: Name or description of object to grasp
                - grasp_type: Type of grasp ('pinch', 'power', 'lateral', 'suction')
                - grasp_position: [x, y, z] position for grasp
                - grasp_orientation: [qx, qy, qz, qw] quaternion for grasp orientation
                - grasp_width: Target grasp width (for parallel jaw grippers)
                - approach_angle: Angle to approach the object from

        Returns:
            Tuple of (success, message, execution_info)
        """
        if self.is_executing:
            return False, "Grasp skill already executing", {}

        self.is_executing = True
        start_time = time.time()

        try:
            self.logger.info(f"Executing grasp skill for object: {parameters.get('target_object', 'unknown')}")

            # Validate parameters
            validation_result = self._validate_parameters(parameters)
            if not validation_result[0]:
                return False, validation_result[1], {}

            # Extract parameters
            target_object = parameters.get('target_object', 'unknown_object')
            grasp_type = parameters.get('grasp_type', 'pinch')
            grasp_position = parameters.get('grasp_position', [0.0, 0.0, 0.0])
            grasp_orientation = parameters.get('grasp_orientation', [0.0, 0.0, 0.0, 1.0])
            grasp_width = parameters.get('grasp_width', 0.05)
            approach_angle = parameters.get('approach_angle', 45.0)

            self.current_object = target_object

            # Perform pre-grasp checks
            pre_grasp_result = self._perform_pre_grasp_checks(target_object, grasp_position)
            if not pre_grasp_result[0]:
                return False, f"Pre-grasp check failed: {pre_grasp_result[1]}", {}

            # Execute grasp sequence
            grasp_result = self._execute_grasp_sequence(
                grasp_type, grasp_position, grasp_orientation,
                grasp_width, approach_angle
            )

            if grasp_result[0]:
                # Verify grasp success
                verification_result = self._verify_grasp_success(target_object)
                success = verification_result[0]
                message = f"Grasp {'succeeded' if success else 'failed'}: {verification_result[1]}"
            else:
                success = False
                message = grasp_result[1]

            execution_info = {
                "execution_time": time.time() - start_time,
                "target_object": target_object,
                "grasp_type": grasp_type,
                "grasp_position": grasp_position,
                "grasp_success": success,
                "attempts": 1  # Simplified for this example
            }

            self.last_grasp_success = success
            self.logger.info(f"Grasp execution completed: {message}")
            return success, message, execution_info

        except Exception as e:
            self.logger.error(f"Error executing grasp skill: {e}")
            execution_info = {
                "execution_time": time.time() - start_time,
                "target_object": parameters.get('target_object', 'unknown'),
                "grasp_success": False,
                "error": str(e)
            }
            return False, f"Grasp execution failed: {e}", execution_info
        finally:
            self.is_executing = False

    def _validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate grasp parameters

        Args:
            parameters: Dictionary of grasp parameters

        Returns:
            Tuple of (is_valid, message)
        """
        target_object = parameters.get('target_object')
        if not target_object:
            return False, "Target object not specified"

        grasp_position = parameters.get('grasp_position', [0.0, 0.0, 0.0])
        if len(grasp_position) != 3:
            return False, "Grasp position must be [x, y, z] coordinates"

        grasp_orientation = parameters.get('grasp_orientation', [0.0, 0.0, 0.0, 1.0])
        if len(grasp_orientation) != 4:
            return False, "Grasp orientation must be [qx, qy, qz, qw] quaternion"

        grasp_width = parameters.get('grasp_width', 0.05)
        if grasp_width < self.min_grasp_width or grasp_width > self.max_grasp_width:
            return False, f"Grasp width {grasp_width}m out of range [{self.min_grasp_width}, {self.max_grasp_width}]"

        return True, "Parameters valid"

    def _perform_pre_grasp_checks(self, target_object: str, grasp_position: list) -> Tuple[bool, str]:
        """
        Perform pre-grasp safety and feasibility checks

        Args:
            target_object: Name of the target object
            grasp_position: Grasp position [x, y, z]

        Returns:
            Tuple of (is_ok, message)
        """
        # Check if object is reachable
        if not self._is_reachable(grasp_position):
            return False, f"Object {target_object} at {grasp_position} is not reachable"

        # Check if object is graspable
        if not self._is_object_graspable(target_object):
            return False, f"Object {target_object} is not graspable"

        # Check for obstacles
        if self._check_for_obstacles(grasp_position):
            return False, f"Obstacles detected near {target_object}"

        return True, "Pre-grasp checks passed"

    def _is_reachable(self, position: list) -> bool:
        """
        Check if the position is within robot's reach

        Args:
            position: [x, y, z] position to check

        Returns:
            True if reachable, False otherwise
        """
        # Simplified reachability check
        # In a real implementation, this would check robot kinematics
        x, y, z = position
        # Assume workspace is roughly within these bounds
        return (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0 and 0.1 <= z <= 1.0)

    def _is_object_graspable(self, target_object: str) -> bool:
        """
        Check if the object is graspable based on its properties

        Args:
            target_object: Name of the target object

        Returns:
            True if graspable, False otherwise
        """
        # Simplified check based on object name
        # In a real implementation, this would use object recognition data
        non_graspable = ["floor", "wall", "ceiling", "table", "counter"]
        return target_object.lower() not in non_graspable

    def _check_for_obstacles(self, position: list) -> bool:
        """
        Check for obstacles near the grasp position

        Args:
            position: [x, y, z] position to check

        Returns:
            True if obstacles detected, False otherwise
        """
        # Simplified obstacle check
        # In a real implementation, this would use perception data
        return False  # No obstacles detected in this mock implementation

    def _execute_grasp_sequence(self, grasp_type: str, grasp_position: list,
                               grasp_orientation: list, grasp_width: float,
                               approach_angle: float) -> Tuple[bool, str]:
        """
        Execute the actual grasp sequence

        Args:
            grasp_type: Type of grasp to perform
            grasp_position: Position to grasp [x, y, z]
            grasp_orientation: Orientation [qx, qy, qz, qw]
            grasp_width: Width for the gripper
            approach_angle: Angle to approach from

        Returns:
            Tuple of (success, message)
        """
        try:
            # Simulate approach motion
            approach_success = self._simulate_approach_motion(grasp_position, approach_angle)
            if not approach_success:
                return False, "Approach motion failed"

            # Simulate grasp execution
            grasp_success = self._simulate_grasp_execution(grasp_type, grasp_width)
            if not grasp_success:
                return False, "Grasp execution failed"

            # Simulate lift motion
            lift_success = self._simulate_lift_motion()
            if not lift_success:
                return False, "Lift motion failed"

            return True, "Grasp sequence completed successfully"

        except Exception as e:
            return False, f"Grasp sequence execution failed: {e}"

    def _simulate_approach_motion(self, grasp_position: list, approach_angle: float) -> bool:
        """
        Simulate the approach motion to the grasp position

        Args:
            grasp_position: Target position [x, y, z]
            approach_angle: Approach angle in degrees

        Returns:
            True if successful, False otherwise
        """
        # In a real implementation, this would control the robot arm
        # For simulation, we'll just wait a bit and return True
        time.sleep(0.5)  # Simulate approach time
        return True

    def _simulate_grasp_execution(self, grasp_type: str, grasp_width: float) -> bool:
        """
        Simulate the actual grasp execution

        Args:
            grasp_type: Type of grasp to perform
            grasp_width: Width to set the gripper to

        Returns:
            True if successful, False otherwise
        """
        # In a real implementation, this would close the gripper
        # For simulation, we'll just wait and return True
        time.sleep(0.3)  # Simulate grasp time
        return True

    def _simulate_lift_motion(self) -> bool:
        """
        Simulate lifting the object after grasping

        Returns:
            True if successful, False otherwise
        """
        # In a real implementation, this would lift the arm
        # For simulation, we'll just wait and return True
        time.sleep(0.3)  # Simulate lift time
        return True

    def _verify_grasp_success(self, target_object: str) -> Tuple[bool, str]:
        """
        Verify that the grasp was successful

        Args:
            target_object: Name of the object that was grasped

        Returns:
            Tuple of (success, message)
        """
        # In a real implementation, this would use force sensors,
        # vision, or other feedback to verify grasp success
        # For simulation, we'll randomly return success/failure
        import random
        success = random.random() > 0.2  # 80% success rate in simulation
        if success:
            return True, f"Successfully grasped {target_object}"
        else:
            return False, f"Failed to grasp {target_object} - object slipped"

    def get_skill_info(self) -> Dict[str, Any]:
        """
        Get information about the grasp skill

        Returns:
            Dictionary with skill information
        """
        return {
            "skill_name": "grasp",
            "is_executing": self.is_executing,
            "current_object": self.current_object,
            "last_success": self.last_grasp_success,
            "config": self.config
        }

    def cancel_execution(self) -> bool:
        """
        Cancel current grasp execution

        Returns:
            True if cancellation was successful, False otherwise
        """
        if not self.is_executing:
            return False

        self.logger.info("Cancelling grasp execution")
        # In a real implementation, this would stop the robot arm
        self.is_executing = False
        return True


class GraspSkillAdapter:
    """
    Adapter to convert GraspSkill to RobotSkill format for the skill chain
    """

    def __init__(self):
        self.grasp_skill = GraspSkill()
        self.logger = get_system_logger("GraspSkillAdapter")

    def execute_from_robot_skill(self, robot_skill: RobotSkill) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute grasp skill from RobotSkill format

        Args:
            robot_skill: RobotSkill object containing grasp parameters

        Returns:
            Tuple of (success, message, execution_info)
        """
        if robot_skill.skill_type != "grasp":
            return False, f"Skill type mismatch: expected 'grasp', got '{robot_skill.skill_type}'", {}

        return self.grasp_skill.execute(robot_skill.parameters)

    def create_robot_skill(self, target_object: str, grasp_type: str = "pinch",
                          grasp_position: Optional[list] = None,
                          grasp_orientation: Optional[list] = None) -> RobotSkill:
        """
        Create a RobotSkill object for grasping

        Args:
            target_object: Name of object to grasp
            grasp_type: Type of grasp ('pinch', 'power', 'lateral', 'suction')
            grasp_position: [x, y, z] position for grasp
            grasp_orientation: [qx, qy, qz, qw] quaternion for grasp orientation

        Returns:
            RobotSkill object configured for grasping
        """
        if grasp_position is None:
            grasp_position = [0.0, 0.0, 0.0]
        if grasp_orientation is None:
            grasp_orientation = [0.0, 0.0, 0.0, 1.0]

        parameters = {
            "target_object": target_object,
            "grasp_type": grasp_type,
            "grasp_position": grasp_position,
            "grasp_orientation": grasp_orientation,
            "grasp_width": 0.05,  # Default grasp width
            "approach_angle": 45.0  # Default approach angle
        }

        return RobotSkill(
            skill_type="grasp",
            parameters=parameters,
            confidence=0.8,  # Default confidence for grasp
            execution_time=time.time(),
            status="pending"
        )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Grasp Skill...")

    # Create grasp skill
    grasp_skill = GraspSkill()

    # Test with different parameters
    test_params = [
        {
            "target_object": "red cup",
            "grasp_type": "pinch",
            "grasp_position": [0.5, 0.2, 0.1],
            "grasp_orientation": [0.0, 0.0, 0.0, 1.0],
            "grasp_width": 0.04
        },
        {
            "target_object": "book",
            "grasp_type": "power",
            "grasp_position": [0.3, -0.1, 0.15],
            "grasp_orientation": [0.0, 0.707, 0.0, 0.707],
            "grasp_width": 0.06
        }
    ]

    for i, params in enumerate(test_params):
        print(f"\nTest {i+1}: Grasping {params['target_object']}")
        success, message, info = grasp_skill.execute(params)
        print(f"  Success: {success}")
        print(f"  Message: {message}")
        print(f"  Execution time: {info.get('execution_time', 0):.3f}s")
        print(f"  Grasp success: {info.get('grasp_success', False)}")

    # Test skill adapter
    print(f"\nTesting Grasp Skill Adapter...")
    adapter = GraspSkillAdapter()

    robot_skill = adapter.create_robot_skill(
        target_object="blue bottle",
        grasp_type="pinch",
        grasp_position=[0.4, 0.0, 0.12]
    )

    print(f"Created RobotSkill: {robot_skill.skill_type}")
    print(f"Parameters: {robot_skill.parameters}")

    # Execute through adapter
    success, message, info = adapter.execute_from_robot_skill(robot_skill)
    print(f"Adapter execution - Success: {success}, Message: {message}")

    print("\nGrasp Skill test completed.")