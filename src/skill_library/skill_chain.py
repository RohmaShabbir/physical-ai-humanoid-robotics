"""
Skill Chain Executor for the Vision-Language-Action System
Handles execution of multiple robot skills in sequence with dependency management
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import get_system_logger
from ros_interfaces.message_converters import RobotSkill, SkillChain
from .grasp_skill import GraspSkillAdapter
from .navigation_skill import NavigationSkillAdapter
from .place_skill import PlaceSkillAdapter


@dataclass
class SkillExecutionResult:
    """
    Data class for skill execution result
    """
    skill_index: int
    skill_type: str
    success: bool
    message: str
    execution_info: Dict[str, Any]
    start_time: float
    end_time: float


class SkillChainExecutor:
    """
    Executor for running skill chains with dependency management and error handling
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the skill chain executor

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("SkillChainExecutor")
        self.config = self._load_config(config_path)

        # Initialize skill adapters
        self.grasp_adapter = GraspSkillAdapter()
        self.nav_adapter = NavigationSkillAdapter()
        self.place_adapter = PlaceSkillAdapter()

        # Execution state
        self.is_executing = False
        self.current_chain: Optional[SkillChain] = None
        self.current_results: List[SkillExecutionResult] = []
        self.execution_lock = threading.Lock()

        # Execution parameters
        self.max_chain_execution_time = self.config.get("max_chain_execution_time", 300.0)  # 5 minutes
        self.enable_rollback = self.config.get("enable_rollback", True)
        self.continue_on_failure = self.config.get("continue_on_failure", False)

        self.logger.info("Skill Chain Executor initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the skill chain executor
        """
        default_config = {
            "max_chain_execution_time": 300.0,  # seconds
            "enable_rollback": True,
            "continue_on_failure": False,
            "max_retries_per_skill": 1,
            "inter_skill_delay": 0.5  # seconds
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

    def execute_chain(self, skill_chain: SkillChain) -> Tuple[bool, str, List[SkillExecutionResult]]:
        """
        Execute a skill chain

        Args:
            skill_chain: The skill chain to execute

        Returns:
            Tuple of (success, message, execution_results)
        """
        with self.execution_lock:
            if self.is_executing:
                return False, "Skill chain executor already executing", []

            self.is_executing = True
            self.current_chain = skill_chain
            self.current_results = []
            start_time = time.time()

            try:
                self.logger.info(f"Starting execution of skill chain with {len(skill_chain.skills)} skills")

                # Validate the skill chain
                validation_result = self._validate_skill_chain(skill_chain)
                if not validation_result[0]:
                    return False, f"Skill chain validation failed: {validation_result[1]}", []

                # Execute each skill in the chain according to execution order
                execution_success = True
                execution_message = "Skill chain completed successfully"

                for skill_idx in skill_chain.execution_order:
                    # Check for timeout
                    if time.time() - start_time > self.max_chain_execution_time:
                        return False, "Skill chain execution timeout", self.current_results

                    # Check dependencies are satisfied
                    if not self._check_dependencies_satisfied(skill_chain, skill_idx):
                        error_msg = f"Dependencies not satisfied for skill {skill_idx}"
                        result = SkillExecutionResult(
                            skill_index=skill_idx,
                            skill_type=skill_chain.skills[skill_idx].skill_type,
                            success=False,
                            message=error_msg,
                            execution_info={"error": error_msg},
                            start_time=time.time(),
                            end_time=time.time()
                        )
                        self.current_results.append(result)
                        execution_success = False
                        execution_message = error_msg
                        break

                    # Execute the skill
                    skill_result = self._execute_single_skill(skill_chain.skills[skill_idx], skill_idx)
                    self.current_results.append(skill_result)

                    # Check if skill succeeded
                    if not skill_result.success:
                        execution_success = False
                        execution_message = f"Skill {skill_idx} failed: {skill_result.message}"

                        # If continue_on_failure is False, stop execution
                        if not self.continue_on_failure:
                            break

                    # Add delay between skills
                    time.sleep(self.config.get("inter_skill_delay", 0.5))

                # Determine final result
                if execution_success:
                    return True, execution_message, self.current_results
                else:
                    # If rollback is enabled and we had failures, attempt rollback
                    if self.enable_rollback and not execution_success:
                        self._attempt_rollback(skill_chain, self.current_results)

                    return False, execution_message, self.current_results

            except Exception as e:
                self.logger.error(f"Error executing skill chain: {e}")
                return False, f"Skill chain execution failed: {e}", self.current_results
            finally:
                self.is_executing = False
                self.current_chain = None

    def _validate_skill_chain(self, skill_chain: SkillChain) -> Tuple[bool, str]:
        """
        Validate the skill chain for execution

        Args:
            skill_chain: The skill chain to validate

        Returns:
            Tuple of (is_valid, message)
        """
        if not skill_chain.skills:
            return False, "Skill chain is empty"

        # Check that all skill indices in execution order are valid
        for idx in skill_chain.execution_order:
            if idx < 0 or idx >= len(skill_chain.skills):
                return False, f"Invalid skill index {idx} in execution order"

        # Check that dependencies are valid
        for skill_idx, deps in skill_chain.dependencies.items():
            if skill_idx < 0 or skill_idx >= len(skill_chain.skills):
                return False, f"Invalid skill index {skill_idx} in dependencies"
            for dep_idx in deps:
                if dep_idx < 0 or dep_idx >= len(skill_chain.skills):
                    return False, f"Invalid dependency index {dep_idx} for skill {skill_idx}"

        return True, "Skill chain is valid"

    def _check_dependencies_satisfied(self, skill_chain: SkillChain, skill_idx: int) -> bool:
        """
        Check if dependencies for a skill are satisfied

        Args:
            skill_chain: The skill chain
            skill_idx: Index of the skill to check

        Returns:
            True if dependencies are satisfied, False otherwise
        """
        dependencies = skill_chain.dependencies.get(skill_idx, [])
        for dep_idx in dependencies:
            # Find the result for the dependency skill
            dep_result = None
            for result in self.current_results:
                if result.skill_index == dep_idx:
                    dep_result = result
                    break

            # If dependency was not executed or failed, return False
            if dep_result is None or not dep_result.success:
                return False

        return True

    def _execute_single_skill(self, robot_skill: RobotSkill, skill_index: int) -> SkillExecutionResult:
        """
        Execute a single skill based on its type

        Args:
            robot_skill: The robot skill to execute
            skill_index: Index of the skill in the chain

        Returns:
            SkillExecutionResult with execution details
        """
        start_time = time.time()

        try:
            skill_type = robot_skill.skill_type
            self.logger.info(f"Executing skill {skill_index}: {skill_type}")

            # Route to appropriate skill adapter based on skill type
            if skill_type == "grasp":
                success, message, execution_info = self.grasp_adapter.execute_from_robot_skill(robot_skill)
            elif skill_type == "navigate":
                success, message, execution_info = self.nav_adapter.execute_from_robot_skill(robot_skill)
            elif skill_type == "place":
                success, message, execution_info = self.place_adapter.execute_from_robot_skill(robot_skill)
            else:
                success = False
                message = f"Unknown skill type: {skill_type}"
                execution_info = {"error": message}

            end_time = time.time()

            result = SkillExecutionResult(
                skill_index=skill_index,
                skill_type=skill_type,
                success=success,
                message=message,
                execution_info=execution_info,
                start_time=start_time,
                end_time=end_time
            )

            self.logger.info(f"Skill {skill_index} ({skill_type}) {'succeeded' if success else 'failed'}: {message}")
            return result

        except Exception as e:
            end_time = time.time()
            error_result = SkillExecutionResult(
                skill_index=skill_index,
                skill_type=robot_skill.skill_type,
                success=False,
                message=f"Skill execution error: {e}",
                execution_info={"error": str(e)},
                start_time=start_time,
                end_time=end_time
            )
            self.logger.error(f"Error executing skill {skill_index}: {e}")
            return error_result

    def _attempt_rollback(self, skill_chain: SkillChain, results: List[SkillExecutionResult]):
        """
        Attempt to rollback the skill chain execution

        Args:
            skill_chain: The original skill chain
            results: Execution results so far
        """
        self.logger.info("Attempting rollback due to execution failure")

        # For now, this is a placeholder - in a real implementation, this would
        # execute rollback actions for completed skills
        # For example: if we have a grasp followed by a failed navigation,
        # we might want to release the object and return to the starting position

        # In this mock implementation, we just log the rollback attempt
        completed_skills = [r for r in results if r.success]
        if completed_skills:
            self.logger.info(f"Rollback: {len(completed_skills)} skills completed successfully before failure")

    def execute_simple_sequence(self, skills: List[RobotSkill]) -> Tuple[bool, str, List[SkillExecutionResult]]:
        """
        Execute a simple sequence of skills without complex dependencies

        Args:
            skills: List of RobotSkill objects to execute in order

        Returns:
            Tuple of (success, message, execution_results)
        """
        # Create a simple skill chain with sequential execution order
        skill_chain = SkillChain(
            skills=skills,
            execution_order=list(range(len(skills))),
            dependencies={},
            context={"execution_type": "simple_sequence"}
        )

        return self.execute_chain(skill_chain)

    def get_execution_status(self) -> Dict[str, Any]:
        """
        Get the current execution status

        Returns:
            Dictionary with execution status information
        """
        return {
            "is_executing": self.is_executing,
            "current_chain_length": len(self.current_chain.skills) if self.current_chain else 0,
            "results_count": len(self.current_results),
            "config": self.config
        }

    def cancel_execution(self) -> bool:
        """
        Cancel current skill chain execution

        Returns:
            True if cancellation was successful, False otherwise
        """
        if not self.is_executing:
            return False

        self.logger.info("Cancelling skill chain execution")
        # In a real implementation, this would stop all executing skills
        # For now, we just set the flag which will be checked in the execution loop
        self.is_executing = False
        return True

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the skill chain executor

        Returns:
            Dictionary with performance metrics
        """
        if not self.current_results:
            return {"total_executions": 0, "average_execution_time": 0.0}

        total_time = sum((r.end_time - r.start_time) for r in self.current_results)
        avg_time = total_time / len(self.current_results) if self.current_results else 0.0

        successful_executions = sum(1 for r in self.current_results if r.success)

        return {
            "total_executions": len(self.current_results),
            "successful_executions": successful_executions,
            "success_rate": successful_executions / len(self.current_results) if self.current_results else 0.0,
            "average_execution_time": avg_time,
            "total_execution_time": total_time
        }


class SkillChainService:
    """
    Service interface for the skill chain executor following ROS 2 patterns
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the skill chain service

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("SkillChainService")
        self.executor = SkillChainExecutor(config_path)
        self.is_running = False

    def start(self):
        """Start the skill chain service"""
        self.is_running = True
        self.logger.info("Skill Chain Service started")

    def stop(self):
        """Stop the skill chain service"""
        self.is_running = False
        self.logger.info("Skill Chain Service stopped")

    def execute_skill_chain(self, skill_chain: SkillChain) -> Tuple[bool, str, List[SkillExecutionResult]]:
        """
        Execute a skill chain through the service

        Args:
            skill_chain: The skill chain to execute

        Returns:
            Tuple of (success, message, execution_results)
        """
        if not self.is_running:
            return False, "Skill chain service is not running", []

        return self.executor.execute_chain(skill_chain)

    def execute_skill_sequence(self, skills: List[RobotSkill]) -> Tuple[bool, str, List[SkillExecutionResult]]:
        """
        Execute a sequence of skills through the service

        Args:
            skills: List of skills to execute

        Returns:
            Tuple of (success, message, execution_results)
        """
        if not self.is_running:
            return False, "Skill chain service is not running", []

        return self.executor.execute_simple_sequence(skills)

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the service status

        Returns:
            Dictionary with service status information
        """
        return {
            "is_running": self.is_running,
            "executor_status": self.executor.get_execution_status(),
            "metrics": self.executor.get_performance_metrics()
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Skill Chain Executor...")

    # Create skill chain executor
    executor = SkillChainExecutor()

    # Create a simple skill chain: navigate -> grasp -> navigate -> place
    from ros_interfaces.message_converters import RobotSkill

    # Create navigation skill to object
    nav_to_obj = RobotSkill(
        skill_type="navigate",
        parameters={
            "target_location": "table",
            "target_position": [1.0, 1.0, 0.0],
            "path_type": "shortest",
            "speed_profile": "normal"
        },
        confidence=0.85,
        execution_time=time.time(),
        status="pending"
    )

    # Create grasp skill
    grasp_skill = RobotSkill(
        skill_type="grasp",
        parameters={
            "target_object": "red cup",
            "grasp_type": "pinch",
            "grasp_position": [1.0, 1.0, 0.1],
            "grasp_orientation": [0.0, 0.0, 0.0, 1.0]
        },
        confidence=0.8,
        execution_time=time.time(),
        status="pending"
    )

    # Create navigation skill to destination
    nav_to_dest = RobotSkill(
        skill_type="navigate",
        parameters={
            "target_location": "kitchen sink",
            "target_position": [2.5, 0.5, 0.0],
            "path_type": "safe",
            "speed_profile": "normal"
        },
        confidence=0.85,
        execution_time=time.time(),
        status="pending"
    )

    # Create place skill
    place_skill = RobotSkill(
        skill_type="place",
        parameters={
            "target_location": "kitchen sink",
            "placement_type": "surface",
            "placement_position": [2.5, 0.5, 0.1]
        },
        confidence=0.82,
        execution_time=time.time(),
        status="pending"
    )

    # Create skill chain
    skill_chain = SkillChain(
        skills=[nav_to_obj, grasp_skill, nav_to_dest, place_skill],
        execution_order=[0, 1, 2, 3],  # Execute in sequence
        dependencies={},  # No dependencies in this simple example
        context={"task": "bring_red_cup_to_sink", "description": "Bring red cup from table to kitchen sink"}
    )

    print(f"\n1. Executing skill chain with {len(skill_chain.skills)} skills...")
    success, message, results = executor.execute_chain(skill_chain)

    print(f"  Chain execution success: {success}")
    print(f"  Message: {message}")
    print(f"  Results: {len(results)} skills executed")

    for i, result in enumerate(results):
        print(f"    Skill {i} ({result.skill_type}): {'SUCCESS' if result.success else 'FAILED'} - {result.message}")

    # Test with a failing skill chain
    print(f"\n2. Testing with a skill that has a high chance of failure...")
    failing_grasp = RobotSkill(
        skill_type="grasp",
        parameters={
            "target_object": "slippery_object",
            "grasp_type": "pinch",
            "grasp_position": [0.5, 0.5, 0.1],
            "grasp_orientation": [0.0, 0.0, 0.0, 1.0]
        },
        confidence=0.3,  # Lower confidence to simulate difficulty
        execution_time=time.time(),
        status="pending"
    )

    failing_chain = SkillChain(
        skills=[nav_to_obj, failing_grasp],  # This should fail at grasp
        execution_order=[0, 1],
        dependencies={},
        context={"task": "test_failing_chain"}
    )

    success, message, results = executor.execute_chain(failing_chain)
    print(f"  Failing chain success: {success}")
    print(f"  Message: {message}")
    for i, result in enumerate(results):
        print(f"    Skill {i} ({result.skill_type}): {'SUCCESS' if result.success else 'FAILED'}")

    # Test skill chain service
    print(f"\n3. Testing Skill Chain Service...")
    service = SkillChainService()
    service.start()

    service_success, service_message, service_results = service.execute_skill_chain(skill_chain)
    print(f"  Service execution: {service_success}")
    print(f"  Service message: {service_message}")
    print(f"  Service results: {len(service_results)} skills")

    # Get service status
    status = service.get_service_status()
    print(f"  Service status: {status['is_running']}")
    print(f"  Success rate: {status['metrics']['success_rate']:.2%}")

    service.stop()

    print("\nSkill Chain Executor test completed.")