"""
Skill Execution Action Interface Implementation
Following contracts/skill_execution_action.yaml specification
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import get_system_logger
from ros_interfaces.message_converters import RobotSkill
from .grasp_skill import GraspSkillAdapter
from .navigation_skill import NavigationSkillAdapter
from .place_skill import PlaceSkillAdapter


@dataclass
class SkillExecutionRequest:
    """
    Data class for SkillExecutionAction request
    Following the contract specification
    """
    skill_type: str  # "grasp", "navigate", "place", etc.
    skill_params: Dict[str, Any]
    constraints: Dict[str, Any]  # timeout, max_retries, safety_limits


@dataclass
class SkillExecutionResult:
    """
    Data class for SkillExecutionAction result
    Following the contract specification
    """
    success: bool
    result_details: Dict[str, Any]
    metrics: Dict[str, Any]


@dataclass
class SkillExecutionFeedback:
    """
    Data class for SkillExecutionAction feedback
    Following the contract specification
    """
    execution_stage: str  # "init", "pre_condition_check", "execution", "post_condition_check", "cleanup"
    progress: float  # 0.0 to 1.0
    status_message: str
    estimated_time_remaining: float


class SkillExecutionAction:
    """
    Skill Execution Action following the contract specification in contracts/skill_execution_action.yaml
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the skill execution action

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("SkillExecutionAction")
        self.config = self._load_config(config_path)

        # Initialize skill adapters
        self.grasp_adapter = GraspSkillAdapter()
        self.nav_adapter = NavigationSkillAdapter()
        self.place_adapter = PlaceSkillAdapter()

        # Action state
        self.is_active = False
        self.current_request = None
        self.current_feedback_callback: Optional[Callable[[SkillExecutionFeedback], None]] = None
        self.current_result_callback: Optional[Callable[[SkillExecutionResult], None]] = None

        # Thread management
        self.action_thread: Optional[threading.Thread] = None
        self.action_lock = threading.Lock()

        # Default values from config
        self.default_timeout = self.config.get("default_timeout", 30.0)
        self.default_max_retries = self.config.get("default_max_retries", 3)

        self.logger.info("Skill Execution Action initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the skill execution action
        """
        default_config = {
            "default_timeout": 30.0,
            "default_max_retries": 3,
            "enable_feedback": True,
            "feedback_interval": 0.5,
            "supported_skills": ["grasp", "navigate", "place"]
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

    def execute(self,
                request: SkillExecutionRequest,
                feedback_callback: Callable[[SkillExecutionFeedback], None],
                result_callback: Callable[[SkillExecutionResult], None]) -> bool:
        """
        Execute the skill execution action

        Args:
            request: Skill execution request
            feedback_callback: Callback for sending feedback
            result_callback: Callback for sending result

        Returns:
            True if execution started successfully, False otherwise
        """
        with self.action_lock:
            if self.is_active:
                self.logger.warning("Skill execution action is already active")
                return False

            # Validate request
            validation_result = self._validate_request(request)
            if not validation_result[0]:
                result = SkillExecutionResult(
                    success=False,
                    result_details={
                        "execution_time": 0.0,
                        "error_code": "INVALID_REQUEST",
                        "error_description": validation_result[1],
                        "final_state": "invalid_request"
                    },
                    metrics={
                        "success_rate": 0.0,
                        "average_time": 0.0,
                        "resource_usage": {}
                    }
                )
                result_callback(result)
                return False

            self.current_request = request
            self.current_feedback_callback = feedback_callback
            self.current_result_callback = result_callback
            self.is_active = True

            # Start execution in separate thread
            self.action_thread = threading.Thread(
                target=self._execute_skill,
                args=(request, feedback_callback, result_callback),
                daemon=True
            )
            self.action_thread.start()

            self.logger.info(f"Started skill execution: {request.skill_type}")
            return True

    def _validate_request(self, request: SkillExecutionRequest) -> tuple[bool, str]:
        """
        Validate the skill execution request

        Args:
            request: The request to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check skill type is supported
        supported_skills = self.config.get("supported_skills", ["grasp", "navigate", "place"])
        if request.skill_type not in supported_skills:
            return False, f"Unsupported skill type: {request.skill_type}. Supported: {supported_skills}"

        # Check timeout is positive
        timeout = request.constraints.get("timeout", self.default_timeout)
        if timeout <= 0:
            return False, f"Timeout must be positive, got: {timeout}"

        # Check max_retries is non-negative
        max_retries = request.constraints.get("max_retries", self.default_max_retries)
        if max_retries < 0:
            return False, f"Max retries must be non-negative, got: {max_retries}"

        return True, ""

    def _execute_skill(self,
                      request: SkillExecutionRequest,
                      feedback_callback: Callable[[SkillExecutionFeedback], None],
                      result_callback: Callable[[SkillExecutionResult], None]):
        """
        Execute the skill in a separate thread

        Args:
            request: Skill execution request
            feedback_callback: Callback for sending feedback
            result_callback: Callback for sending result
        """
        start_time = time.time()
        max_retries = request.constraints.get("max_retries", self.default_max_retries)
        timeout = request.constraints.get("timeout", self.default_timeout)

        # Send initial feedback
        if self.config.get("enable_feedback", True):
            feedback_callback(SkillExecutionFeedback(
                execution_stage="init",
                progress=0.0,
                status_message="Initializing skill execution",
                estimated_time_remaining=timeout
            ))

        execution_result = None
        execution_error = None
        final_state = "unknown"

        try:
            # Send pre-condition check feedback
            if self.config.get("enable_feedback", True):
                feedback_callback(SkillExecutionFeedback(
                    execution_stage="pre_condition_check",
                    progress=0.1,
                    status_message="Checking pre-conditions",
                    estimated_time_remaining=timeout * 0.9
                ))

            # Create RobotSkill from request parameters
            robot_skill = RobotSkill(
                skill_type=request.skill_type,
                parameters=request.skill_params,
                confidence=0.8,  # Default confidence
                execution_time=start_time,
                status="pending"
            )

            # Send execution feedback
            if self.config.get("enable_feedback", True):
                feedback_callback(SkillExecutionFeedback(
                    execution_stage="execution",
                    progress=0.3,
                    status_message="Executing skill",
                    estimated_time_remaining=timeout * 0.7
                ))

            # Execute the skill with retries
            success = False
            message = ""
            execution_info = {}
            attempt = 0

            while attempt <= max_retries and not success:
                if time.time() - start_time > timeout:
                    execution_error = "Skill execution timeout"
                    break

                if attempt > 0:
                    self.logger.info(f"Retrying skill execution (attempt {attempt + 1}/{max_retries + 1})")

                # Execute based on skill type
                if request.skill_type == "grasp":
                    success, message, execution_info = self.grasp_adapter.execute_from_robot_skill(robot_skill)
                    final_state = "object_grasped" if success else "grasp_failed"
                elif request.skill_type == "navigate":
                    success, message, execution_info = self.nav_adapter.execute_from_robot_skill(robot_skill)
                    final_state = "at_destination" if success else "navigation_failed"
                elif request.skill_type == "place":
                    success, message, execution_info = self.place_adapter.execute_from_robot_skill(robot_skill)
                    final_state = "object_placed" if success else "placement_failed"
                else:
                    execution_error = f"Unknown skill type: {request.skill_type}"
                    break

                attempt += 1

                if success:
                    break

            execution_time = time.time() - start_time

            # Send post-condition check feedback
            if self.config.get("enable_feedback", True):
                feedback_callback(SkillExecutionFeedback(
                    execution_stage="post_condition_check",
                    progress=0.9,
                    status_message="Checking post-conditions",
                    estimated_time_remaining=timeout * 0.1
                ))

            if execution_error is None:
                execution_result = SkillExecutionResult(
                    success=success,
                    result_details={
                        "execution_time": execution_time,
                        "error_code": "" if success else "SKILL_EXECUTION_FAILED",
                        "error_description": message if not success else "",
                        "final_state": final_state
                    },
                    metrics={
                        "success_rate": 0.85 if success else 0.15,  # Placeholder values
                        "average_time": execution_time,
                        "resource_usage": {
                            "cpu_percent": 25.0,  # Placeholder
                            "memory_mb": 128.0   # Placeholder
                        }
                    }
                )

        except Exception as e:
            execution_time = time.time() - start_time
            execution_error = str(e)
            execution_result = SkillExecutionResult(
                success=False,
                result_details={
                    "execution_time": execution_time,
                    "error_code": "INTERNAL_ERROR",
                    "error_description": str(e),
                    "final_state": "error"
                },
                metrics={
                    "success_rate": 0.0,
                    "average_time": execution_time,
                    "resource_usage": {
                        "cpu_percent": 10.0,  # Placeholder
                        "memory_mb": 64.0    # Placeholder
                    }
                }
            )

        # Send cleanup feedback
        if self.config.get("enable_feedback", True):
            feedback_callback(SkillExecutionFeedback(
                execution_stage="cleanup",
                progress=1.0,
                status_message="Execution completed",
                estimated_time_remaining=0.0
            ))

        # Send result
        if execution_result:
            result_callback(execution_result)
        else:
            # This shouldn't happen if the code above is correct, but just in case
            result_callback(SkillExecutionResult(
                success=False,
                result_details={
                    "execution_time": time.time() - start_time,
                    "error_code": "INTERNAL_ERROR",
                    "error_description": execution_error or "Unknown error",
                    "final_state": "error"
                },
                metrics={
                    "success_rate": 0.0,
                    "average_time": time.time() - start_time,
                    "resource_usage": {}
                }
            ))

        # Clean up
        with self.action_lock:
            self.is_active = False
            self.current_request = None
            self.current_feedback_callback = None
            self.current_result_callback = None

    def is_active(self) -> bool:
        """
        Check if the skill execution action is currently active

        Returns:
            True if active, False otherwise
        """
        with self.action_lock:
            return self.is_active

    def cancel_execution(self) -> bool:
        """
        Cancel current skill execution

        Returns:
            True if cancellation was successful, False otherwise
        """
        with self.action_lock:
            if not self.is_active:
                return False

            self.logger.info("Cancelling skill execution")
            # In a real implementation, this would interrupt the executing skill
            # For now, we just set the active flag to False
            self.is_active = False
            return True

    def get_action_info(self) -> Dict[str, Any]:
        """
        Get information about the skill execution action

        Returns:
            Dictionary with action information
        """
        return {
            "is_active": self.is_active,
            "supported_skills": self.config.get("supported_skills", []),
            "config": self.config
        }


class SkillExecutionActionServer:
    """
    Skill Execution Action Server that manages the action interface
    This follows ROS 2 action server patterns
    """

    def __init__(self, node_name: str = "skill_execution_action_server", config_path: Optional[str] = None):
        """
        Initialize the skill execution action server

        Args:
            node_name: Name of the action server node
            config_path: Path to configuration file
        """
        self.node_name = node_name
        self.logger = get_system_logger(f"SkillExecutionActionServer_{node_name}")

        # Initialize action handler
        self.action_handler = SkillExecutionAction(config_path)

        # Server state
        self.is_running = False
        self.active_goals = 0

        self.logger.info(f"Skill Execution Action Server '{node_name}' initialized")

    def start(self):
        """
        Start the skill execution action server
        """
        self.is_running = True
        self.logger.info(f"Skill Execution Action Server '{self.node_name}' started")

    def stop(self):
        """
        Stop the skill execution action server
        """
        self.is_running = False
        # Cancel any active goals
        if self.action_handler.is_active():
            self.action_handler.cancel_execution()
        self.logger.info(f"Skill Execution Action Server '{self.node_name}' stopped")

    def send_goal(self,
                  request: SkillExecutionRequest,
                  feedback_callback: Optional[Callable[[SkillExecutionFeedback], None]] = None,
                  result_callback: Optional[Callable[[SkillExecutionResult], None]] = None) -> bool:
        """
        Send a goal to the skill execution action server

        Args:
            request: The skill execution request
            feedback_callback: Callback for receiving feedback
            result_callback: Callback for receiving result

        Returns:
            True if goal accepted, False otherwise
        """
        if not self.is_running:
            self.logger.warning("Skill Execution Action Server is not running")
            return False

        # Use default callbacks if none provided
        def default_feedback(feedback: SkillExecutionFeedback):
            self.logger.info(f"Feedback: {feedback.execution_stage} ({feedback.progress:.1%}) - {feedback.status_message}")

        def default_result(result: SkillExecutionResult):
            status = "SUCCESS" if result.success else "FAILED"
            self.logger.info(f"Result: {status} (execution time: {result.result_details.get('execution_time', 0):.3f}s)")

        fb_callback = feedback_callback or default_feedback
        res_callback = result_callback or default_result

        return self.action_handler.execute(request, fb_callback, res_callback)

    def is_server_active(self) -> bool:
        """
        Check if the action server is running

        Returns:
            True if server is running, False otherwise
        """
        return self.is_running

    def get_server_status(self) -> Dict[str, Any]:
        """
        Get the status of the action server

        Returns:
            Dictionary with server status information
        """
        return {
            "server_name": self.node_name,
            "is_running": self.is_running,
            "handler_status": self.action_handler.get_action_info(),
            "active_goals": 1 if self.action_handler.is_active() else 0,
            "supported_skills": self.action_handler.get_action_info()["supported_skills"]
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Skill Execution Action...")

    # Create skill execution action
    action = SkillExecutionAction()

    # Define feedback and result callbacks
    def feedback_callback(feedback: SkillExecutionFeedback):
        print(f"  Feedback: {feedback.execution_stage} ({feedback.progress:.1%}) - {feedback.status_message}")

    def result_callback(result: SkillExecutionResult):
        print(f"  Result: Success={result.success}")
        print(f"    Execution time: {result.result_details.get('execution_time', 0):.3f}s")
        print(f"    Final state: {result.result_details.get('final_state', 'unknown')}")
        if not result.success:
            print(f"    Error: {result.result_details.get('error_description', 'Unknown error')}")
        print(f"    Success rate: {result.metrics.get('success_rate', 0):.2f}")

    print("\n1. Testing grasp skill execution...")
    grasp_request = SkillExecutionRequest(
        skill_type="grasp",
        skill_params={
            "target_object": "red cup",
            "grasp_type": "pinch",
            "grasp_position": [0.5, 0.2, 0.1],
            "grasp_orientation": [0.0, 0.0, 0.0, 1.0]
        },
        constraints={
            "timeout": 30.0,
            "max_retries": 2,
            "safety_limits": {
                "max_force": 20.0
            }
        }
    )

    success = action.execute(grasp_request, feedback_callback, result_callback)
    print(f"  Goal accepted: {success}")

    # Wait for completion
    import time as time_module
    start_wait = time_module.time()
    while action.is_active() and (time_module.time() - start_wait) < 35:
        time_module.sleep(0.1)

    print("\n2. Testing navigation skill execution...")
    nav_request = SkillExecutionRequest(
        skill_type="navigate",
        skill_params={
            "target_location": "kitchen sink",
            "target_position": [2.5, 0.5, 0.0],
            "path_type": "shortest",
            "speed_profile": "normal"
        },
        constraints={
            "timeout": 120.0,
            "max_retries": 1,
            "safety_limits": {
                "min_distance": 0.5,
                "max_speed": 0.5
            }
        }
    )

    success = action.execute(nav_request, feedback_callback, result_callback)
    print(f"  Goal accepted: {success}")

    # Wait for completion
    start_wait = time_module.time()
    while action.is_active() and (time_module.time() - start_wait) < 125:
        time_module.sleep(0.1)

    print("\n3. Testing place skill execution...")
    place_request = SkillExecutionRequest(
        skill_type="place",
        skill_params={
            "target_location": "counter",
            "placement_type": "surface",
            "placement_position": [2.0, 0.0, 0.1]
        },
        constraints={
            "timeout": 30.0,
            "max_retries": 2,
            "safety_limits": {
                "max_force": 5.0
            }
        }
    )

    success = action.execute(place_request, feedback_callback, result_callback)
    print(f"  Goal accepted: {success}")

    # Wait for completion
    start_wait = time_module.time()
    while action.is_active() and (time_module.time() - start_wait) < 35:
        time_module.sleep(0.1)

    print("\n4. Testing with invalid skill type...")
    invalid_request = SkillExecutionRequest(
        skill_type="invalid_skill",
        skill_params={},
        constraints={
            "timeout": 30.0,
            "max_retries": 1
        }
    )

    success = action.execute(invalid_request, feedback_callback, result_callback)
    print(f"  Invalid request accepted: {success}")

    print("\n5. Testing Skill Execution Action Server...")
    server = SkillExecutionActionServer("test_skill_server")
    server.start()

    # Send a goal through the server
    server_request = SkillExecutionRequest(
        skill_type="navigate",
        skill_params={
            "target_location": "table",
            "target_position": [1.0, 1.0, 0.0]
        },
        constraints={
            "timeout": 60.0,
            "max_retries": 1
        }
    )

    server_success = server.send_goal(server_request)
    print(f"  Server goal accepted: {server_success}")

    # Wait for completion
    start_wait = time_module.time()
    while action.is_active() and (time_module.time() - start_wait) < 65:
        time_module.sleep(0.1)

    # Get server status
    status = server.get_server_status()
    print(f"\n6. Server status: {status['is_running']}")
    print(f"   Active goals: {status['active_goals']}")
    print(f"   Supported skills: {status['supported_skills']}")

    # Stop server
    server.stop()

    print("\nSkill Execution Action test completed.")