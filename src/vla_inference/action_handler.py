"""
VLA Action Request Handler
Handles VLA inference action requests in a ROS 2 action server format
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import get_system_logger
from ros_interfaces.message_converters import VLAActionRequest, RobotSkill, SkillChain
from .inference_service import VLAInferenceService, VLAInferenceResponse


@dataclass
class VLAActionGoal:
    """
    Data class for VLA action goal
    """
    image_data: Optional[bytes] = None
    text_instruction: str = ""
    context: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    goal_id: str = ""


@dataclass
class VLAActionResult:
    """
    Data class for VLA action result
    """
    success: bool = False
    robot_skill: Optional[RobotSkill] = None
    skill_chain: Optional[SkillChain] = None
    confidence: float = 0.0
    error_message: Optional[str] = None
    execution_plan: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class VLAActionFeedback:
    """
    Data class for VLA action feedback
    """
    status: str = ""  # e.g., "processing", "generating_skill", "completed"
    progress: float = 0.0  # 0.0 to 1.0
    current_step: str = ""
    confidence: float = 0.0


class VLAActionHandler:
    """
    Handler for VLA inference actions following ROS 2 action server patterns
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the VLA action handler

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("VLAActionHandler")
        self.config = self._load_config(config_path)

        # Initialize VLA inference service
        self.inference_service = VLAInferenceService(config_path)

        # Action state
        self.is_active = False
        self.current_goal = None
        self.current_feedback_callback: Optional[Callable[[VLAActionFeedback], None]] = None
        self.current_result_callback: Optional[Callable[[VLAActionResult], None]] = None

        # Thread management
        self.action_thread: Optional[threading.Thread] = None
        self.action_lock = threading.Lock()

        self.logger.info("VLA Action Handler initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the VLA action handler
        """
        default_config = {
            "enable_feedback": True,
            "feedback_interval": 0.5,  # seconds
            "max_action_time": 60.0,   # seconds
            "default_timeout": 30.0    # seconds
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

    def handle_goal(self,
                   goal: VLAActionGoal,
                   feedback_callback: Callable[[VLAActionFeedback], None],
                   result_callback: Callable[[VLAActionResult], None]) -> bool:
        """
        Handle a new VLA action goal

        Args:
            goal: The VLA action goal to handle
            feedback_callback: Callback for sending feedback
            result_callback: Callback for sending result

        Returns:
            True if goal accepted, False otherwise
        """
        with self.action_lock:
            if self.is_active:
                self.logger.warning("VLA action handler is already processing a goal")
                return False

            self.current_goal = goal
            self.current_feedback_callback = feedback_callback
            self.current_result_callback = result_callback
            self.is_active = True

            # Start action processing in a separate thread
            self.action_thread = threading.Thread(
                target=self._execute_action,
                args=(goal,),
                daemon=True
            )
            self.action_thread.start()

            self.logger.info(f"Started processing VLA action goal: {goal.goal_id}")
            return True

    def _execute_action(self, goal: VLAActionGoal):
        """
        Execute the VLA action in a separate thread

        Args:
            goal: The VLA action goal to execute
        """
        start_time = time.time()
        feedback_interval = self.config.get("feedback_interval", 0.5)
        last_feedback_time = start_time

        try:
            # Send initial feedback
            if self.config.get("enable_feedback", True) and self.current_feedback_callback:
                initial_feedback = VLAActionFeedback(
                    status="processing",
                    progress=0.0,
                    current_step="Initializing VLA inference",
                    confidence=0.0
                )
                self.current_feedback_callback(initial_feedback)

            # Create inference request
            from .inference_service import VLAInferenceRequest
            inference_request = VLAInferenceRequest(
                image_data=goal.image_data,
                text_instruction=goal.text_instruction,
                context=goal.context,
                timeout=goal.timeout
            )

            # Send feedback during processing
            if self.config.get("enable_feedback", True) and self.current_feedback_callback:
                processing_feedback = VLAActionFeedback(
                    status="generating_skill",
                    progress=0.3,
                    current_step="Running VLA model inference",
                    confidence=0.0
                )
                self.current_feedback_callback(processing_feedback)

            # Execute inference
            response = self.inference_service.execute(inference_request)

            # Send final feedback
            if self.config.get("enable_feedback", True) and self.current_feedback_callback:
                final_feedback = VLAActionFeedback(
                    status="completed",
                    progress=1.0,
                    current_step="Inference completed",
                    confidence=response.confidence
                )
                self.current_feedback_callback(final_feedback)

            # Create result
            result = VLAActionResult(
                success=response.success,
                robot_skill=response.robot_skill,
                confidence=response.confidence,
                error_message=response.error_message,
                execution_plan=response.execution_plan,
                processing_time=response.processing_time
            )

            # If successful, potentially create a skill chain
            if response.success and response.robot_skill:
                skill_chain = self._create_skill_chain_from_skill(response.robot_skill, goal.text_instruction)
                result.skill_chain = skill_chain

            # Send result
            if self.current_result_callback:
                self.current_result_callback(result)

        except Exception as e:
            self.logger.error(f"Error executing VLA action: {e}")
            result = VLAActionResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
            if self.current_result_callback:
                self.current_result_callback(result)
        finally:
            # Clean up
            with self.action_lock:
                self.is_active = False
                self.current_goal = None
                self.current_feedback_callback = None
                self.current_result_callback = None

    def _create_skill_chain_from_skill(self, skill: RobotSkill, original_instruction: str) -> Optional[SkillChain]:
        """
        Create a skill chain from a single robot skill, potentially expanding complex instructions

        Args:
            skill: The primary robot skill
            original_instruction: The original text instruction

        Returns:
            SkillChain object or None if not applicable
        """
        # For simple skills, return a single-skill chain
        if skill.skill_type in ["grasp", "navigate", "place", "find"]:
            return SkillChain(
                skills=[skill],
                execution_order=[0],
                dependencies={},
                context={"original_instruction": original_instruction}
            )

        # For complex instructions that might involve multiple steps,
        # we would parse the original instruction to create a chain
        # For now, this is a simplified implementation
        text_lower = original_instruction.lower()

        # Example: "bring the red cup to the kitchen sink" -> navigate -> grasp -> navigate -> place
        skill_chain = []
        if "bring" in text_lower or "take" in text_lower:
            # Extract object and destination
            import re
            object_match = re.search(r"(?:the|a|an)\s+(\w+(?:\s+\w+)?)\s+(?:to|from|at)", text_lower)
            location_match = re.search(r"(?:to|at|in)\s+(?:the\s+)?(\w+(?:\s+\w+)?)", text_lower)

            if object_match and location_match:
                target_object = object_match.group(1)
                target_location = location_match.group(1)

                # Create navigation skill to object
                navigate_to_obj = RobotSkill(
                    skill_type="navigate",
                    parameters={"target_location": target_object},
                    confidence=0.8,
                    execution_time=time.time(),
                    status="pending"
                )
                skill_chain.append(navigate_to_obj)

                # Create grasp skill
                grasp_skill = RobotSkill(
                    skill_type="grasp",
                    parameters={"target_object": target_object},
                    confidence=0.85,
                    execution_time=time.time(),
                    status="pending"
                )
                skill_chain.append(grasp_skill)

                # Create navigation skill to destination
                navigate_to_dest = RobotSkill(
                    skill_type="navigate",
                    parameters={"target_location": target_location},
                    confidence=0.8,
                    execution_time=time.time(),
                    status="pending"
                )
                skill_chain.append(navigate_to_dest)

                # Create place skill
                place_skill = RobotSkill(
                    skill_type="place",
                    parameters={"target_location": target_location},
                    confidence=0.85,
                    execution_time=time.time(),
                    status="pending"
                )
                skill_chain.append(place_skill)

        if skill_chain:
            return SkillChain(
                skills=skill_chain,
                execution_order=list(range(len(skill_chain))),
                dependencies={},
                context={
                    "original_instruction": original_instruction,
                    "target_object": target_object if 'target_object' in locals() else "",
                    "target_location": target_location if 'target_location' in locals() else ""
                }
            )

        return None

    def cancel_current_goal(self) -> bool:
        """
        Cancel the currently executing goal

        Returns:
            True if cancellation was successful, False otherwise
        """
        with self.action_lock:
            if not self.is_active:
                return False

            self.logger.info("Cancelling current VLA action goal")
            # In a real implementation, this would involve more sophisticated cancellation
            # For now, we'll just set the active flag to False
            self.is_active = False
            return True

    def is_goal_active(self) -> bool:
        """
        Check if a goal is currently active

        Returns:
            True if a goal is active, False otherwise
        """
        with self.action_lock:
            return self.is_active

    def get_current_status(self) -> str:
        """
        Get the current status of the action handler

        Returns:
            Current status string
        """
        if self.is_active:
            return "executing"
        else:
            return "idle"

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the action handler
        """
        return {
            "is_active": self.is_active,
            "inference_service_metrics": self.inference_service.get_performance_metrics(),
            "config": self.config
        }


class VLAActionServer:
    """
    VLA Action Server that manages the action interface
    This follows ROS 2 action server patterns
    """

    def __init__(self, node_name: str = "vla_action_server", config_path: Optional[str] = None):
        """
        Initialize the VLA action server

        Args:
            node_name: Name of the action server node
            config_path: Path to configuration file
        """
        self.node_name = node_name
        self.logger = get_system_logger(f"VLAActionServer_{node_name}")

        # Initialize action handler
        self.action_handler = VLAActionHandler(config_path)

        # Server state
        self.is_running = False
        self.goal_queue = []
        self.feedback_callbacks = {}
        self.result_callbacks = {}

        self.logger.info(f"VLA Action Server '{node_name}' initialized")

    def start(self):
        """
        Start the VLA action server
        """
        self.is_running = True
        self.logger.info(f"VLA Action Server '{self.node_name}' started")

    def stop(self):
        """
        Stop the VLA action server
        """
        self.is_running = False
        # Cancel any active goals
        if self.action_handler.is_goal_active():
            self.action_handler.cancel_current_goal()
        self.logger.info(f"VLA Action Server '{self.node_name}' stopped")

    def send_goal(self,
                  goal: VLAActionGoal,
                  feedback_callback: Optional[Callable[[VLAActionFeedback], None]] = None,
                  result_callback: Optional[Callable[[VLAActionResult], None]] = None) -> bool:
        """
        Send a goal to the VLA action server

        Args:
            goal: The VLA action goal to send
            feedback_callback: Callback for receiving feedback
            result_callback: Callback for receiving result

        Returns:
            True if goal accepted, False otherwise
        """
        if not self.is_running:
            self.logger.warning("VLA Action Server is not running")
            return False

        # Use provided callbacks or create default ones
        def default_feedback(feedback: VLAActionFeedback):
            self.logger.info(f"Feedback: {feedback.status} ({feedback.progress:.1%})")

        def default_result(result: VLAActionResult):
            status = "SUCCESS" if result.success else "FAILED"
            self.logger.info(f"Result: {status} (confidence: {result.confidence:.2f})")

        fb_callback = feedback_callback or default_feedback
        res_callback = result_callback or default_result

        return self.action_handler.handle_goal(goal, fb_callback, res_callback)

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
            "handler_status": self.action_handler.get_current_status(),
            "active_goal": self.action_handler.is_goal_active(),
            "metrics": self.action_handler.get_performance_metrics()
        }


# Example usage and testing
if __name__ == "__main__":
    import numpy as np

    # Create logger
    logger = get_system_logger("VLAActionHandlerTest")

    print("Testing VLA Action Handler...")

    # Create VLA action handler
    handler = VLAActionHandler()

    # Create a test goal
    test_goal = VLAActionGoal(
        image_data=None,  # Will use mock image
        text_instruction="pick up the red cup on the table",
        context={
            "robot_state": "idle",
            "environment_state": "kitchen",
            "available_objects": ["red cup", "blue plate", "green bottle"]
        },
        goal_id="test_goal_001"
    )

    # Define feedback and result callbacks
    def feedback_callback(feedback: VLAActionFeedback):
        print(f"  Feedback: {feedback.status} ({feedback.progress:.1%}) - {feedback.current_step}")

    def result_callback(result: VLAActionResult):
        print(f"  Result: Success={result.success}")
        print(f"    Confidence: {result.confidence:.3f}")
        print(f"    Processing time: {result.processing_time:.3f}s")
        if result.robot_skill:
            print(f"    Skill type: {result.robot_skill.skill_type}")
            print(f"    Skill params: {result.robot_skill.parameters}")
        if result.skill_chain:
            print(f"    Skill chain: {len(result.skill_chain.skills)} skills")
        if result.error_message:
            print(f"    Error: {result.error_message}")

    print("\n1. Testing simple grasp command...")
    success = handler.handle_goal(test_goal, feedback_callback, result_callback)
    print(f"  Goal accepted: {success}")

    # Wait for completion
    import time as time_module
    start_wait = time_module.time()
    while handler.is_goal_active() and (time_module.time() - start_wait) < 10:
        time_module.sleep(0.1)

    print("\n2. Testing navigation command...")
    nav_goal = VLAActionGoal(
        image_data=None,
        text_instruction="go to the kitchen sink",
        context={"robot_state": "idle", "environment_state": "kitchen"},
        goal_id="test_goal_002"
    )

    success = handler.handle_goal(nav_goal, feedback_callback, result_callback)
    print(f"  Goal accepted: {success}")

    # Wait for completion
    start_wait = time_module.time()
    while handler.is_goal_active() and (time_module.time() - start_wait) < 10:
        time_module.sleep(0.1)

    print("\n3. Testing complex command (bring object)...")
    complex_goal = VLAActionGoal(
        image_data=None,
        text_instruction="bring the red cup to the kitchen sink",
        context={"robot_state": "idle", "environment_state": "kitchen"},
        goal_id="test_goal_003"
    )

    success = handler.handle_goal(complex_goal, feedback_callback, result_callback)
    print(f"  Goal accepted: {success}")

    # Wait for completion
    start_wait = time_module.time()
    while handler.is_goal_active() and (time_module.time() - start_wait) < 15:
        time_module.sleep(0.1)

    print("\n4. Testing action server...")
    server = VLAActionServer("test_vla_server")
    server.start()

    # Send a goal through the server
    server_goal = VLAActionGoal(
        image_data=None,
        text_instruction="place the bottle on the counter",
        context={"robot_state": "holding_object", "environment_state": "kitchen"},
        goal_id="server_goal_001"
    )

    server.send_goal(server_goal)

    # Wait for completion
    start_wait = time_module.time()
    while handler.is_goal_active() and (time_module.time() - start_wait) < 10:
        time_module.sleep(0.1)

    # Get server status
    status = server.get_server_status()
    print(f"\n5. Server status: {status['handler_status']}")
    print(f"   Active goal: {status['active_goal']}")

    # Stop server
    server.stop()

    print("\nVLA Action Handler test completed.")