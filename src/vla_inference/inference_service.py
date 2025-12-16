"""
VLA Inference Service Implementation
Following contracts/vla_inference_action.yaml specification
"""

import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import get_system_logger
from ros_interfaces.message_converters import MessageConverter, VLAActionRequest, RobotSkill
from .vla_model import VLAModel


@dataclass
class VLAInferenceRequest:
    """
    Data class for VLA inference service request
    Following the contract specification
    """
    image_data: Optional[bytes] = None  # Serialized image data
    text_instruction: Optional[str] = None  # Natural language instruction
    context: Optional[Dict[str, Any]] = None  # Additional context (robot state, environment, etc.)
    timeout: float = 30.0  # Request timeout in seconds


@dataclass
class VLAInferenceResponse:
    """
    Data class for VLA inference service response
    Following the contract specification
    """
    success: bool
    robot_skill: Optional[RobotSkill] = None
    confidence: float = 0.0
    error_message: Optional[str] = None
    execution_plan: Optional[str] = None  # High-level execution plan
    processing_time: float = 0.0  # Time taken for inference


class VLAInferenceService:
    """
    VLA Inference Service following the contract specification in contracts/vla_inference_action.yaml
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the VLA inference service

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("VLAInferenceService")
        self.config = self._load_config(config_path)

        # Initialize VLA model
        self.vla_model = VLAModel(
            engine_path=self.config.get("engine_path"),
            model_path=self.config.get("model_path"),
            precision=self.config.get("precision", "float16"),
            max_batch_size=self.config.get("max_batch_size", 1)
        )

        # Message converter for ROS interface
        self.converter = MessageConverter()

        # Service state
        self.is_running = False
        self.service_lock = threading.Lock()
        self.request_count = 0
        self.total_inference_time = 0.0

        # Default context
        self.default_context = {
            "robot_state": "idle",
            "environment_state": "unknown",
            "available_objects": [],
            "robot_capabilities": ["grasp", "navigate", "place", "speak"],
            "execution_mode": "autonomous"  # shadow, shared, or autonomous
        }

        self.logger.info("VLA Inference Service initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the VLA inference service
        """
        default_config = {
            "engine_path": None,
            "model_path": None,
            "precision": "float16",
            "max_batch_size": 1,
            "image_size": [224, 224],
            "enable_caching": True,
            "min_confidence_threshold": 0.5,
            "max_inference_time": 10.0  # Maximum time allowed for inference
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

    def execute(self, request: VLAInferenceRequest) -> VLAInferenceResponse:
        """
        Execute VLA inference based on the request
        Following the contract in contracts/vla_inference_action.yaml

        Args:
            request: VLA inference request containing image, text, and context

        Returns:
            VLA inference response with robot skill prediction
        """
        with self.service_lock:
            start_time = time.time()
            self.request_count += 1

            try:
                # Validate request
                validation_result = self._validate_request(request)
                if not validation_result[0]:
                    return VLAInferenceResponse(
                        success=False,
                        error_message=validation_result[1],
                        processing_time=time.time() - start_time
                    )

                # Check timeout
                if time.time() - start_time > request.timeout:
                    return VLAInferenceResponse(
                        success=False,
                        error_message="Request timeout exceeded",
                        processing_time=time.time() - start_time
                    )

                # Get context (use request context or default)
                context = request.context or self.default_context

                # Deserialize image data (in a real implementation, this would convert bytes to numpy array)
                # For now, we'll create a mock image
                import numpy as np
                image_data = self._deserialize_image(request.image_data)
                if image_data is None:
                    # Create a mock image for testing
                    image_data = np.random.random((224, 224, 3)).astype(np.float32)

                # Create internal VLA action request
                internal_request = VLAActionRequest(
                    action_type="inference",
                    image_data=image_data,
                    text_instruction=request.text_instruction,
                    context=context
                )

                # Process with VLA model
                robot_skill = self.vla_model.process_vla_request(internal_request)

                if robot_skill is None:
                    return VLAInferenceResponse(
                        success=False,
                        error_message="VLA model failed to generate skill",
                        processing_time=time.time() - start_time
                    )

                # Check confidence threshold
                if robot_skill.confidence < self.config.get("min_confidence_threshold", 0.5):
                    return VLAInferenceResponse(
                        success=False,
                        error_message=f"Skill confidence ({robot_skill.confidence}) below threshold",
                        processing_time=time.time() - start_time
                    )

                # Generate execution plan
                execution_plan = self._generate_execution_plan(robot_skill, request.text_instruction)

                # Update performance metrics
                inference_time = time.time() - start_time
                self.total_inference_time += inference_time

                self.logger.info(f"VLA inference completed in {inference_time:.3f}s: {request.text_instruction}")

                return VLAInferenceResponse(
                    success=True,
                    robot_skill=robot_skill,
                    confidence=robot_skill.confidence,
                    execution_plan=execution_plan,
                    processing_time=inference_time
                )

            except Exception as e:
                processing_time = time.time() - start_time
                self.logger.error(f"Error in VLA inference execution after {processing_time:.3f}s: {e}")
                return VLAInferenceResponse(
                    success=False,
                    error_message=str(e),
                    processing_time=processing_time
                )

    def _validate_request(self, request: VLAInferenceRequest) -> tuple[bool, str]:
        """
        Validate the VLA inference request according to the contract

        Args:
            request: The request to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if we have either image data or text instruction (or both)
        if not request.image_data and not request.text_instruction:
            return False, "Either image_data or text_instruction must be provided"

        # Check text instruction length
        if request.text_instruction and len(request.text_instruction.strip()) == 0:
            return False, "Text instruction cannot be empty"

        # Check text instruction length (reasonable limit)
        if request.text_instruction and len(request.text_instruction) > 500:
            return False, "Text instruction is too long (max 500 characters)"

        # Check timeout is reasonable
        if request.timeout <= 0 or request.timeout > 60.0:  # Max 60 seconds
            return False, "Timeout must be between 0 and 60 seconds"

        return True, ""

    def _deserialize_image(self, image_data: Optional[bytes]) -> Optional[Any]:
        """
        Deserialize image data from bytes to numpy array
        In a real implementation, this would handle various image formats

        Args:
            image_data: Serialized image data

        Returns:
            Deserialized image as numpy array or None if failed
        """
        if image_data is None:
            return None

        try:
            # In a real implementation, this would decode the image bytes
            # For now, return None to indicate this needs real implementation
            # The VLA model will create a mock image in this case
            return None
        except Exception as e:
            self.logger.warning(f"Failed to deserialize image: {e}")
            return None

    def _generate_execution_plan(self, robot_skill: RobotSkill, text_instruction: str) -> str:
        """
        Generate a high-level execution plan based on the predicted robot skill

        Args:
            robot_skill: The predicted robot skill
            text_instruction: The original text instruction

        Returns:
            High-level execution plan as string
        """
        skill_type = robot_skill.skill_type
        params = robot_skill.parameters

        if skill_type == "grasp":
            target_obj = params.get("target_object", "unknown object")
            return f"Approach and grasp the {target_obj} using visual servoing"
        elif skill_type == "navigate":
            target_loc = params.get("target_location", "unknown location")
            return f"Navigate to {target_loc} using path planning and obstacle avoidance"
        elif skill_type == "place":
            target_loc = params.get("target_location", "unknown location")
            return f"Place object at {target_loc} with appropriate placement strategy"
        elif skill_type == "find":
            target_obj = params.get("target_object", "unknown object")
            return f"Search for {target_obj} in the environment using visual detection"
        else:
            return f"Execute {skill_type} action with parameters: {params}"

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the inference service
        """
        avg_inference_time = (
            self.total_inference_time / self.request_count
            if self.request_count > 0
            else 0.0
        )

        return {
            "request_count": self.request_count,
            "total_inference_time": self.total_inference_time,
            "avg_inference_time": avg_inference_time,
            "model_info": self.vla_model.get_model_info(),
            "config": self.config
        }

    def process_image_text_instruction(self,
                                     image_data: bytes,
                                     text_instruction: str,
                                     context: Optional[Dict[str, Any]] = None) -> VLAInferenceResponse:
        """
        Convenience method to process an image and text instruction directly

        Args:
            image_data: Serialized image data
            text_instruction: Natural language instruction
            context: Additional context

        Returns:
            VLA inference response
        """
        request = VLAInferenceRequest(
            image_data=image_data,
            text_instruction=text_instruction,
            context=context
        )
        return self.execute(request)

    def process_text_only(self,
                         text_instruction: str,
                         context: Optional[Dict[str, Any]] = None) -> VLAInferenceResponse:
        """
        Convenience method to process a text instruction without image data

        Args:
            text_instruction: Natural language instruction
            context: Additional context

        Returns:
            VLA inference response
        """
        request = VLAInferenceRequest(
            text_instruction=text_instruction,
            context=context
        )
        return self.execute(request)


class VLAInferenceServiceNode:
    """
    ROS 2 node wrapper for the VLA Inference Service
    This would be used in a real ROS 2 environment
    """

    def __init__(self, node_name: str = "vla_inference_service"):
        self.node_name = node_name
        self.service = VLAInferenceService()
        self.logger = get_system_logger(f"VLAInferenceServiceNode_{node_name}")
        self.is_active = False

    def start(self):
        """Start the VLA inference service node"""
        self.is_active = True
        self.logger.info(f"VLA Inference Service Node '{self.node_name}' started")

    def stop(self):
        """Stop the VLA inference service node"""
        self.is_active = False
        self.logger.info(f"VLA Inference Service Node '{self.node_name}' stopped")

    def get_service(self) -> VLAInferenceService:
        """Get the underlying VLA inference service"""
        return self.service


# Example usage and testing
if __name__ == "__main__":
    import numpy as np

    # Create logger
    logger = get_system_logger("VLAInferenceServiceTest")

    print("Testing VLA Inference Service...")

    # Create VLA inference service
    service = VLAInferenceService()

    # Test with a mock image and text instruction
    print("\n1. Testing with mock image and text instruction...")

    # Create a mock image (in real usage, this would be actual image bytes)
    mock_image = np.random.random((224, 224, 3)).astype(np.float32)
    # In a real system, image would be serialized, but for testing we'll pass None
    # and let the model create a mock image

    text_instruction = "pick up the red cup on the table"

    # Create request
    request = VLAInferenceRequest(
        image_data=None,  # Will use mock image
        text_instruction=text_instruction,
        context={
            "robot_state": "idle",
            "environment_state": "kitchen",
            "available_objects": ["red cup", "blue plate", "green bottle"]
        }
    )

    # Execute inference
    response = service.execute(request)

    print(f"Success: {response.success}")
    print(f"Confidence: {response.confidence:.3f}")
    print(f"Processing time: {response.processing_time:.3f}s")
    if response.robot_skill:
        print(f"Predicted skill: {response.robot_skill.skill_type}")
        print(f"Skill parameters: {response.robot_skill.parameters}")
    if response.error_message:
        print(f"Error: {response.error_message}")
    if response.execution_plan:
        print(f"Execution plan: {response.execution_plan}")

    # Test with different instruction
    print(f"\n2. Testing with different instruction: 'go to the kitchen sink'")
    response2 = service.process_text_only("go to the kitchen sink")
    print(f"Success: {response2.success}")
    if response2.robot_skill:
        print(f"Predicted skill: {response2.robot_skill.skill_type}")
        if response2.execution_plan:
            print(f"Execution plan: {response2.execution_plan}")

    # Test with place instruction
    print(f"\n3. Testing with place instruction: 'place the bottle on the counter'")
    response3 = service.process_text_only("place the bottle on the counter")
    print(f"Success: {response3.success}")
    if response3.robot_skill:
        print(f"Predicted skill: {response3.robot_skill.skill_type}")
        if response3.execution_plan:
            print(f"Execution plan: {response3.execution_plan}")

    # Get performance metrics
    print(f"\n4. Performance metrics:")
    metrics = service.get_performance_metrics()
    print(f"  Requests processed: {metrics['request_count']}")
    print(f"  Average inference time: {metrics['avg_inference_time']:.3f}s")

    print("\nVLA Inference Service test completed.")