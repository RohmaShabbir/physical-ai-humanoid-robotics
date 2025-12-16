"""
Voice Command Service Interface Implementation
Following contracts/voice_command_service.yaml specification
"""

import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..utils.logging_utils import get_system_logger
from ..ros_interfaces.message_converters import MessageConverter
from .voice_processor import VoiceProcessor


@dataclass
class VoiceCommandRequest:
    """Data class for voice command service request"""
    audio_data: Optional[bytes] = None
    text_command: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class VoiceCommandResponse:
    """Data class for voice command service response"""
    success: bool
    command_structure: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    error_message: Optional[str] = None
    llm_prompt_context: Optional[str] = None


class VoiceCommandService:
    """
    Voice Command Service following the contract specification
    """

    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_system_logger("VoiceCommandService")
        self.voice_processor = VoiceProcessor(config_path)
        self.converter = MessageConverter()
        self.is_running = False
        self.service_lock = threading.Lock()

        # Initialize with default context
        self.default_context = {
            "robot_state": "idle",
            "environment_state": "unknown",
            "available_objects": [],
            "robot_capabilities": ["grasp", "navigate", "place"]
        }

    def call_service(self, request: VoiceCommandRequest) -> VoiceCommandResponse:
        """
        Process a voice command service request and return response
        Following the contract in contracts/voice_command_service.yaml
        """
        with self.service_lock:
            start_time = time.time()

            try:
                # Validate request
                validation_result = self._validate_request(request)
                if not validation_result[0]:
                    return VoiceCommandResponse(
                        success=False,
                        error_message=validation_result[1]
                    )

                # Get context (use request context or default)
                context = request.context or self.default_context

                # Get the command text (from audio or direct text)
                command_text = request.text_command

                # If audio data is provided, convert it to text
                if request.audio_data and not command_text:
                    # In a real implementation, we would use the speech recognition module
                    # to convert audio_data to text. For now, we'll simulate this.
                    command_text = self._convert_audio_to_text(request.audio_data)
                    if not command_text:
                        return VoiceCommandResponse(
                            success=False,
                            error_message="Could not convert audio to text"
                        )

                if not command_text:
                    return VoiceCommandResponse(
                        success=False,
                        error_message="No command text provided"
                    )

                # Process the voice command using the voice processor
                goals = self.voice_processor.process_voice_command(command_text)

                # Convert goals to the required command structure format
                command_structure = self._goals_to_command_structure(goals, command_text, context)

                # Calculate confidence based on the processing
                confidence = self._calculate_response_confidence(command_structure)

                # Create LLM prompt context
                llm_prompt_context = self._create_llm_prompt_context(command_text, context)

                processing_time = time.time() - start_time
                self.logger.info(f"Voice command processed in {processing_time:.3f}s: {command_text}")

                return VoiceCommandResponse(
                    success=True,
                    command_structure=command_structure,
                    confidence=confidence,
                    llm_prompt_context=llm_prompt_context
                )

            except Exception as e:
                processing_time = time.time() - start_time
                self.logger.error(f"Error processing voice command after {processing_time:.3f}s: {e}")
                return VoiceCommandResponse(
                    success=False,
                    error_message=str(e)
                )

    def _validate_request(self, request: VoiceCommandRequest) -> tuple[bool, str]:
        """
        Validate the service request according to the contract
        """
        # Either audio_data OR text_command must be provided (but not necessarily both)
        if not request.audio_data and not request.text_command:
            return False, "Either audio_data or text_command must be provided"

        # If both are provided, that's acceptable too
        return True, ""

    def _convert_audio_to_text(self, audio_data: bytes) -> Optional[str]:
        """
        Convert audio data to text (placeholder implementation)
        In a real system, this would interface with the speech recognition module
        """
        # This is a placeholder - in a real implementation, we would:
        # 1. Use the audio preprocessing from audio_input.py
        # 2. Interface with the speech recognition module
        # 3. Convert the audio data to text

        # For now, return None to indicate this needs real implementation
        self.logger.warning("Audio-to-text conversion not fully implemented in this mock")
        return None

    def _goals_to_command_structure(self, goals: list, command_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the generated goals to the required command structure format
        """
        action_sequence = []
        for goal in goals:
            action_sequence.append({
                "action_type": goal.action_type.replace("Action", "").lower(),  # e.g., "GraspAction" -> "grasp"
                "parameters": goal.parameters,
                "priority": goal.priority
            })

        # Identify target object and location from the command
        target_object = self._extract_target_object(command_text, context.get("available_objects", []))
        target_location = self._extract_target_location(command_text)

        return {
            "action_sequence": action_sequence,
            "target_object": target_object,
            "target_location": target_location,
            "execution_plan": f"Execute sequence: {[action['action_type'] for action in action_sequence]}"
        }

    def _extract_target_object(self, command_text: str, available_objects: list) -> str:
        """
        Extract target object from command text
        """
        command_lower = command_text.lower()

        # Look for known objects in the command
        for obj in available_objects:
            if obj.lower() in command_lower:
                return obj

        # If no known object, try to extract any noun that sounds like an object
        # This is a simple heuristic - in practice, you'd use NLP techniques
        import re
        words = re.findall(r'\b\w+\b', command_lower)
        potential_objects = ["cup", "bottle", "book", "box", "plate", "phone", "keys", "wallet"]

        for word in words:
            if word in potential_objects:
                return word

        return ""

    def _extract_target_location(self, command_text: str) -> str:
        """
        Extract target location from command text
        """
        command_lower = command_text.lower()

        # Look for location keywords
        location_keywords = [
            "kitchen", "living room", "bedroom", "bathroom", "office",
            "table", "counter", "couch", "chair", "desk", "shelf", "cabinet",
            "sink", "refrigerator", "microwave", "oven", "stove", "bed",
            "door", "window", "hallway", "dining room"
        ]

        for location in location_keywords:
            if location in command_lower:
                return location

        return ""

    def _calculate_response_confidence(self, command_structure: Dict[str, Any]) -> float:
        """
        Calculate confidence in the response
        """
        if not command_structure or not command_structure.get("action_sequence"):
            return 0.0

        # Base confidence on the number of actions and completeness of structure
        action_count = len(command_structure["action_sequence"])
        has_target_object = bool(command_structure.get("target_object"))
        has_target_location = bool(command_structure.get("target_location"))

        base_confidence = 0.5
        if action_count > 0:
            base_confidence += 0.2
        if has_target_object:
            base_confidence += 0.15
        if has_target_location:
            base_confidence += 0.15

        return min(base_confidence, 1.0)  # Cap at 1.0

    def _create_llm_prompt_context(self, command_text: str, context: Dict[str, Any]) -> str:
        """
        Create context for LLM processing
        """
        prompt_context = f"User command: '{command_text}'\n"
        prompt_context += f"Robot state: {context.get('robot_state', 'unknown')}\n"
        prompt_context += f"Environment: {context.get('environment_state', 'unknown')}\n"

        if context.get('available_objects'):
            prompt_context += f"Available objects: {', '.join(context['available_objects'])}\n"

        return prompt_context

    def process_voice_command_text(self, text_command: str, context: Optional[Dict[str, Any]] = None) -> VoiceCommandResponse:
        """
        Convenience method to process a text command directly
        """
        request = VoiceCommandRequest(
            text_command=text_command,
            context=context
        )
        return self.call_service(request)

    def process_voice_command_audio(self, audio_data: bytes, context: Optional[Dict[str, Any]] = None) -> VoiceCommandResponse:
        """
        Convenience method to process an audio command
        """
        request = VoiceCommandRequest(
            audio_data=audio_data,
            context=context
        )
        return self.call_service(request)


class VoiceCommandServiceNode:
    """
    ROS 2 node wrapper for the Voice Command Service
    This would be used in a real ROS 2 environment
    """

    def __init__(self, node_name: str = "voice_command_service"):
        self.node_name = node_name
        self.service = VoiceCommandService()
        self.logger = get_system_logger(f"VoiceCommandServiceNode_{node_name}")
        self.is_active = False

    def start(self):
        """Start the voice command service node"""
        self.is_active = True
        self.logger.info(f"Voice Command Service Node '{self.node_name}' started")

    def stop(self):
        """Stop the voice command service node"""
        self.is_active = False
        self.logger.info(f"Voice Command Service Node '{self.node_name}' stopped")

    def get_service(self) -> VoiceCommandService:
        """Get the underlying voice command service"""
        return self.service


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from utils.logging_utils import get_system_logger

    # Create logger
    logger = get_system_logger("VoiceServiceTest")

    # Create voice command service
    service = VoiceCommandService()

    # Test with text commands
    test_commands = [
        "pick up the red cup",
        "go to the kitchen sink",
        "place the bottle on the table"
    ]

    print("Testing Voice Command Service...")
    for cmd in test_commands:
        print(f"\nProcessing command: '{cmd}'")
        response = service.process_voice_command_text(cmd)

        print(f"  Success: {response.success}")
        print(f"  Confidence: {response.confidence:.2f}")
        if response.success and response.command_structure:
            print(f"  Action sequence: {[action['action_type'] for action in response.command_structure['action_sequence']]}")
            print(f"  Target object: {response.command_structure.get('target_object', 'N/A')}")
            print(f"  Target location: {response.command_structure.get('target_location', 'N/A')}")
        if response.error_message:
            print(f"  Error: {response.error_message}")

    # Test with a more complex command
    print(f"\nTesting complex command: 'bring the blue book to the living room'")
    complex_response = service.process_voice_command_text("bring the blue book to the living room")
    print(f"  Success: {complex_response.success}")
    print(f"  Confidence: {complex_response.confidence:.2f}")
    if complex_response.success and complex_response.command_structure:
        print(f"  Command structure: {json.dumps(complex_response.command_structure, indent=2)}")

    print("\nVoice Command Service test completed.")