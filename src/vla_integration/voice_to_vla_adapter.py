"""
Voice to VLA Adapter for the Vision-Language-Action System
Integrates voice pipeline with VLA inference service for end-to-end processing
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import get_system_logger
from voice_pipeline.voice_service import VoiceCommandService
from vla_inference.inference_service import VLAInferenceService
from skill_library.skill_chain import SkillChainExecutor
from ros_interfaces.message_converters import RobotSkill, SkillChain


@dataclass
class VoiceToVLARequest:
    """
    Data class for voice to VLA integration request
    """
    audio_data: Optional[bytes] = None
    text_command: Optional[str] = None
    image_data: Optional[bytes] = None  # Visual input for VLA
    context: Optional[Dict[str, Any]] = None


@dataclass
class VoiceToVLAResponse:
    """
    Data class for voice to VLA integration response
    """
    success: bool
    voice_processing_result: Optional[Dict[str, Any]] = None
    vla_inference_result: Optional[Dict[str, Any]] = None
    skill_execution_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    total_processing_time: float = 0.0


class VoiceToVLAAdapter:
    """
    Adapter that integrates voice pipeline with VLA inference service
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the voice to VLA adapter

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("VoiceToVLAAdapter")
        self.config = self._load_config(config_path)

        # Initialize components
        self.voice_service = VoiceCommandService()
        self.vla_service = VLAInferenceService()
        self.skill_executor = SkillChainExecutor()

        # Integration state
        self.is_processing = False
        self.processing_lock = threading.Lock()

        # Performance tracking
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.successful_completions = 0

        self.logger.info("Voice to VLA Adapter initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the voice to VLA adapter
        """
        default_config = {
            "enable_vla_fallback": True,
            "enable_skill_chaining": True,
            "max_end_to_end_time": 10.0,  # seconds
            "enable_context_awareness": True,
            "default_image_timeout": 5.0  # seconds to wait for image
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

    def process_voice_command(self, request: VoiceToVLARequest) -> VoiceToVLAResponse:
        """
        Process a voice command through the full pipeline: voice -> VLA -> skills

        Args:
            request: Voice to VLA integration request

        Returns:
            Voice to VLA integration response
        """
        with self.processing_lock:
            if self.is_processing:
                return VoiceToVLAResponse(
                    success=False,
                    error_message="Adapter already processing a request",
                    total_processing_time=0.0
                )

            self.is_processing = True
            start_time = time.time()
            self.total_requests += 1

            try:
                self.logger.info(f"Processing voice command: {request.text_command or 'audio command'}")

                # Step 1: Process voice command to extract intent
                voice_result = self._process_voice_command(request)
                if not voice_result[0]:
                    return VoiceToVLAResponse(
                        success=False,
                        error_message=f"Voice processing failed: {voice_result[1]}",
                        total_processing_time=time.time() - start_time
                    )

                voice_processing_result = voice_result[1]

                # Step 2: Use VLA inference to generate robot skills
                vla_result = self._process_vla_inference(request, voice_processing_result)
                if not vla_result[0]:
                    if self.config.get("enable_vla_fallback", True):
                        # Create basic skills based on voice command if VLA fails
                        robot_skills = self._create_fallback_skills(voice_processing_result)
                    else:
                        return VoiceToVLAResponse(
                            success=False,
                            voice_processing_result=voice_processing_result,
                            error_message=f"VLA inference failed: {vla_result[1]}",
                            total_processing_time=time.time() - start_time
                        )
                else:
                    robot_skills = [vla_result[1]]  # Convert single skill to list for skill chain

                # Step 3: Execute the generated skills
                skill_result = self._execute_skills(robot_skills, voice_processing_result)
                if not skill_result[0]:
                    return VoiceToVLAResponse(
                        success=False,
                        voice_processing_result=voice_processing_result,
                        vla_inference_result=vla_result[1] if vla_result[0] else None,
                        error_message=f"Skill execution failed: {skill_result[1]}",
                        total_processing_time=time.time() - start_time
                    )

                # Success case
                total_time = time.time() - start_time
                self.total_processing_time += total_time
                self.successful_completions += 1

                response = VoiceToVLAResponse(
                    success=True,
                    voice_processing_result=voice_processing_result,
                    vla_inference_result=vla_result[1] if vla_result[0] else None,
                    skill_execution_result=skill_result[1],
                    total_processing_time=total_time
                )

                self.logger.info(f"Voice command processed successfully in {total_time:.3f}s")
                return response

            except Exception as e:
                total_time = time.time() - start_time
                self.logger.error(f"Error in voice to VLA processing: {e}")
                return VoiceToVLAResponse(
                    success=False,
                    error_message=f"Voice to VLA processing failed: {e}",
                    total_processing_time=total_time
                )
            finally:
                self.is_processing = False

    def _process_voice_command(self, request: VoiceToVLARequest) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Process the voice command using the voice service

        Args:
            request: Voice to VLA request

        Returns:
            Tuple of (success, voice_processing_result)
        """
        try:
            if request.text_command:
                # Process text command directly
                response = self.voice_service.process_voice_command_text(
                    request.text_command,
                    request.context
                )
            elif request.audio_data:
                # Process audio command
                response = self.voice_service.process_voice_command_audio(
                    request.audio_data,
                    request.context
                )
            else:
                return False, {"error": "No audio data or text command provided"}

            if not response.success:
                return False, {"error": response.error_message}

            return True, {
                "command_structure": response.command_structure,
                "confidence": response.confidence,
                "llm_prompt_context": response.llm_prompt_context,
                "original_command": request.text_command or "audio_command"
            }

        except Exception as e:
            return False, {"error": str(e)}

    def _process_vla_inference(self, request: VoiceToVLARequest, voice_result: Dict[str, Any]) -> Tuple[bool, Optional[RobotSkill]]:
        """
        Process VLA inference using the VLA service

        Args:
            request: Voice to VLA request
            voice_result: Result from voice processing

        Returns:
            Tuple of (success, robot_skill)
        """
        try:
            # Extract target object and location from voice processing
            cmd_struct = voice_result.get("command_structure", {})
            target_obj = cmd_struct.get("target_object", "")
            target_loc = cmd_struct.get("target_location", "")

            # Create a text instruction for VLA based on voice command
            text_instruction = f"{voice_result['original_command']}"
            if target_obj and target_loc:
                text_instruction += f" (object: {target_obj}, location: {target_loc})"

            # Create VLA inference request
            from vla_inference.inference_service import VLAInferenceRequest
            vla_request = VLAInferenceRequest(
                image_data=request.image_data,
                text_instruction=text_instruction,
                context=request.context or {}
            )

            # Execute VLA inference
            vla_response = self.vla_service.execute(vla_request)

            if not vla_response.success:
                return False, None

            return True, vla_response.robot_skill

        except Exception as e:
            return False, None

    def _create_fallback_skills(self, voice_processing_result: Dict[str, Any]) -> List[RobotSkill]:
        """
        Create fallback skills based on voice processing result when VLA fails

        Args:
            voice_processing_result: Result from voice processing

        Returns:
            List of RobotSkill objects
        """
        cmd_struct = voice_processing_result.get("command_structure", {})
        action_sequence = cmd_struct.get("action_sequence", [])
        target_obj = cmd_struct.get("target_object", "")
        target_loc = cmd_struct.get("target_location", "")

        skills = []

        for action in action_sequence:
            action_type = action.get("action_type", "")
            params = action.get("parameters", {})

            if action_type == "grasp":
                skill = RobotSkill(
                    skill_type="grasp",
                    parameters={
                        "target_object": target_obj or params.get("target_object", "unknown"),
                        "grasp_type": params.get("grasp_type", "pinch"),
                        "grasp_position": params.get("grasp_position", [0.5, 0.2, 0.1]),
                        "grasp_orientation": params.get("grasp_orientation", [0.0, 0.0, 0.0, 1.0])
                    },
                    confidence=0.7,  # Lower confidence for fallback
                    execution_time=time.time(),
                    status="pending"
                )
            elif action_type == "navigate":
                skill = RobotSkill(
                    skill_type="navigate",
                    parameters={
                        "target_location": target_loc or params.get("target_location", "unknown"),
                        "target_position": params.get("target_position", [1.0, 1.0, 0.0]),
                        "path_type": params.get("path_type", "shortest"),
                        "speed_profile": params.get("speed_profile", "normal")
                    },
                    confidence=0.75,
                    execution_time=time.time(),
                    status="pending"
                )
            elif action_type == "place":
                skill = RobotSkill(
                    skill_type="place",
                    parameters={
                        "target_location": target_loc or params.get("target_location", "unknown"),
                        "placement_type": params.get("placement_type", "surface"),
                        "placement_position": params.get("placement_position", [1.0, 1.0, 0.1])
                    },
                    confidence=0.72,
                    execution_time=time.time(),
                    status="pending"
                )
            else:
                # For other action types, create a generic skill
                skill = RobotSkill(
                    skill_type=action_type,
                    parameters=params,
                    confidence=0.6,
                    execution_time=time.time(),
                    status="pending"
                )

            skills.append(skill)

        return skills

    def _execute_skills(self, robot_skills: List[RobotSkill], voice_result: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Execute the generated robot skills

        Args:
            robot_skills: List of RobotSkill objects to execute
            voice_result: Result from voice processing (for context)

        Returns:
            Tuple of (success, execution_result)
        """
        try:
            if not robot_skills:
                return False, {"error": "No skills to execute"}

            # If we have multiple skills, create a skill chain
            if len(robot_skills) > 1 or self.config.get("enable_skill_chaining", True):
                skill_chain = SkillChain(
                    skills=robot_skills,
                    execution_order=list(range(len(robot_skills))),
                    dependencies={},
                    context=voice_result.get("llm_prompt_context", {})
                )

                # Execute the skill chain
                chain_success, chain_message, chain_results = self.skill_executor.execute_chain(skill_chain)

                return chain_success, {
                    "skill_chain_execution": {
                        "success": chain_success,
                        "message": chain_message,
                        "results": [result.__dict__ for result in chain_results],
                        "skill_count": len(robot_skills)
                    }
                }
            else:
                # Execute single skill
                skill = robot_skills[0]
                result = self.skill_executor._execute_single_skill(skill, 0)

                return result.success, {
                    "single_skill_execution": {
                        "success": result.success,
                        "message": result.message,
                        "skill_type": skill.skill_type,
                        "execution_time": result.end_time - result.start_time
                    }
                }

        except Exception as e:
            return False, {"error": str(e)}

    def process_text_command(self, text_command: str, context: Optional[Dict[str, Any]] = None) -> VoiceToVLAResponse:
        """
        Convenience method to process a text command directly

        Args:
            text_command: The text command to process
            context: Additional context for processing

        Returns:
            Voice to VLA integration response
        """
        request = VoiceToVLARequest(
            text_command=text_command,
            context=context
        )
        return self.process_voice_command(request)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the adapter

        Returns:
            Dictionary with performance metrics
        """
        avg_processing_time = (
            self.total_processing_time / self.successful_completions
            if self.successful_completions > 0
            else 0.0
        )

        success_rate = (
            self.successful_completions / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "successful_completions": self.successful_completions,
            "success_rate": success_rate,
            "average_processing_time": avg_processing_time,
            "total_processing_time": self.total_processing_time,
            "is_processing": self.is_processing,
            "config": self.config
        }

    def is_active(self) -> bool:
        """
        Check if the adapter is currently processing

        Returns:
            True if processing, False otherwise
        """
        return self.is_processing

    def cancel_processing(self) -> bool:
        """
        Cancel current processing if active

        Returns:
            True if cancellation was successful, False otherwise
        """
        if not self.is_processing:
            return False

        self.logger.info("Cancelling voice to VLA processing")
        # In a real implementation, this would cancel ongoing operations
        return True


class VoiceToVLAService:
    """
    Service interface for the voice to VLA integration
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the voice to VLA service

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("VoiceToVLAService")
        self.adapter = VoiceToVLAAdapter(config_path)
        self.is_running = False

    def start(self):
        """Start the voice to VLA service"""
        self.is_running = True
        self.logger.info("Voice to VLA Service started")

    def stop(self):
        """Stop the voice to VLA service"""
        self.is_running = False
        self.logger.info("Voice to VLA Service stopped")

    def process_command(self, request: VoiceToVLARequest) -> VoiceToVLAResponse:
        """
        Process a command through the voice to VLA service

        Args:
            request: Voice to VLA request

        Returns:
            Voice to VLA response
        """
        if not self.is_running:
            return VoiceToVLAResponse(
                success=False,
                error_message="Voice to VLA service is not running",
                total_processing_time=0.0
            )

        return self.adapter.process_voice_command(request)

    def process_text_command(self, text_command: str, context: Optional[Dict[str, Any]] = None) -> VoiceToVLAResponse:
        """
        Process a text command through the service

        Args:
            text_command: The text command to process
            context: Additional context for processing

        Returns:
            Voice to VLA response
        """
        if not self.is_running:
            return VoiceToVLAResponse(
                success=False,
                error_message="Voice to VLA service is not running",
                total_processing_time=0.0
            )

        return self.adapter.process_text_command(text_command, context)

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the service status

        Returns:
            Dictionary with service status information
        """
        return {
            "is_running": self.is_running,
            "adapter_status": self.adapter.get_performance_metrics(),
            "components": {
                "voice_service": self.adapter.voice_service.get_performance_metrics() if hasattr(self.adapter.voice_service, 'get_performance_metrics') else {},
                "vla_service": self.adapter.vla_service.get_performance_metrics() if hasattr(self.adapter.vla_service, 'get_performance_metrics') else {},
                "skill_executor": self.adapter.skill_executor.get_performance_metrics() if hasattr(self.adapter.skill_executor, 'get_performance_metrics') else {}
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Voice to VLA Adapter...")

    # Create voice to VLA adapter
    adapter = VoiceToVLAAdapter()

    # Test with various commands
    test_commands = [
        "pick up the red cup",
        "go to the kitchen sink",
        "place the bottle on the table",
        "bring the blue book to the living room"
    ]

    for i, cmd in enumerate(test_commands):
        print(f"\nTest {i+1}: Processing command '{cmd}'")

        response = adapter.process_text_command(cmd)

        print(f"  Success: {response.success}")
        print(f"  Total processing time: {response.total_processing_time:.3f}s")

        if response.voice_processing_result:
            cmd_struct = response.voice_processing_result.get("command_structure", {})
            print(f"  Command structure: {cmd_struct.get('action_sequence', [])}")
            print(f"  Target object: {cmd_struct.get('target_object', 'N/A')}")
            print(f"  Target location: {cmd_struct.get('target_location', 'N/A')}")

        if response.error_message:
            print(f"  Error: {response.error_message}")

        if response.skill_execution_result:
            print(f"  Skill execution: {response.skill_execution_result}")

    # Test with the service wrapper
    print(f"\nTesting Voice to VLA Service...")
    service = VoiceToVLAService()
    service.start()

    # Process a complex command
    complex_response = service.process_text_command("bring the red cup from the table to the kitchen sink")
    print(f"Complex command success: {complex_response.success}")
    print(f"Processing time: {complex_response.total_processing_time:.3f}s")

    # Get service status
    status = service.get_service_status()
    print(f"\nService status: {status['is_running']}")
    print(f"Success rate: {status['adapter_status']['success_rate']:.2%}")
    print(f"Average processing time: {status['adapter_status']['average_processing_time']:.3f}s")

    service.stop()

    print("\nVoice to VLA Adapter test completed.")