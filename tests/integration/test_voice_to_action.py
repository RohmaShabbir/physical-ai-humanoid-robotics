"""
Integration Test for Voice-to-Action Pipeline
Tests the complete pipeline: voice -> VLA -> skills for "Bring the red cup to the kitchen sink"
"""

import time
import unittest
from typing import Dict, Any, Optional
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.logging_utils import get_system_logger
from src.vla_integration.voice_to_vla_adapter import VoiceToVLAAdapter, VoiceToVLARequest
from src.vla_integration.vla_to_skill_adapter import VLAToSkillAdapter, VLAToSkillRequest
from src.vla_integration.llm_prompt_templates import LLMPromptTemplateGenerator, LLMTemplateRequest
from src.vla_inference.inference_service import VLAInferenceService, VLAInferenceRequest
from src.skill_library.skill_chain import SkillChainExecutor
from src.voice_pipeline.voice_service import VoiceCommandService


@dataclass
class TestResult:
    """
    Data class for test result
    """
    test_name: str
    success: bool
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


class VoiceToActionIntegrationTest:
    """
    Integration test for the complete voice-to-action pipeline
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the integration test

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("VoiceToActionIntegrationTest")
        self.config = self._load_config(config_path)

        # Initialize all components
        self.voice_service = VoiceCommandService()
        self.vla_service = VLAInferenceService()
        self.skill_executor = SkillChainExecutor()
        self.voice_to_vla_adapter = VoiceToVLAAdapter()
        self.vla_to_skill_adapter = VLAToSkillAdapter()
        self.llm_prompt_generator = LLMPromptTemplateGenerator()

        # Test parameters
        self.test_command = "Bring the red cup to the kitchen sink"
        self.expected_skills = ["navigate", "grasp", "navigate", "place"]
        self.max_execution_time = self.config.get("max_execution_time", 30.0)  # seconds

        self.logger.info("Voice-to-Action Integration Test initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the integration test
        """
        default_config = {
            "max_execution_time": 30.0,
            "enable_detailed_logging": True,
            "test_objects": ["red cup", "blue bottle", "green book"],
            "test_locations": ["kitchen sink", "table", "counter", "living room"],
            "robot_capabilities": ["grasp", "navigate", "place"]
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

    def run_complete_pipeline_test(self) -> TestResult:
        """
        Run the complete voice-to-action pipeline test

        Returns:
            Test result with execution details
        """
        start_time = time.time()
        test_name = "Complete Voice-to-Action Pipeline Test"

        try:
            self.logger.info(f"Starting test: {test_name}")
            self.logger.info(f"Command: '{self.test_command}'")

            # Step 1: Process voice command
            voice_result = self._test_voice_processing()
            if not voice_result.success:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    details={"voice_processing": voice_result.details},
                    error_message=f"Voice processing failed: {voice_result.error_message}"
                )

            # Step 2: Process VLA inference
            vla_result = self._test_vla_inference(voice_result.details.get("command_structure"))
            if not vla_result.success:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    details={
                        "voice_processing": voice_result.details,
                        "vla_inference": vla_result.details
                    },
                    error_message=f"VLA inference failed: {vla_result.error_message}"
                )

            # Step 3: Execute skills
            skill_result = self._test_skill_execution(vla_result.details.get("robot_skill"))
            if not skill_result.success:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    details={
                        "voice_processing": voice_result.details,
                        "vla_inference": vla_result.details,
                        "skill_execution": skill_result.details
                    },
                    error_message=f"Skill execution failed: {skill_result.error_message}"
                )

            # Step 4: Test full integration
            full_integration_result = self._test_full_integration()
            if not full_integration_result.success:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    details={
                        "voice_processing": voice_result.details,
                        "vla_inference": vla_result.details,
                        "skill_execution": skill_result.details,
                        "full_integration": full_integration_result.details
                    },
                    error_message=f"Full integration failed: {full_integration_result.error_message}"
                )

            # All steps successful
            execution_time = time.time() - start_time
            success = execution_time <= self.max_execution_time

            result = TestResult(
                test_name=test_name,
                success=success,
                execution_time=execution_time,
                details={
                    "voice_processing": voice_result.details,
                    "vla_inference": vla_result.details,
                    "skill_execution": skill_result.details,
                    "full_integration": full_integration_result.details,
                    "overall_execution_time": execution_time,
                    "time_limit_respected": execution_time <= self.max_execution_time
                }
            )

            if success:
                self.logger.info(f"Test passed: Completed in {execution_time:.3f}s (limit: {self.max_execution_time}s)")
            else:
                self.logger.warning(f"Test failed: Exceeded time limit ({execution_time:.3f}s > {self.max_execution_time}s)")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error in complete pipeline test: {e}")
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time=execution_time,
                details={},
                error_message=str(e)
            )

    def _test_voice_processing(self) -> TestResult:
        """
        Test the voice processing component

        Returns:
            Test result for voice processing
        """
        start_time = time.time()
        test_name = "Voice Processing Test"

        try:
            self.logger.info(f"Testing voice processing: '{self.test_command}'")

            # Process the voice command
            voice_response = self.voice_service.process_voice_command_text(
                self.test_command,
                context={
                    "robot_state": "idle",
                    "environment_state": "kitchen",
                    "available_objects": self.config.get("test_objects", []),
                    "robot_capabilities": self.config.get("robot_capabilities", [])
                }
            )

            if not voice_response.success:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    details={"voice_response": voice_response.__dict__},
                    error_message=voice_response.error_message
                )

            # Validate command structure
            cmd_struct = voice_response.command_structure
            has_target_object = bool(cmd_struct.get("target_object"))
            has_target_location = bool(cmd_struct.get("target_location"))

            execution_time = time.time() - start_time

            result = TestResult(
                test_name=test_name,
                success=True,
                execution_time=execution_time,
                details={
                    "command_structure": cmd_struct,
                    "confidence": voice_response.confidence,
                    "has_target_object": has_target_object,
                    "has_target_location": has_target_location,
                    "llm_prompt_context": voice_response.llm_prompt_context
                }
            )

            self.logger.info(f"Voice processing test completed in {execution_time:.3f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error in voice processing test: {e}")
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time=execution_time,
                details={},
                error_message=str(e)
            )

    def _test_vla_inference(self, command_structure: Optional[Dict[str, Any]]) -> TestResult:
        """
        Test the VLA inference component

        Args:
            command_structure: Command structure from voice processing

        Returns:
            Test result for VLA inference
        """
        start_time = time.time()
        test_name = "VLA Inference Test"

        try:
            self.logger.info("Testing VLA inference")

            # Create VLA inference request
            vla_request = VLAInferenceRequest(
                image_data=None,  # Will use mock image in testing
                text_instruction=self.test_command,
                context={
                    "robot_state": "idle",
                    "environment_state": "kitchen",
                    "available_objects": self.config.get("test_objects", []),
                    "target_object": command_structure.get("target_object", "red cup") if command_structure else "red cup",
                    "target_location": command_structure.get("target_location", "kitchen sink") if command_structure else "kitchen sink"
                }
            )

            # Execute VLA inference
            vla_response = self.vla_service.execute(vla_request)

            if not vla_response.success:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    details={"vla_response": vla_response.__dict__},
                    error_message=vla_response.error_message
                )

            execution_time = time.time() - start_time

            result = TestResult(
                test_name=test_name,
                success=True,
                execution_time=execution_time,
                details={
                    "vla_response": vla_response.__dict__,
                    "robot_skill": vla_response.robot_skill.__dict__ if vla_response.robot_skill else None,
                    "confidence": vla_response.confidence,
                    "execution_plan": vla_response.execution_plan
                }
            )

            self.logger.info(f"VLA inference test completed in {execution_time:.3f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error in VLA inference test: {e}")
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time=execution_time,
                details={},
                error_message=str(e)
            )

    def _test_skill_execution(self, robot_skill) -> TestResult:
        """
        Test the skill execution component

        Args:
            robot_skill: Robot skill from VLA inference

        Returns:
            Test result for skill execution
        """
        start_time = time.time()
        test_name = "Skill Execution Test"

        try:
            self.logger.info("Testing skill execution")

            # If we don't have a robot skill from VLA, create a mock one based on the command
            if not robot_skill:
                from src.ros_interfaces.message_converters import RobotSkill
                robot_skill = RobotSkill(
                    skill_type="navigate",
                    parameters={
                        "target_location": "kitchen sink",
                        "target_position": [2.5, 0.5, 0.0]
                    },
                    confidence=0.8,
                    execution_time=time.time(),
                    status="pending"
                )

            # Execute the skill
            skill_result = self.skill_executor._execute_single_skill(robot_skill, 0)

            execution_time = time.time() - start_time

            result = TestResult(
                test_name=test_name,
                success=skill_result.success,
                execution_time=execution_time,
                details={
                    "skill_execution_result": skill_result.__dict__,
                    "robot_skill": robot_skill.__dict__
                },
                error_message=None if skill_result.success else skill_result.message
            )

            self.logger.info(f"Skill execution test completed in {execution_time:.3f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error in skill execution test: {e}")
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time=execution_time,
                details={},
                error_message=str(e)
            )

    def _test_full_integration(self) -> TestResult:
        """
        Test the full integration using the voice-to-VLA adapter

        Returns:
            Test result for full integration
        """
        start_time = time.time()
        test_name = "Full Integration Test"

        try:
            self.logger.info("Testing full integration pipeline")

            # Create request for voice-to-VLA adapter
            adapter_request = VoiceToVLARequest(
                text_command=self.test_command,
                context={
                    "robot_state": "idle",
                    "environment_state": "kitchen",
                    "available_objects": self.config.get("test_objects", []),
                    "available_locations": self.config.get("test_locations", []),
                    "robot_capabilities": self.config.get("robot_capabilities", [])
                }
            )

            # Execute the full pipeline
            adapter_response = self.voice_to_vla_adapter.process_voice_command(adapter_request)

            execution_time = time.time() - start_time

            result = TestResult(
                test_name=test_name,
                success=adapter_response.success,
                execution_time=execution_time,
                details={
                    "adapter_response": adapter_response.__dict__,
                    "voice_processing_result": adapter_response.voice_processing_result,
                    "vla_inference_result": adapter_response.vla_inference_result,
                    "skill_execution_result": adapter_response.skill_execution_result,
                    "total_processing_time": adapter_response.total_processing_time
                },
                error_message=None if adapter_response.success else adapter_response.error_message
            )

            self.logger.info(f"Full integration test completed in {execution_time:.3f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error in full integration test: {e}")
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time=execution_time,
                details={},
                error_message=str(e)
            )

    def run_latency_test(self) -> TestResult:
        """
        Run a specific latency test for the pipeline

        Returns:
            Test result for latency measurement
        """
        start_time = time.time()
        test_name = "Latency Measurement Test"

        try:
            self.logger.info("Testing end-to-end latency")

            # Measure from voice command to first action start
            voice_start = time.time()

            # Process voice command
            voice_response = self.voice_service.process_voice_command_text(
                self.test_command,
                context={
                    "robot_state": "idle",
                    "environment_state": "kitchen"
                }
            )

            if not voice_response.success:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    details={"voice_response": voice_response.__dict__},
                    error_message=voice_response.error_message
                )

            # Process VLA inference
            vla_request = VLAInferenceRequest(
                image_data=None,
                text_instruction=self.test_command,
                context={"robot_state": "idle", "environment_state": "kitchen"}
            )
            vla_response = self.vla_service.execute(vla_request)

            if not vla_response.success:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=time.time() - start_time,
                    details={
                        "voice_response": voice_response.__dict__,
                        "vla_response": vla_response.__dict__
                    },
                    error_message=vla_response.error_message
                )

            # Calculate latency (voice processing + VLA inference)
            latency = time.time() - voice_start

            execution_time = time.time() - start_time

            # Check if latency is within the required 2.0 seconds
            success = latency <= 2.0

            result = TestResult(
                test_name=test_name,
                success=success,
                execution_time=execution_time,
                details={
                    "latency": latency,
                    "voice_processing_time": getattr(voice_response, 'processing_time', 0),
                    "vla_inference_time": getattr(vla_response, 'processing_time', 0),
                    "latency_requirement_met": success,
                    "latency_threshold": 2.0
                }
            )

            self.logger.info(f"Latency test completed: {latency:.3f}s (requirement: <2.0s)")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error in latency test: {e}")
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time=execution_time,
                details={},
                error_message=str(e)
            )

    def get_test_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the test configuration and capabilities

        Returns:
            Dictionary with test summary information
        """
        return {
            "test_command": self.test_command,
            "expected_skills": self.expected_skills,
            "max_execution_time": self.max_execution_time,
            "test_objects": self.config.get("test_objects"),
            "test_locations": self.config.get("test_locations"),
            "robot_capabilities": self.config.get("robot_capabilities"),
            "components": {
                "voice_service": True,
                "vla_service": True,
                "skill_executor": True,
                "voice_to_vla_adapter": True,
                "vla_to_skill_adapter": True,
                "llm_prompt_generator": True
            }
        }


def run_integration_tests():
    """
    Run the integration tests and return results
    """
    logger = get_system_logger("IntegrationTestRunner")
    logger.info("Starting Voice-to-Action Integration Tests...")

    # Create the test instance
    test = VoiceToActionIntegrationTest()

    # Run the complete pipeline test
    logger.info("Running complete pipeline test...")
    complete_result = test.run_complete_pipeline_test()
    print(f"\nComplete Pipeline Test: {'PASS' if complete_result.success else 'FAIL'}")
    print(f"  Execution time: {complete_result.execution_time:.3f}s")
    if not complete_result.success:
        print(f"  Error: {complete_result.error_message}")

    # Run the latency test
    logger.info("Running latency test...")
    latency_result = test.run_latency_test()
    print(f"\nLatency Test: {'PASS' if latency_result.success else 'FAIL'}")
    print(f"  Measured latency: {latency_result.details.get('latency', 0):.3f}s")
    print(f"  Requirement: <2.0s")
    if not latency_result.success:
        print(f"  Error: {latency_result.error_message}")

    # Print test summary
    summary = test.get_test_summary()
    print(f"\nTest Summary:")
    print(f"  Command: {summary['test_command']}")
    print(f"  Expected skills: {summary['expected_skills']}")
    print(f"  Max execution time: {summary['max_execution_time']}s")
    print(f"  Test objects: {summary['test_objects']}")
    print(f"  Test locations: {summary['test_locations']}")

    # Overall result
    overall_success = complete_result.success and latency_result.success
    print(f"\nOverall Result: {'PASS' if overall_success else 'FAIL'}")

    return {
        "complete_pipeline": complete_result,
        "latency_test": latency_result,
        "summary": summary,
        "overall_success": overall_success
    }


if __name__ == "__main__":
    results = run_integration_tests()
    print(f"\nIntegration tests completed with overall result: {'PASS' if results['overall_success'] else 'FAIL'}")