"""
VLA to Skill Adapter for the Vision-Language-Action System
Connects VLA inference output to skill chain execution
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import get_system_logger
from vla_inference.inference_service import VLAInferenceResponse
from skill_library.skill_chain import SkillChainExecutor, SkillChain
from ros_interfaces.message_converters import RobotSkill, SkillChain


@dataclass
class VLAToSkillRequest:
    """
    Data class for VLA to skill adapter request
    """
    vla_response: VLAInferenceResponse
    context: Optional[Dict[str, Any]] = None


@dataclass
class VLAToSkillResponse:
    """
    Data class for VLA to skill adapter response
    """
    success: bool
    skill_chain: Optional[SkillChain] = None
    execution_results: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0


class VLAToSkillAdapter:
    """
    Adapter that converts VLA inference output to skill chain and executes it
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the VLA to skill adapter

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("VLAToSkillAdapter")
        self.config = self._load_config(config_path)

        # Initialize skill chain executor
        self.skill_executor = SkillChainExecutor()

        # Adapter state
        self.is_executing = False

        # Performance tracking
        self.total_conversions = 0
        self.total_executions = 0
        self.successful_executions = 0

        self.logger.info("VLA to Skill Adapter initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the VLA to skill adapter
        """
        default_config = {
            "enable_skill_chaining": True,
            "enable_context_expansion": True,
            "max_skills_per_chain": 10,
            "default_confidence_threshold": 0.6,
            "enable_fallback_skills": True
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

    def convert_and_execute(self, request: VLAToSkillRequest) -> VLAToSkillResponse:
        """
        Convert VLA response to skill chain and execute it

        Args:
            request: VLA to skill adapter request

        Returns:
            VLA to skill adapter response
        """
        if self.is_executing:
            return VLAToSkillResponse(
                success=False,
                error_message="Adapter already executing",
                processing_time=0.0
            )

        self.is_executing = True
        start_time = time.time()
        self.total_conversions += 1

        try:
            self.logger.info("Converting VLA response to skill chain and executing...")

            # Validate VLA response
            if not request.vla_response.success:
                if self.config.get("enable_fallback_skills", True):
                    # Create fallback skills based on the original request context
                    skill_chain = self._create_fallback_skill_chain(request)
                else:
                    return VLAToSkillResponse(
                        success=False,
                        error_message=f"VLA response not successful: {request.vla_response.error_message}",
                        processing_time=time.time() - start_time
                    )
            else:
                # Convert VLA response to skill chain
                skill_chain = self._convert_vla_response_to_skill_chain(request.vla_response, request.context)

            if not skill_chain or not skill_chain.skills:
                return VLAToSkillResponse(
                    success=False,
                    error_message="No skills generated from VLA response",
                    processing_time=time.time() - start_time
                )

            # Execute the skill chain
            self.total_executions += 1
            execution_success, execution_message, execution_results = self.skill_executor.execute_chain(skill_chain)

            if execution_success:
                self.successful_executions += 1

            response = VLAToSkillResponse(
                success=execution_success,
                skill_chain=skill_chain,
                execution_results=[result.__dict__ for result in execution_results],
                error_message=execution_message if not execution_success else None,
                processing_time=time.time() - start_time
            )

            self.logger.info(f"VLA to skill conversion and execution completed: {execution_success}")
            return response

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error in VLA to skill conversion: {e}")
            return VLAToSkillResponse(
                success=False,
                error_message=f"VLA to skill conversion failed: {e}",
                processing_time=processing_time
            )
        finally:
            self.is_executing = False

    def _convert_vla_response_to_skill_chain(self, vla_response: VLAInferenceResponse, context: Optional[Dict[str, Any]]) -> Optional[SkillChain]:
        """
        Convert VLA inference response to a skill chain

        Args:
            vla_response: VLA inference response
            context: Additional context

        Returns:
            SkillChain object or None if conversion failed
        """
        try:
            if not vla_response.robot_skill:
                return None

            # Extract the main skill from the VLA response
            main_skill = vla_response.robot_skill

            # Create a basic skill chain with the main skill
            skills = [main_skill]

            # If the VLA response suggests a complex action (like bringing something),
            # expand it into multiple skills
            if vla_response.execution_plan:
                expanded_skills = self._expand_execution_plan(vla_response.execution_plan, main_skill, context)
                if expanded_skills:
                    skills = expanded_skills

            # Create dependencies if needed (for multi-skill chains)
            dependencies = {}
            if len(skills) > 1:
                # For a sequence, each skill depends on the previous one
                for i in range(1, len(skills)):
                    dependencies[i] = [i-1]

            return SkillChain(
                skills=skills,
                execution_order=list(range(len(skills))),
                dependencies=dependencies,
                context=context or {}
            )

        except Exception as e:
            self.logger.error(f"Error converting VLA response to skill chain: {e}")
            return None

    def _expand_execution_plan(self, execution_plan: str, base_skill: RobotSkill, context: Optional[Dict[str, Any]]) -> Optional[List[RobotSkill]]:
        """
        Expand an execution plan into multiple skills if needed

        Args:
            execution_plan: Text description of the execution plan
            base_skill: The base robot skill from VLA
            context: Additional context

        Returns:
            List of RobotSkill objects or None if expansion not needed
        """
        plan_lower = execution_plan.lower()

        # Check if this is a complex plan that needs expansion
        # For example: "navigate to object -> grasp -> navigate to destination -> place"
        if "bring" in plan_lower or "take" in plan_lower or "move" in plan_lower and "to" in plan_lower and "from" in plan_lower:
            # Extract object and locations if available in context
            obj_name = base_skill.parameters.get("target_object", "object")
            target_loc = base_skill.parameters.get("target_location", "destination")

            # Create a sequence of skills: navigate -> grasp -> navigate -> place
            skills = []

            # 1. Navigate to object
            nav_to_obj = RobotSkill(
                skill_type="navigate",
                parameters={
                    "target_location": f"near_{obj_name}",
                    "target_position": base_skill.parameters.get("grasp_position", [1.0, 1.0, 0.0]),
                    "path_type": "shortest",
                    "speed_profile": "normal"
                },
                confidence=base_skill.confidence * 0.9,  # Slightly lower confidence for derived skill
                execution_time=time.time(),
                status="pending"
            )
            skills.append(nav_to_obj)

            # 2. Grasp object
            grasp_skill = RobotSkill(
                skill_type="grasp",
                parameters=base_skill.parameters,
                confidence=base_skill.confidence,
                execution_time=time.time(),
                status="pending"
            )
            skills.append(grasp_skill)

            # 3. Navigate to destination
            nav_to_dest = RobotSkill(
                skill_type="navigate",
                parameters={
                    "target_location": target_loc,
                    "target_position": base_skill.parameters.get("placement_position", [2.0, 2.0, 0.0]),
                    "path_type": "shortest",
                    "speed_profile": "normal"
                },
                confidence=base_skill.confidence * 0.9,
                execution_time=time.time(),
                status="pending"
            )
            skills.append(nav_to_dest)

            # 4. Place object
            place_skill = RobotSkill(
                skill_type="place",
                parameters={
                    "target_location": target_loc,
                    "placement_type": "surface",
                    "placement_position": base_skill.parameters.get("placement_position", [2.0, 2.0, 0.1])
                },
                confidence=base_skill.confidence,
                execution_time=time.time(),
                status="pending"
            )
            skills.append(place_skill)

            return skills

        return None  # No expansion needed

    def _create_fallback_skill_chain(self, request: VLAToSkillRequest) -> Optional[SkillChain]:
        """
        Create a fallback skill chain when VLA inference fails

        Args:
            request: VLA to skill adapter request

        Returns:
            SkillChain object or None if unable to create fallback
        """
        # Try to extract command information from context
        if request.context:
            # Look for command information in context
            original_command = request.context.get("original_command", "")
            if original_command:
                # Parse the command to create basic skills
                skills = self._parse_command_to_skills(original_command)
                if skills:
                    return SkillChain(
                        skills=skills,
                        execution_order=list(range(len(skills))),
                        dependencies={},
                        context=request.context
                    )

        return None

    def _parse_command_to_skills(self, command: str) -> List[RobotSkill]:
        """
        Parse a natural language command into a sequence of skills

        Args:
            command: Natural language command

        Returns:
            List of RobotSkill objects
        """
        command_lower = command.lower()
        skills = []

        # Simple command parsing
        if any(word in command_lower for word in ["pick", "grasp", "grab", "take"]):
            # Extract object if possible
            import re
            obj_match = re.search(r"(?:the|a|an)\s+(\w+(?:\s+\w+)?)\s+(?:on|from|at)", command_lower)
            obj_name = obj_match.group(1) if obj_match else "object"

            skills.append(RobotSkill(
                skill_type="grasp",
                parameters={
                    "target_object": obj_name,
                    "grasp_type": "pinch",
                    "grasp_position": [0.5, 0.5, 0.1],
                    "grasp_orientation": [0.0, 0.0, 0.0, 1.0]
                },
                confidence=0.6,  # Lower confidence for fallback
                execution_time=time.time(),
                status="pending"
            ))

        if any(word in command_lower for word in ["go", "move", "navigate", "walk", "to"]):
            # Extract destination if possible
            import re
            dest_match = re.search(r"(?:to|toward|at)\s+(?:the\s+)?(\w+(?:\s+\w+)?)", command_lower)
            dest_name = dest_match.group(1) if dest_match else "destination"

            skills.append(RobotSkill(
                skill_type="navigate",
                parameters={
                    "target_location": dest_name,
                    "target_position": [1.0, 1.0, 0.0],
                    "path_type": "shortest",
                    "speed_profile": "normal"
                },
                confidence=0.65,
                execution_time=time.time(),
                status="pending"
            ))

        if any(word in command_lower for word in ["place", "put", "drop", "set"]):
            # Extract placement location if possible
            import re
            place_match = re.search(r"(?:on|at|in)\s+(?:the\s+)?(\w+(?:\s+\w+)?)", command_lower)
            place_name = place_match.group(1) if place_match else "location"

            skills.append(RobotSkill(
                skill_type="place",
                parameters={
                    "target_location": place_name,
                    "placement_type": "surface",
                    "placement_position": [1.0, 1.0, 0.1]
                },
                confidence=0.62,
                execution_time=time.time(),
                status="pending"
            ))

        return skills

    def execute_skill_chain(self, skill_chain: SkillChain) -> VLAToSkillResponse:
        """
        Execute a skill chain directly (bypassing VLA conversion)

        Args:
            skill_chain: The skill chain to execute

        Returns:
            VLA to skill adapter response
        """
        if self.is_executing:
            return VLAToSkillResponse(
                success=False,
                error_message="Adapter already executing",
                processing_time=0.0
            )

        self.is_executing = True
        start_time = time.time()
        self.total_executions += 1

        try:
            execution_success, execution_message, execution_results = self.skill_executor.execute_chain(skill_chain)

            if execution_success:
                self.successful_executions += 1

            response = VLAToSkillResponse(
                success=execution_success,
                skill_chain=skill_chain,
                execution_results=[result.__dict__ for result in execution_results],
                error_message=execution_message if not execution_success else None,
                processing_time=time.time() - start_time
            )

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error executing skill chain: {e}")
            return VLAToSkillResponse(
                success=False,
                error_message=f"Skill chain execution failed: {e}",
                processing_time=processing_time
            )
        finally:
            self.is_executing = False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the adapter

        Returns:
            Dictionary with performance metrics
        """
        success_rate = (
            self.successful_executions / self.total_executions
            if self.total_executions > 0
            else 0.0
        )

        return {
            "total_conversions": self.total_conversions,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "success_rate": success_rate,
            "is_executing": self.is_executing,
            "config": self.config,
            "skill_executor_metrics": self.skill_executor.get_performance_metrics()
        }

    def is_active(self) -> bool:
        """
        Check if the adapter is currently executing

        Returns:
            True if executing, False otherwise
        """
        return self.is_executing

    def cancel_execution(self) -> bool:
        """
        Cancel current execution if active

        Returns:
            True if cancellation was successful, False otherwise
        """
        if not self.is_executing:
            return False

        self.logger.info("Cancelling VLA to skill execution")
        # In a real implementation, this would cancel ongoing operations
        return True


class VLAToSkillService:
    """
    Service interface for the VLA to skill adapter
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the VLA to skill service

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("VLAToSkillService")
        self.adapter = VLAToSkillAdapter(config_path)
        self.is_running = False

    def start(self):
        """Start the VLA to skill service"""
        self.is_running = True
        self.logger.info("VLA to Skill Service started")

    def stop(self):
        """Stop the VLA to skill service"""
        self.is_running = False
        self.logger.info("VLA to Skill Service stopped")

    def convert_and_execute(self, request: VLAToSkillRequest) -> VLAToSkillResponse:
        """
        Convert VLA response to skill chain and execute it through the service

        Args:
            request: VLA to skill adapter request

        Returns:
            VLA to skill adapter response
        """
        if not self.is_running:
            return VLAToSkillResponse(
                success=False,
                error_message="VLA to Skill service is not running",
                processing_time=0.0
            )

        return self.adapter.convert_and_execute(request)

    def execute_skill_chain(self, skill_chain: SkillChain) -> VLAToSkillResponse:
        """
        Execute a skill chain directly through the service

        Args:
            skill_chain: The skill chain to execute

        Returns:
            VLA to skill adapter response
        """
        if not self.is_running:
            return VLAToSkillResponse(
                success=False,
                error_message="VLA to Skill service is not running",
                processing_time=0.0
            )

        return self.adapter.execute_skill_chain(skill_chain)

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the service status

        Returns:
            Dictionary with service status information
        """
        return {
            "is_running": self.is_running,
            "adapter_status": self.adapter.get_performance_metrics()
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing VLA to Skill Adapter...")

    # Create VLA to skill adapter
    adapter = VLAToSkillAdapter()

    # Create a mock VLA inference response
    from vla_inference.inference_service import VLAInferenceResponse
    from ros_interfaces.message_converters import RobotSkill

    mock_robot_skill = RobotSkill(
        skill_type="grasp",
        parameters={
            "target_object": "red cup",
            "grasp_type": "pinch",
            "grasp_position": [0.5, 0.2, 0.1],
            "grasp_orientation": [0.0, 0.0, 0.0, 1.0]
        },
        confidence=0.85,
        execution_time=time.time(),
        status="pending"
    )

    mock_vla_response = VLAInferenceResponse(
        success=True,
        robot_skill=mock_robot_skill,
        confidence=0.85,
        error_message=None,
        execution_plan="Grasp the red cup using pinch grasp",
        processing_time=0.5
    )

    # Test the adapter
    print("\n1. Testing VLA to skill conversion and execution...")
    request = VLAToSkillRequest(
        vla_response=mock_vla_response,
        context={"original_command": "pick up the red cup", "task": "grasp_demo"}
    )

    response = adapter.convert_and_execute(request)

    print(f"  Success: {response.success}")
    print(f"  Processing time: {response.processing_time:.3f}s")
    if response.skill_chain:
        print(f"  Skills in chain: {len(response.skill_chain.skills)}")
        for i, skill in enumerate(response.skill_chain.skills):
            print(f"    Skill {i}: {skill.skill_type} (confidence: {skill.confidence:.2f})")
    if response.error_message:
        print(f"  Error: {response.error_message}")

    # Test with a more complex plan
    print(f"\n2. Testing with complex execution plan...")
    complex_skill = RobotSkill(
        skill_type="navigate",
        parameters={
            "target_location": "kitchen",
            "target_position": [2.0, 0.0, 0.0]
        },
        confidence=0.8,
        execution_time=time.time(),
        status="pending"
    )

    complex_vla_response = VLAInferenceResponse(
        success=True,
        robot_skill=complex_skill,
        confidence=0.8,
        error_message=None,
        execution_plan="Navigate to kitchen, then grasp object, then navigate to destination and place",
        processing_time=0.7
    )

    complex_request = VLAToSkillRequest(
        vla_response=complex_vla_response,
        context={"original_command": "bring the cup to the table", "task": "complex_demo"}
    )

    complex_response = adapter.convert_and_execute(complex_request)
    print(f"  Success: {complex_response.success}")
    print(f"  Skills in chain: {len(complex_response.skill_chain.skills) if complex_response.skill_chain else 0}")

    # Test with VLA failure (should use fallback)
    print(f"\n3. Testing with failed VLA response (fallback)...")
    failed_vla_response = VLAInferenceResponse(
        success=False,
        robot_skill=None,
        confidence=0.0,
        error_message="VLA inference failed",
        execution_plan=None,
        processing_time=0.1
    )

    fallback_request = VLAToSkillRequest(
        vla_response=failed_vla_response,
        context={"original_command": "pick up the red cup", "task": "fallback_demo"}
    )

    fallback_response = adapter.convert_and_execute(fallback_request)
    print(f"  Success: {fallback_response.success}")
    print(f"  Skills in chain: {len(fallback_response.skill_chain.skills) if fallback_response.skill_chain else 0}")

    # Test with the service wrapper
    print(f"\n4. Testing VLA to Skill Service...")
    service = VLAToSkillService()
    service.start()

    service_response = service.convert_and_execute(request)
    print(f"  Service execution success: {service_response.success}")
    print(f"  Service processing time: {service_response.processing_time:.3f}s")

    # Get service status
    status = service.get_service_status()
    print(f"\n  Service status: {status['is_running']}")
    print(f"  Success rate: {status['adapter_status']['success_rate']:.2%}")

    service.stop()

    print("\nVLA to Skill Adapter test completed.")