"""
LLM Prompt Templates for the Vision-Language-Action System
Handles conversion of voice commands to structured ROS 2 goals using LLM prompting
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import get_system_logger
from ros_interfaces.message_converters import VLAActionRequest, ROS2Goal


@dataclass
class LLMTemplateRequest:
    """
    Data class for LLM template request
    """
    voice_command: str
    context: Optional[Dict[str, Any]] = None
    available_objects: Optional[List[str]] = None
    available_locations: Optional[List[str]] = None
    robot_capabilities: Optional[List[str]] = None


@dataclass
class LLMTemplateResponse:
    """
    Data class for LLM template response
    """
    success: bool
    prompt_template: Optional[str] = None
    structured_goals: Optional[List[ROS2Goal]] = None
    error_message: Optional[str] = None


class LLMPromptTemplateGenerator:
    """
    Generator for LLM prompts that convert voice commands to ROS 2 goals
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the LLM prompt template generator

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("LLMPromptTemplateGenerator")
        self.config = self._load_config(config_path)

        # Default templates for different command types
        self.command_templates = {
            "grasp": {
                "system_prompt": "You are a robotics command interpreter. Convert natural language commands to structured robot goals. Identify the target object and grasp parameters.",
                "user_prompt": "Command: '{command}'\nAvailable objects: {available_objects}\nRobot capabilities: {capabilities}\n\nGenerate a structured grasp goal with target object, grasp type, and position.",
                "output_format": {
                    "action_type": "GraspAction",
                    "parameters": {
                        "target_object": "object name",
                        "grasp_type": "pinch|power|lateral|suction",
                        "approach_angle": "degrees",
                        "grasp_width": "meters"
                    }
                }
            },
            "navigate": {
                "system_prompt": "You are a robotics command interpreter. Convert natural language commands to structured robot goals. Identify the target location and navigation parameters.",
                "user_prompt": "Command: '{command}'\nAvailable locations: {available_locations}\nRobot capabilities: {capabilities}\n\nGenerate a structured navigation goal with target location and path parameters.",
                "output_format": {
                    "action_type": "NavigateAction",
                    "parameters": {
                        "target_location": "location name",
                        "path_type": "shortest|safe|fast",
                        "speed_profile": "slow|normal|fast",
                        "target_position": "[x, y, theta]"
                    }
                }
            },
            "place": {
                "system_prompt": "You are a robotics command interpreter. Convert natural language commands to structured robot goals. Identify the placement location and parameters.",
                "user_prompt": "Command: '{command}'\nAvailable locations: {available_locations}\nRobot capabilities: {capabilities}\n\nGenerate a structured place goal with target location and placement parameters.",
                "output_format": {
                    "action_type": "PlaceAction",
                    "parameters": {
                        "target_location": "location name",
                        "placement_type": "surface|shelf|container|stack",
                        "placement_position": "[x, y, z]",
                        "orientation": "[qx, qy, qz, qw]"
                    }
                }
            },
            "complex": {
                "system_prompt": "You are a robotics command interpreter. Convert complex natural language commands to multiple structured robot goals. Break down the command into a sequence of actions.",
                "user_prompt": "Command: '{command}'\nAvailable objects: {available_objects}\nAvailable locations: {available_locations}\nRobot capabilities: {capabilities}\n\nGenerate a sequence of structured goals for the robot to execute.",
                "output_format": {
                    "action_sequence": [
                        {
                            "action_type": "action type",
                            "parameters": {"param": "value"}
                        }
                    ]
                }
            }
        }

        # Default objects and locations
        self.default_objects = [
            "red cup", "blue bottle", "green book", "white plate", "black phone",
            "silver keys", "yellow notebook", "orange container", "purple mug",
            "pink glass", "brown box", "gray laptop", "gold pen", "silver bowl"
        ]
        self.default_locations = [
            "kitchen", "living room", "bedroom", "bathroom", "office",
            "kitchen sink", "kitchen counter", "dining table", "coffee table",
            "bedside table", "couch", "chair", "desk", "shelf", "cabinet"
        ]
        self.default_capabilities = [
            "grasp", "navigate", "place", "speak", "perceive", "manipulate"
        ]

        self.logger.info("LLM Prompt Template Generator initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the LLM prompt template generator
        """
        default_config = {
            "enable_chain_of_thought": True,
            "enable_self_correction": True,
            "default_model": "gpt-4-turbo",
            "temperature": 0.3,
            "max_tokens": 1000,
            "enable_validation": True
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

    def generate_prompt_template(self, request: LLMTemplateRequest) -> LLMTemplateResponse:
        """
        Generate an LLM prompt template for the given voice command

        Args:
            request: LLM template request containing voice command and context

        Returns:
            LLM template response with prompt and structured goals
        """
        try:
            self.logger.info(f"Generating LLM prompt for command: {request.voice_command}")

            # Determine command type
            command_type = self._classify_command(request.voice_command)

            # Get the appropriate template
            template = self.command_templates.get(command_type, self.command_templates["complex"])

            # Prepare context
            available_objects = request.available_objects or self.default_objects
            available_locations = request.available_locations or self.default_locations
            capabilities = request.robot_capabilities or self.default_capabilities

            # Format the prompt
            user_prompt = template["user_prompt"].format(
                command=request.voice_command,
                available_objects=", ".join(available_objects),
                available_locations=", ".join(available_locations),
                capabilities=", ".join(capabilities)
            )

            system_prompt = template["system_prompt"]

            # Create the full prompt template
            full_template = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "output_format": template["output_format"],
                "command_type": command_type,
                "voice_command": request.voice_command
            }

            # Generate structured goals from the command
            structured_goals = self._generate_structured_goals(
                request.voice_command, command_type,
                available_objects, available_locations, capabilities
            )

            response = LLMTemplateResponse(
                success=True,
                prompt_template=json.dumps(full_template, indent=2),
                structured_goals=structured_goals
            )

            return response

        except Exception as e:
            self.logger.error(f"Error generating LLM prompt template: {e}")
            return LLMTemplateResponse(
                success=False,
                error_message=str(e)
            )

    def _classify_command(self, command: str) -> str:
        """
        Classify the voice command to determine the appropriate template

        Args:
            command: The voice command to classify

        Returns:
            Command type ('grasp', 'navigate', 'place', or 'complex')
        """
        command_lower = command.lower()

        # Count keywords for each action type
        grasp_keywords = ["pick", "grasp", "grab", "take", "lift", "get", "hold"]
        navigate_keywords = ["go", "move", "navigate", "walk", "travel", "head", "come", "approach"]
        place_keywords = ["place", "put", "set", "drop", "lay", "position", "depose"]

        grasp_count = sum(1 for keyword in grasp_keywords if keyword in command_lower)
        navigate_count = sum(1 for keyword in navigate_keywords if keyword in command_lower)
        place_count = sum(1 for keyword in place_keywords if keyword in command_lower)

        # Determine primary action
        max_count = max(grasp_count, navigate_count, place_count)

        if max_count == 0:
            # No clear action, might be complex
            if any(word in command_lower for word in ["bring", "take", "move"]):
                return "complex"
            else:
                return "complex"  # Default to complex for ambiguous commands
        elif grasp_count == max_count:
            return "grasp"
        elif navigate_count == max_count:
            return "navigate"
        elif place_count == max_count:
            return "place"

        return "complex"

    def _generate_structured_goals(self, command: str, command_type: str,
                                 available_objects: List[str],
                                 available_locations: List[str],
                                 capabilities: List[str]) -> List[ROS2Goal]:
        """
        Generate structured ROS 2 goals from the voice command

        Args:
            command: The voice command
            command_type: The classified command type
            available_objects: List of available objects
            available_locations: List of available locations
            capabilities: List of robot capabilities

        Returns:
            List of ROS2Goal objects
        """
        goals = []

        if command_type == "grasp":
            # Extract target object
            target_object = self._extract_object_from_command(command, available_objects)
            goal_params = {
                "target_object": target_object,
                "grasp_type": "pinch",
                "approach_angle": 45.0,
                "grasp_width": 0.05
            }
            goal = ROS2Goal(
                id=f"grasp_{hash(command) % 10000}",
                action_type="GraspAction",
                parameters=goal_params,
                priority=5,
                creation_time=0,  # Will be set by system
                execution_status="pending"
            )
            goals.append(goal)

        elif command_type == "navigate":
            # Extract target location
            target_location = self._extract_location_from_command(command, available_locations)
            goal_params = {
                "target_location": target_location,
                "path_type": "shortest",
                "speed_profile": "normal",
                "target_position": [1.0, 1.0, 0.0]
            }
            goal = ROS2Goal(
                id=f"navigate_{hash(command) % 10000}",
                action_type="NavigateAction",
                parameters=goal_params,
                priority=5,
                creation_time=0,
                execution_status="pending"
            )
            goals.append(goal)

        elif command_type == "place":
            # Extract target location
            target_location = self._extract_location_from_command(command, available_locations)
            goal_params = {
                "target_location": target_location,
                "placement_type": "surface",
                "placement_position": [1.0, 1.0, 0.1],
                "orientation": [0.0, 0.0, 0.0, 1.0]
            }
            goal = ROS2Goal(
                id=f"place_{hash(command) % 10000}",
                action_type="PlaceAction",
                parameters=goal_params,
                priority=5,
                creation_time=0,
                execution_status="pending"
            )
            goals.append(goal)

        elif command_type == "complex":
            # Handle complex commands that might involve multiple steps
            # For example: "Bring the red cup from the table to the kitchen sink"
            goals.extend(self._generate_complex_goals(command, available_objects, available_locations))

        return goals

    def _extract_object_from_command(self, command: str, available_objects: List[str]) -> str:
        """
        Extract the target object from the command by matching with available objects

        Args:
            command: The voice command
            available_objects: List of available objects

        Returns:
            The best matching object name
        """
        command_lower = command.lower()

        # Look for exact matches first
        for obj in available_objects:
            if obj.lower() in command_lower:
                return obj

        # If no exact match, look for partial matches
        for obj in available_objects:
            obj_words = obj.lower().split()
            for word in obj_words:
                if word in command_lower:
                    return obj

        # If still no match, return a default
        return "unknown_object"

    def _extract_location_from_command(self, command: str, available_locations: List[str]) -> str:
        """
        Extract the target location from the command by matching with available locations

        Args:
            command: The voice command
            available_locations: List of available locations

        Returns:
            The best matching location name
        """
        command_lower = command.lower()

        # Look for exact matches first
        for loc in available_locations:
            if loc.lower() in command_lower:
                return loc

        # If no exact match, look for partial matches
        for loc in available_locations:
            loc_words = loc.lower().split()
            for word in loc_words:
                if word in command_lower:
                    return loc

        # If still no match, return a default
        return "unknown_location"

    def _generate_complex_goals(self, command: str, available_objects: List[str],
                               available_locations: List[str]) -> List[ROS2Goal]:
        """
        Generate goals for complex commands that involve multiple actions

        Args:
            command: The complex voice command
            available_objects: List of available objects
            available_locations: List of available locations

        Returns:
            List of ROS2Goal objects for the complex task
        """
        goals = []
        command_lower = command.lower()

        # Check if this is a "bring/take X from Y to Z" command
        import re

        # Pattern for "bring X from Y to Z" or "take X from Y to Z"
        bring_pattern = r"(?:bring|take|move)\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+from\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+to\s+(?:the\s+)?(\w+(?:\s+\w+)?)"
        bring_match = re.search(bring_pattern, command_lower)

        if bring_match:
            target_object = bring_match.group(1)
            start_location = bring_match.group(2)
            end_location = bring_match.group(3)

            # 1. Navigate to start location
            nav_to_obj = ROS2Goal(
                id=f"navigate_to_{hash(start_location) % 10000}",
                action_type="NavigateAction",
                parameters={
                    "target_location": start_location,
                    "path_type": "shortest",
                    "speed_profile": "normal"
                },
                priority=5,
                creation_time=0,
                execution_status="pending"
            )
            goals.append(nav_to_obj)

            # 2. Grasp the object
            grasp_obj = ROS2Goal(
                id=f"grasp_{hash(target_object) % 10000}",
                action_type="GraspAction",
                parameters={
                    "target_object": target_object,
                    "grasp_type": "pinch",
                    "approach_angle": 45.0
                },
                priority=5,
                creation_time=0,
                execution_status="pending"
            )
            goals.append(grasp_obj)

            # 3. Navigate to end location
            nav_to_dest = ROS2Goal(
                id=f"navigate_to_{hash(end_location) % 10000}",
                action_type="NavigateAction",
                parameters={
                    "target_location": end_location,
                    "path_type": "shortest",
                    "speed_profile": "normal"
                },
                priority=5,
                creation_time=0,
                execution_status="pending"
            )
            goals.append(nav_to_dest)

            # 4. Place the object
            place_obj = ROS2Goal(
                id=f"place_{hash(target_object) % 10000}",
                action_type="PlaceAction",
                parameters={
                    "target_location": end_location,
                    "placement_type": "surface"
                },
                priority=5,
                creation_time=0,
                execution_status="pending"
            )
            goals.append(place_obj)

        else:
            # For other complex commands, try to identify multiple actions
            if any(word in command_lower for word in ["and", "&"]):
                # Command might have multiple parts separated by "and"
                parts = re.split(r'\s+(?:and|&)\s+', command_lower)
                for part in parts:
                    part_type = self._classify_command(part)
                    if part_type in ["grasp", "navigate", "place"]:
                        sub_goals = self._generate_structured_goals(
                            part, part_type, available_objects, available_locations, self.default_capabilities
                        )
                        goals.extend(sub_goals)

        return goals

    def get_prompt_for_llm(self, request: LLMTemplateRequest) -> str:
        """
        Get a formatted prompt ready to send to an LLM

        Args:
            request: LLM template request

        Returns:
            Formatted prompt string
        """
        response = self.generate_prompt_template(request)
        if response.success and response.prompt_template:
            return response.prompt_template
        else:
            # Return a simple fallback prompt
            return f"Convert this command to robot goals: {request.voice_command}"

    def validate_generated_goals(self, goals: List[ROS2Goal]) -> bool:
        """
        Validate that the generated goals are reasonable

        Args:
            goals: List of ROS2Goal objects to validate

        Returns:
            True if goals are valid, False otherwise
        """
        if not goals:
            return False

        for goal in goals:
            if not goal.action_type or not goal.parameters:
                return False

            # Check that action type is one of the expected types
            valid_actions = ["GraspAction", "NavigateAction", "PlaceAction", "FindObjectAction"]
            if goal.action_type not in valid_actions:
                return False

        return True


class LLMPromptService:
    """
    Service interface for the LLM prompt template generator
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the LLM prompt service

        Args:
            config_path: Path to configuration file
        """
        self.logger = get_system_logger("LLMPromptService")
        self.generator = LLMPromptTemplateGenerator(config_path)
        self.is_running = False

    def start(self):
        """Start the LLM prompt service"""
        self.is_running = True
        self.logger.info("LLM Prompt Service started")

    def stop(self):
        """Stop the LLM prompt service"""
        self.is_running = False
        self.logger.info("LLM Prompt Service stopped")

    def generate_prompt(self, request: LLMTemplateRequest) -> LLMTemplateResponse:
        """
        Generate an LLM prompt through the service

        Args:
            request: LLM template request

        Returns:
            LLM template response
        """
        if not self.is_running:
            return LLMTemplateResponse(
                success=False,
                error_message="LLM Prompt service is not running"
            )

        return self.generator.generate_prompt_template(request)

    def get_formatted_prompt(self, voice_command: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get a formatted prompt ready to send to an LLM

        Args:
            voice_command: The voice command to convert
            context: Additional context

        Returns:
            Formatted prompt string
        """
        if not self.is_running:
            return f"Error: LLM Prompt service is not running. Command: {voice_command}"

        request = LLMTemplateRequest(
            voice_command=voice_command,
            context=context
        )
        return self.generator.get_prompt_for_llm(request)

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the service status

        Returns:
            Dictionary with service status information
        """
        return {
            "is_running": self.is_running,
            "config": self.generator.config
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing LLM Prompt Template Generator...")

    # Create LLM prompt template generator
    generator = LLMPromptTemplateGenerator()

    # Test with different commands
    test_commands = [
        "pick up the red cup",
        "go to the kitchen sink",
        "place the bottle on the table",
        "bring the red cup from the table to the kitchen sink",
        "take the blue book to the living room"
    ]

    for i, cmd in enumerate(test_commands):
        print(f"\nTest {i+1}: Command '{cmd}'")

        request = LLMTemplateRequest(
            voice_command=cmd,
            available_objects=["red cup", "blue bottle", "green book", "white plate"],
            available_locations=["kitchen sink", "table", "living room", "bedroom"],
            robot_capabilities=["grasp", "navigate", "place"]
        )

        response = generator.generate_prompt_template(request)

        print(f"  Success: {response.success}")
        if response.structured_goals:
            print(f"  Generated {len(response.structured_goals)} goals:")
            for j, goal in enumerate(response.structured_goals):
                print(f"    {j+1}. {goal.action_type}: {goal.parameters}")

        if response.error_message:
            print(f"  Error: {response.error_message}")

    # Test with the service wrapper
    print(f"\nTesting LLM Prompt Service...")
    service = LLMPromptService()
    service.start()

    formatted_prompt = service.get_formatted_prompt("pick up the red cup")
    print(f"Formatted prompt:\n{formatted_prompt}")

    # Get service status
    status = service.get_service_status()
    print(f"\nService status: {status['is_running']}")

    service.stop()

    print("\nLLM Prompt Template Generator test completed.")