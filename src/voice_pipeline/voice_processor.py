"""
Voice Processor for the Vision-Language-Action System
Handles voice command parsing and conversion to structured robot goals
"""

import re
import json
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import get_system_logger
from ros_interfaces.message_converters import VLAActionRequest, ROS2Goal


class CommandType(Enum):
    """Types of voice commands that can be processed"""
    GRASP = "grasp"
    NAVIGATE = "navigate"
    PLACE = "place"
    FOLLOW = "follow"
    FIND = "find"
    GREET = "greet"
    OTHER = "other"


@dataclass
class ParsedCommand:
    """Data class for parsed voice commands"""
    command_type: CommandType
    target_object: Optional[str] = None
    target_location: Optional[str] = None
    action_params: Optional[Dict[str, Any]] = None
    original_text: str = ""
    confidence: float = 0.0


class VoiceCommandParser:
    """
    Parses natural language voice commands into structured robot commands
    """

    def __init__(self):
        self.logger = get_system_logger("VoiceCommandParser")
        self.object_keywords = [
            "cup", "bottle", "plate", "book", "box", "ball", "toy", "phone",
            "computer", "laptop", "remote", "keys", "wallet", "hat", "shoe",
            "glass", "mug", "bowl", "spoon", "fork", "knife"
        ]
        self.location_keywords = [
            "kitchen", "living room", "bedroom", "bathroom", "office",
            "table", "counter", "couch", "chair", "desk", "shelf", "cabinet",
            "sink", "refrigerator", "microwave", "oven", "stove", "bed",
            "door", "window", "hallway", "dining room", "pantry", "laundry room"
        ]
        self.color_keywords = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "brown", "black", "white", "gray", "grey", "silver", "gold"
        ]

        # Define command patterns for different action types
        self.command_patterns = {
            CommandType.GRASP: [
                r"pick up the? (?P<object>[\w\s]+)",
                r"grab the? (?P<object>[\w\s]+)",
                r"take the? (?P<object>[\w\s]+)",
                r"get the? (?P<object>[\w\s]+)",
                r"lift the? (?P<object>[\w\s]+)"
            ],
            CommandType.NAVIGATE: [
                r"go to the? (?P<location>[\w\s]+)",
                r"move to the? (?P<location>[\w\s]+)",
                r"navigate to the? (?P<location>[\w\s]+)",
                r"walk to the? (?P<location>[\w\s]+)",
                r"go near the? (?P<location>[\w\s]+)"
            ],
            CommandType.PLACE: [
                r"place (?:it|the [\w\s]+) in?o? the? (?P<location>[\w\s]+)",
                r"put (?:it|the [\w\s]+) in?o? the? (?P<location>[\w\s]+)",
                r"drop (?:it|the [\w\s]+) in?o? the? (?P<location>[\w\s]+)",
                r"set (?:it|the [\w\s]+) in?o? the? (?P<location>[\w\s]+)"
            ]
        }

    def parse_command(self, text: str) -> ParsedCommand:
        """
        Parse a natural language command into structured format

        Args:
            text: The natural language command text

        Returns:
            ParsedCommand with structured information
        """
        text = text.lower().strip()
        if not text:
            return ParsedCommand(CommandType.OTHER, original_text=text, confidence=0.0)

        # Try to match each command type
        for cmd_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    groups = match.groupdict()
                    target_object = groups.get('object', '').strip()
                    target_location = groups.get('location', '').strip()

                    # Extract additional context
                    action_params = self._extract_action_params(text)

                    # Calculate confidence based on match quality
                    confidence = self._calculate_confidence(text, cmd_type, target_object, target_location)

                    return ParsedCommand(
                        command_type=cmd_type,
                        target_object=target_object or None,
                        target_location=target_location or None,
                        action_params=action_params,
                        original_text=text,
                        confidence=confidence
                    )

        # If no specific pattern matched, try to identify general intent
        return self._identify_general_intent(text)

    def _extract_action_params(self, text: str) -> Dict[str, Any]:
        """
        Extract additional parameters from the command text
        """
        params = {}

        # Extract color if mentioned
        for color in self.color_keywords:
            if color in text:
                params['color'] = color
                break

        # Extract size if mentioned
        size_patterns = [
            (r"small|tiny|little", "small"),
            (r"big|large|huge|massive", "large"),
            (r"medium|average|normal", "medium")
        ]
        for pattern, size in size_patterns:
            if re.search(pattern, text):
                params['size'] = size
                break

        # Extract any numeric quantities
        numbers = re.findall(r'\d+', text)
        if numbers:
            params['quantity'] = int(numbers[0])

        return params

    def _calculate_confidence(self, text: str, cmd_type: CommandType,
                            target_object: str, target_location: str) -> float:
        """
        Calculate confidence score for the parsed command
        """
        confidence = 0.7  # Base confidence

        # Increase confidence if we have specific targets
        if target_object:
            confidence += 0.15
        if target_location:
            confidence += 0.15

        # Increase confidence if text is well-formed
        if len(text.split()) >= 3:  # At least 3 words
            confidence += 0.1

        # Cap at 1.0
        return min(confidence, 1.0)

    def _identify_general_intent(self, text: str) -> ParsedCommand:
        """
        Identify general intent when specific patterns don't match
        """
        # Look for keywords that might indicate intent
        if any(word in text for word in ["find", "locate", "where"]):
            return ParsedCommand(
                command_type=CommandType.FIND,
                original_text=text,
                confidence=0.6
            )
        elif any(word in text for word in ["follow", "come", "behind"]):
            return ParsedCommand(
                command_type=CommandType.FOLLOW,
                original_text=text,
                confidence=0.6
            )
        elif any(word in text for word in ["hello", "hi", "hey", "greetings"]):
            return ParsedCommand(
                command_type=CommandType.GREET,
                original_text=text,
                confidence=0.8
            )
        else:
            return ParsedCommand(
                command_type=CommandType.OTHER,
                original_text=text,
                confidence=0.3
            )

    def extract_objects_and_locations(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract potential objects and locations from text
        """
        text_lower = text.lower()
        found_objects = []
        found_locations = []

        # Extract known objects
        for obj in self.object_keywords:
            if obj in text_lower:
                found_objects.append(obj)

        # Extract known locations
        for loc in self.location_keywords:
            if loc in text_lower:
                found_locations.append(loc)

        return found_objects, found_locations


class VoiceProcessor:
    """
    Main voice processor that integrates speech recognition, command parsing,
    and goal generation for the robot
    """

    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_system_logger("VoiceProcessor")
        self.command_parser = VoiceCommandParser()
        self.config = self._load_config(config_path)
        self.is_listening = False
        self.command_queue = []
        self.command_lock = threading.Lock()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for the voice processor
        """
        if config_path and not config_path.endswith('.yaml'):
            config_path += '.yaml'

        default_config = {
            "command_threshold": 0.5,
            "max_command_history": 10,
            "enable_context_awareness": True
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

    def process_voice_command(self, text: str) -> List[ROS2Goal]:
        """
        Process a voice command and generate ROS 2 goals

        Args:
            text: The voice command text

        Returns:
            List of ROS2Goal objects representing the actions to execute
        """
        self.logger.info(f"Processing voice command: '{text}'")

        # Parse the command
        parsed_cmd = self.command_parser.parse_command(text)

        if parsed_cmd.confidence < self.config["command_threshold"]:
            self.logger.warning(f"Command confidence too low ({parsed_cmd.confidence}), ignoring")
            return []

        # Generate goals based on the parsed command
        goals = self._generate_goals_from_command(parsed_cmd)

        # Log the generated goals
        self.logger.info(f"Generated {len(goals)} goals from command: {[g.action_type for g in goals]}")

        return goals

    def _generate_goals_from_command(self, parsed_cmd: ParsedCommand) -> List[ROS2Goal]:
        """
        Generate ROS 2 goals from a parsed command
        """
        goals = []

        if parsed_cmd.command_type == CommandType.GRASP:
            # Generate grasp goal
            goal_params = {
                "target_object": parsed_cmd.target_object or "",
                "grasp_type": "pinch",  # Default grasp type
                "approach_angle": 45.0
            }
            if parsed_cmd.action_params:
                goal_params.update(parsed_cmd.action_params)

            grasp_goal = ROS2Goal(
                id=f"grasp_{int(time.time())}",
                action_type="GraspAction",
                parameters=goal_params,
                priority=5,
                creation_time=time.time(),
                execution_status="pending"
            )
            goals.append(grasp_goal)

        elif parsed_cmd.command_type == CommandType.NAVIGATE:
            # Generate navigation goal
            goal_params = {
                "target_location": parsed_cmd.target_location or "",
                "path_type": "shortest",
                "speed_profile": "normal"
            }

            navigate_goal = ROS2Goal(
                id=f"navigate_{int(time.time())}",
                action_type="NavigateAction",
                parameters=goal_params,
                priority=5,
                creation_time=time.time(),
                execution_status="pending"
            )
            goals.append(navigate_goal)

        elif parsed_cmd.command_type == CommandType.PLACE:
            # Generate place goal
            goal_params = {
                "target_location": parsed_cmd.target_location or "",
                "placement_type": "surface"
            }

            place_goal = ROS2Goal(
                id=f"place_{int(time.time())}",
                action_type="PlaceAction",
                parameters=goal_params,
                priority=5,
                creation_time=time.time(),
                execution_status="pending"
            )
            goals.append(place_goal)

        elif parsed_cmd.command_type == CommandType.FIND:
            # Generate find/locate goal
            goal_params = {
                "target_object": parsed_cmd.target_object or parsed_cmd.original_text,
                "search_area": "current_room"
            }

            find_goal = ROS2Goal(
                id=f"find_{int(time.time())}",
                action_type="FindObjectAction",
                parameters=goal_params,
                priority=3,
                creation_time=time.time(),
                execution_status="pending"
            )
            goals.append(find_goal)

        # Add additional goals based on context or command sequence
        if self.config["enable_context_awareness"]:
            goals.extend(self._add_contextual_goals(parsed_cmd, goals))

        return goals

    def _add_contextual_goals(self, parsed_cmd: ParsedCommand, existing_goals: List[ROS2Goal]) -> List[ROS2Goal]:
        """
        Add contextual goals based on command history or current state
        """
        contextual_goals = []

        # Example: If the command involves bringing something to a location,
        # we might need a sequence of navigate -> grasp -> navigate -> place
        original_text = parsed_cmd.original_text.lower()
        if ("bring" in original_text or "take" in original_text) and parsed_cmd.target_location:
            # This is likely a multi-step command: go get X and bring it to Y
            # We would need to identify the object to get and add the appropriate sequence
            pass

        return contextual_goals

    def process_command_sequence(self, commands: List[str]) -> List[ROS2Goal]:
        """
        Process a sequence of voice commands and generate a combined plan

        Args:
            commands: List of command strings

        Returns:
            Combined list of ROS2Goal objects
        """
        all_goals = []
        for cmd in commands:
            goals = self.process_voice_command(cmd)
            all_goals.extend(goals)

        return all_goals

    def validate_command(self, text: str) -> Tuple[bool, str]:
        """
        Validate if a command is appropriate for execution

        Args:
            text: The command text to validate

        Returns:
            Tuple of (is_valid, reason)
        """
        text_lower = text.lower().strip()

        # Check for potentially unsafe commands
        unsafe_keywords = [
            "harm", "hurt", "break", "destroy", "damage", "attack",
            "fight", "hit", "slam", "crash", "dangerous"
        ]

        for keyword in unsafe_keywords:
            if keyword in text_lower:
                return False, f"Command contains potentially unsafe keyword: {keyword}"

        # Check command length
        if len(text) > 200:  # Arbitrary limit
            return False, "Command is too long"

        # Check if it's empty
        if not text:
            return False, "Command is empty"

        return True, "Command is valid"

    def get_command_suggestions(self, partial_command: str) -> List[str]:
        """
        Provide command suggestions based on partial input
        """
        suggestions = []
        partial_lower = partial_command.lower()

        # Suggest common commands that match the partial input
        common_commands = [
            "pick up the red cup",
            "go to the kitchen",
            "place the bottle on the table",
            "find my keys",
            "navigate to the living room",
            "grasp the blue book"
        ]

        for cmd in common_commands:
            if partial_lower in cmd.lower():
                suggestions.append(cmd)

        return suggestions[:5]  # Return top 5 suggestions


# Example usage and testing
if __name__ == "__main__":
    import time

    # Create voice processor
    logger = get_system_logger("VoiceProcessorTest")
    processor = VoiceProcessor()

    # Test commands
    test_commands = [
        "pick up the red cup",
        "go to the kitchen sink",
        "place the bottle on the table",
        "find my keys",
        "grab the blue book",
        "move to the living room"
    ]

    print("Testing Voice Command Processing...")
    for cmd in test_commands:
        print(f"\nProcessing: '{cmd}'")
        is_valid, reason = processor.validate_command(cmd)
        print(f"  Valid: {is_valid} ({reason})")

        if is_valid:
            goals = processor.process_voice_command(cmd)
            print(f"  Generated {len(goals)} goals:")
            for i, goal in enumerate(goals):
                print(f"    {i+1}. {goal.action_type} with params: {goal.parameters}")

    # Test command parsing in detail
    print("\nDetailed command parsing test:")
    parser = VoiceCommandParser()
    test_command = "pick up the red cup and bring it to the kitchen sink"
    parsed = parser.parse_command(test_command)
    print(f"Original: {test_command}")
    print(f"Parsed - Type: {parsed.command_type}, Object: {parsed.target_object}, Location: {parsed.target_location}, Params: {parsed.action_params}, Confidence: {parsed.confidence}")