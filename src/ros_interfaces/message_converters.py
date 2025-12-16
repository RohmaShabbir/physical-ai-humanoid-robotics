"""
Message Converters for ROS 2 interfaces in the Vision-Language-Action System
"""

from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

# ROS 2 message types
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from action_msgs.msg import GoalStatus


@dataclass
class VLAActionRequest:
    """Data class for VLA action request"""
    id: str
    voice_command: str
    processed_text: str
    visual_context: Dict[str, Any]
    timestamp: float
    status: str  # 'pending', 'processing', 'completed', 'failed'
    target_object: str
    target_location: str


@dataclass
class RobotSkill:
    """Data class for robot skill"""
    id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    preconditions: list
    postconditions: list
    timeout: int


@dataclass
class SkillChain:
    """Data class for skill chain"""
    id: str
    name: str
    skills: list  # List of RobotSkill IDs
    current_index: int
    status: str  # 'pending', 'executing', 'completed', 'failed', 'interrupted'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_info: Optional[Dict[str, Any]] = None


@dataclass
class ROS2Goal:
    """Data class for ROS 2 goal"""
    id: str
    action_type: str
    parameters: Dict[str, Any]
    priority: int
    creation_time: float
    execution_status: str  # 'pending', 'active', 'succeeded', 'aborted', 'canceled'
    result: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics"""
    id: str
    timestamp: float
    speech_to_action_latency: float
    vla_inference_time: float
    task_completion_time: float
    success_rate: float
    tokens_per_second: float
    memory_usage: float


class MessageConverter:
    """Converter for converting between internal data structures and ROS 2 messages"""

    @staticmethod
    def vla_request_to_ros_msg(request: VLAActionRequest) -> Dict[str, Any]:
        """
        Convert VLAActionRequest to ROS 2 message format
        """
        return {
            'id': request.id,
            'voice_command': request.voice_command,
            'processed_text': request.processed_text,
            'visual_context': request.visual_context,
            'timestamp': request.timestamp,
            'status': request.status,
            'target_object': request.target_object,
            'target_location': request.target_location
        }

    @staticmethod
    def ros_msg_to_vla_request(msg: Dict[str, Any]) -> VLAActionRequest:
        """
        Convert ROS 2 message to VLAActionRequest
        """
        return VLAActionRequest(
            id=msg.get('id', ''),
            voice_command=msg.get('voice_command', ''),
            processed_text=msg.get('processed_text', ''),
            visual_context=msg.get('visual_context', {}),
            timestamp=msg.get('timestamp', 0.0),
            status=msg.get('status', 'pending'),
            target_object=msg.get('target_object', ''),
            target_location=msg.get('target_location', '')
        )

    @staticmethod
    def skill_to_ros_msg(skill: RobotSkill) -> Dict[str, Any]:
        """
        Convert RobotSkill to ROS 2 message format
        """
        return {
            'id': skill.id,
            'name': skill.name,
            'description': skill.description,
            'parameters': skill.parameters,
            'preconditions': skill.preconditions,
            'postconditions': skill.postconditions,
            'timeout': skill.timeout
        }

    @staticmethod
    def ros_msg_to_skill(msg: Dict[str, Any]) -> RobotSkill:
        """
        Convert ROS 2 message to RobotSkill
        """
        return RobotSkill(
            id=msg.get('id', ''),
            name=msg.get('name', ''),
            description=msg.get('description', ''),
            parameters=msg.get('parameters', {}),
            preconditions=msg.get('preconditions', []),
            postconditions=msg.get('postconditions', []),
            timeout=msg.get('timeout', 30)
        )

    @staticmethod
    def skill_chain_to_ros_msg(skill_chain: SkillChain) -> Dict[str, Any]:
        """
        Convert SkillChain to ROS 2 message format
        """
        return {
            'id': skill_chain.id,
            'name': skill_chain.name,
            'skills': skill_chain.skills,
            'current_index': skill_chain.current_index,
            'status': skill_chain.status,
            'start_time': skill_chain.start_time,
            'end_time': skill_chain.end_time,
            'error_info': skill_chain.error_info
        }

    @staticmethod
    def ros_msg_to_skill_chain(msg: Dict[str, Any]) -> SkillChain:
        """
        Convert ROS 2 message to SkillChain
        """
        return SkillChain(
            id=msg.get('id', ''),
            name=msg.get('name', ''),
            skills=msg.get('skills', []),
            current_index=msg.get('current_index', 0),
            status=msg.get('status', 'pending'),
            start_time=msg.get('start_time'),
            end_time=msg.get('end_time'),
            error_info=msg.get('error_info')
        )

    @staticmethod
    def goal_to_ros_msg(goal: ROS2Goal) -> Dict[str, Any]:
        """
        Convert ROS2Goal to ROS 2 message format
        """
        return {
            'id': goal.id,
            'action_type': goal.action_type,
            'parameters': goal.parameters,
            'priority': goal.priority,
            'creation_time': goal.creation_time,
            'execution_status': goal.execution_status,
            'result': goal.result
        }

    @staticmethod
    def ros_msg_to_goal(msg: Dict[str, Any]) -> ROS2Goal:
        """
        Convert ROS 2 message to ROS2Goal
        """
        return ROS2Goal(
            id=msg.get('id', ''),
            action_type=msg.get('action_type', ''),
            parameters=msg.get('parameters', {}),
            priority=msg.get('priority', 5),
            creation_time=msg.get('creation_time', 0.0),
            execution_status=msg.get('execution_status', 'pending'),
            result=msg.get('result')
        )

    @staticmethod
    def metrics_to_ros_msg(metrics: PerformanceMetrics) -> Dict[str, Any]:
        """
        Convert PerformanceMetrics to ROS 2 message format
        """
        return {
            'id': metrics.id,
            'timestamp': metrics.timestamp,
            'speech_to_action_latency': metrics.speech_to_action_latency,
            'vla_inference_time': metrics.vla_inference_time,
            'task_completion_time': metrics.task_completion_time,
            'success_rate': metrics.success_rate,
            'tokens_per_second': metrics.tokens_per_second,
            'memory_usage': metrics.memory_usage
        }

    @staticmethod
    def ros_msg_to_metrics(msg: Dict[str, Any]) -> PerformanceMetrics:
        """
        Convert ROS 2 message to PerformanceMetrics
        """
        return PerformanceMetrics(
            id=msg.get('id', ''),
            timestamp=msg.get('timestamp', 0.0),
            speech_to_action_latency=msg.get('speech_to_action_latency', 0.0),
            vla_inference_time=msg.get('vla_inference_time', 0.0),
            task_completion_time=msg.get('task_completion_time', 0.0),
            success_rate=msg.get('success_rate', 0.0),
            tokens_per_second=msg.get('tokens_per_second', 0.0),
            memory_usage=msg.get('memory_usage', 0.0)
        )

    @staticmethod
    def image_to_numpy(image_msg: Image) -> np.ndarray:
        """
        Convert ROS Image message to numpy array
        """
        # This is a simplified conversion - in practice, you'd need to handle
        # different encoding formats
        if image_msg.encoding == 'rgb8':
            # Convert image data to numpy array
            img_array = np.frombuffer(image_msg.data, dtype=np.uint8)
            img_array = img_array.reshape((image_msg.height, image_msg.width, 3))
            return img_array
        else:
            # Handle other encodings as needed
            raise NotImplementedError(f"Image encoding {image_msg.encoding} not implemented")

    @staticmethod
    def numpy_to_image(numpy_img: np.ndarray, frame_id: str = 'camera') -> Image:
        """
        Convert numpy array to ROS Image message
        """
        # This is a simplified conversion - in practice, you'd need to properly
        # encode the image data according to ROS standards
        img_msg = Image()
        img_msg.header = Header()
        img_msg.header.stamp.sec = 0
        img_msg.header.stamp.nanosec = 0
        img_msg.header.frame_id = frame_id
        img_msg.height = numpy_img.shape[0]
        img_msg.width = numpy_img.shape[1]
        img_msg.encoding = 'rgb8'
        img_msg.is_bigendian = False
        img_msg.step = numpy_img.shape[1] * 3  # 3 channels for RGB
        img_msg.data = numpy_img.tobytes()
        return img_msg