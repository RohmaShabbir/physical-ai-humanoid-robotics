"""
VLA Model Wrapper for the Vision-Language-Action System
Handles VLA model loading, preprocessing, and inference integration
"""

import os
import sys
import time
import json
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logging_utils import get_system_logger
from .tensorrt_engine import TensorRTEngine, get_engine_cache
from ros_interfaces.message_converters import VLAActionRequest, RobotSkill


class VLAProcessor:
    """
    Preprocessor for VLA model inputs - handles image and text preprocessing
    """

    def __init__(self,
                 image_size: Tuple[int, int] = (224, 224),
                 normalize_mean: List[float] = [0.485, 0.456, 0.406],
                 normalize_std: List[float] = [0.229, 0.224, 0.225]):
        """
        Initialize the VLA preprocessor

        Args:
            image_size: Size to resize input images to
            normalize_mean: Mean values for image normalization
            normalize_std: Standard deviation values for image normalization
        """
        self.image_size = image_size
        self.normalize_mean = np.array(normalize_mean, dtype=np.float32)
        self.normalize_std = np.array(normalize_std, dtype=np.float32)
        self.logger = get_system_logger("VLAProcessor")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for VLA model input

        Args:
            image: Input image as numpy array (H, W, C) or (C, H, W)

        Returns:
            Preprocessed image ready for model input
        """
        # Ensure image is in correct format (H, W, C)
        if image.ndim == 3:
            if image.shape[0] == 3:  # Channel-first format
                image = np.transpose(image, (1, 2, 0))
        elif image.ndim != 3:
            raise ValueError(f"Invalid image dimensions: {image.shape}")

        # Resize image
        from PIL import Image
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize(self.image_size)
        resized_image = np.array(pil_image).astype(np.float32) / 255.0

        # Normalize
        normalized_image = (resized_image - self.normalize_mean) / self.normalize_std

        # Convert to channel-first format for model input
        normalized_image = np.transpose(normalized_image, (2, 0, 1))

        return normalized_image.astype(np.float32)

    def preprocess_text(self, text: str) -> np.ndarray:
        """
        Preprocess text for VLA model input
        In a real implementation, this would convert text to token embeddings

        Args:
            text: Input text string

        Returns:
            Preprocessed text as numpy array (placeholder implementation)
        """
        # Placeholder implementation - in a real system, this would use a tokenizer
        # to convert text to token embeddings
        self.logger.warning("Text preprocessing is using placeholder implementation")
        # Return a simple representation of text length and basic features
        # This would be replaced with actual tokenization in a real implementation
        text_features = np.array([len(text), hash(text) % 1000], dtype=np.float32)
        return text_features


class VLAModel:
    """
    VLA Model wrapper that integrates with TensorRT engine for efficient inference
    """

    def __init__(self,
                 engine_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 precision: str = "float16",
                 max_batch_size: int = 1,
                 image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the VLA model wrapper

        Args:
            engine_path: Path to pre-built TensorRT engine file
            model_path: Path to original model file (for building engine if needed)
            precision: Precision for inference ('float16', 'int8', 'float32')
            max_batch_size: Maximum batch size for inference
            image_size: Size to resize input images to
        """
        self.logger = get_system_logger("VLAModel")
        self.engine_path = engine_path
        self.model_path = model_path
        self.precision = precision
        self.max_batch_size = max_batch_size
        self.image_size = image_size

        # Initialize preprocessor
        self.preprocessor = VLAProcessor(image_size=image_size)

        # Initialize TensorRT engine
        self.engine = None
        self._load_engine()

        # Model metadata
        self.model_info = {
            "name": "VLA Model Wrapper",
            "version": "1.0",
            "input_shape": (3, image_size[0], image_size[1]),  # C, H, W
            "max_batch_size": max_batch_size
        }

        self.logger.info(f"VLA Model initialized with engine: {engine_path is not None}")

    def _load_engine(self):
        """
        Load or create the TensorRT engine for VLA model
        """
        if self.engine_path:
            # Try to get from cache first
            cache = get_engine_cache()
            self.engine = cache.get_engine(
                self.engine_path,
                model_path=self.model_path,
                precision=self.precision,
                max_batch_size=self.max_batch_size
            )

            if self.engine is None:
                # If cache failed, create directly
                self.engine = TensorRTEngine(
                    engine_path=self.engine_path,
                    model_path=self.model_path,
                    precision=self.precision,
                    max_batch_size=self.max_batch_size
                )
        else:
            # Create engine directly if no path provided
            self.engine = TensorRTEngine(
                model_path=self.model_path,
                precision=self.precision,
                max_batch_size=self.max_batch_size
            )

        if self.engine.engine is None:
            self.logger.warning("VLA model engine not loaded - using mock functionality")

    def encode_image_text(self, image: np.ndarray, text: str) -> Optional[np.ndarray]:
        """
        Encode image and text to create VLA embedding

        Args:
            image: Input image as numpy array
            text: Input text string

        Returns:
            VLA embedding as numpy array or None if failed
        """
        if self.engine is None:
            # Mock implementation
            self.logger.warning("Using mock VLA encoding")
            # Create a mock embedding combining image and text features
            image_features = np.mean(image, axis=(1, 2))  # Simple spatial average
            text_features = self.preprocessor.preprocess_text(text)
            combined_features = np.concatenate([image_features.flatten(), text_features])
            return combined_features.astype(np.float32)

        try:
            # Preprocess inputs
            processed_image = self.preprocessor.preprocess_image(image)
            processed_text = self.preprocessor.preprocess_text(text)

            # Combine image and text features for VLA model
            # In a real implementation, this would follow the specific VLA model's input format
            # For now, we'll concatenate them with a simple approach
            combined_input = np.concatenate([
                processed_image.flatten(),
                processed_text
            ]).astype(np.float32)

            # Reshape to match expected input format (batch_size, features)
            combined_input = combined_input.reshape(1, -1)

            # Perform inference
            output = self.engine.infer(combined_input)

            return output

        except Exception as e:
            self.logger.error(f"Error in VLA encoding: {e}")
            return None

    def predict_action(self, image: np.ndarray, text: str) -> Optional[Dict[str, Any]]:
        """
        Predict robot action from image and text input

        Args:
            image: Input image as numpy array
            text: Input text string (command)

        Returns:
            Predicted action as dictionary or None if failed
        """
        embedding = self.encode_image_text(image, text)

        if embedding is None:
            return None

        if self.engine is None:
            # Mock implementation for action prediction
            self.logger.warning("Using mock action prediction")
            # Parse the text command to extract action type and parameters
            action_type = self._parse_action_from_text(text)

            # Create mock action based on command
            if action_type == "grasp":
                action = {
                    "action_type": "grasp",
                    "target_object": self._extract_object_from_text(text),
                    "grasp_position": [0.5, 0.5, 0.1],  # Mock coordinates
                    "grasp_orientation": [0.0, 0.0, 0.0, 1.0]  # Mock quaternion
                }
            elif action_type == "navigate":
                action = {
                    "action_type": "navigate",
                    "target_location": self._extract_location_from_text(text),
                    "target_position": [1.0, 1.0, 0.0]  # Mock coordinates
                }
            elif action_type == "place":
                action = {
                    "action_type": "place",
                    "target_location": self._extract_location_from_text(text),
                    "placement_position": [0.8, 0.8, 0.1]  # Mock coordinates
                }
            else:
                action = {
                    "action_type": "unknown",
                    "raw_command": text
                }

            return action

        try:
            # In a real implementation, the embedding would be processed further
            # to extract specific action parameters
            # For now, we'll use the embedding to generate a mock action
            action_probs = np.random.random(4)  # Mock probabilities for 4 action types
            action_types = ["grasp", "navigate", "place", "other"]
            predicted_action_type = action_types[np.argmax(action_probs)]

            return {
                "action_type": predicted_action_type,
                "confidence": float(np.max(action_probs)),
                "raw_output": embedding.tolist() if embedding is not None else []
            }

        except Exception as e:
            self.logger.error(f"Error in action prediction: {e}")
            return None

    def _parse_action_from_text(self, text: str) -> str:
        """
        Parse action type from text command (mock implementation)
        """
        text_lower = text.lower()

        if any(word in text_lower for word in ["pick", "grab", "take", "get", "lift"]):
            return "grasp"
        elif any(word in text_lower for word in ["go", "move", "navigate", "walk", "go to"]):
            return "navigate"
        elif any(word in text_lower for word in ["place", "put", "drop", "set"]):
            return "place"
        else:
            return "other"

    def _extract_object_from_text(self, text: str) -> str:
        """
        Extract object name from text command (mock implementation)
        """
        # Simple keyword matching for common objects
        object_keywords = [
            "cup", "bottle", "plate", "book", "box", "ball", "toy", "phone",
            "computer", "laptop", "remote", "keys", "wallet", "hat", "shoe",
            "glass", "mug", "bowl", "spoon", "fork", "knife", "red cup",
            "blue book", "green bottle"
        ]

        text_lower = text.lower()
        for obj in object_keywords:
            if obj in text_lower:
                # Extract with color if present
                words = text_lower.split()
                for i, word in enumerate(words):
                    if obj.split()[0] in word and i + 1 < len(words):
                        possible_obj = f"{word} {words[i+1]}"
                        if possible_obj in object_keywords:
                            return possible_obj
                return obj

        return "unknown_object"

    def _extract_location_from_text(self, text: str) -> str:
        """
        Extract location from text command (mock implementation)
        """
        # Simple keyword matching for common locations
        location_keywords = [
            "kitchen", "living room", "bedroom", "bathroom", "office",
            "table", "counter", "couch", "chair", "desk", "shelf", "cabinet",
            "sink", "refrigerator", "microwave", "oven", "stove", "bed",
            "door", "window", "hallway", "dining room", "pantry", "laundry room"
        ]

        text_lower = text.lower()
        for loc in location_keywords:
            if loc in text_lower:
                return loc

        return "unknown_location"

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded VLA model
        """
        if self.engine:
            engine_info = self.engine.get_engine_info()
            self.model_info["engine_info"] = engine_info

        return self.model_info

    def process_vla_request(self, request: VLAActionRequest) -> Optional[RobotSkill]:
        """
        Process a VLA action request and return a robot skill

        Args:
            request: VLA action request containing image, text, and context

        Returns:
            RobotSkill object or None if processing failed
        """
        try:
            self.logger.info(f"Processing VLA request: {request.action_type}")

            # Predict action from image and text
            action_prediction = self.predict_action(request.image_data, request.text_instruction)

            if action_prediction is None:
                self.logger.error("Failed to predict action from VLA request")
                return None

            # Convert prediction to robot skill
            skill = RobotSkill(
                skill_type=action_prediction.get("action_type", "unknown"),
                parameters=action_prediction,
                confidence=action_prediction.get("confidence", 0.5),
                execution_time=time.time(),
                status="pending"
            )

            self.logger.info(f"Generated robot skill: {skill.skill_type}")
            return skill

        except Exception as e:
            self.logger.error(f"Error processing VLA request: {e}")
            return None


class VLAInferenceService:
    """
    VLA Inference Service following the contract specification
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

        self.is_running = False
        self.request_count = 0
        self.total_inference_time = 0.0

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
            "enable_caching": True
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

    def call_inference(self, request: VLAActionRequest) -> Optional[RobotSkill]:
        """
        Call the VLA inference service with a request

        Args:
            request: VLA action request

        Returns:
            Robot skill prediction or None if failed
        """
        start_time = time.time()
        self.request_count += 1

        try:
            # Process the request with the VLA model
            skill = self.vla_model.process_vla_request(request)

            if skill:
                inference_time = time.time() - start_time
                self.total_inference_time += inference_time
                avg_time = self.total_inference_time / self.request_count
                self.logger.info(f"VLA inference completed in {inference_time:.3f}s (avg: {avg_time:.3f}s)")

                return skill
            else:
                self.logger.error("VLA inference returned no skill")
                return None

        except Exception as e:
            self.logger.error(f"Error in VLA inference service: {e}")
            return None

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
            "model_info": self.vla_model.get_model_info()
        }


# Example usage and testing
if __name__ == "__main__":
    import numpy as np

    # Create logger
    logger = get_system_logger("VLAModelTest")

    print("Testing VLA Model Wrapper...")

    # Create a mock image (224x224 RGB)
    mock_image = np.random.random((224, 224, 3)).astype(np.float32)

    # Test VLA model wrapper
    print("\n1. Testing VLA Model wrapper...")
    vla_model = VLAModel()

    # Test encoding
    text_command = "pick up the red cup"
    embedding = vla_model.encode_image_text(mock_image, text_command)
    print(f"  Embedding shape: {embedding.shape if embedding is not None else 'None'}")

    # Test action prediction
    action = vla_model.predict_action(mock_image, text_command)
    print(f"  Predicted action: {action}")

    # Test VLA inference service
    print("\n2. Testing VLA Inference Service...")
    service = VLAInferenceService()

    # Create a mock VLA action request
    from ros_interfaces.message_converters import VLAActionRequest
    request = VLAActionRequest(
        action_type="grasp",
        image_data=mock_image,
        text_instruction=text_command,
        context={"robot_state": "idle", "environment": "kitchen"}
    )

    # Call inference
    skill = service.call_inference(request)
    print(f"  Generated skill: {skill.skill_type if skill else 'None'}")

    # Get performance metrics
    metrics = service.get_performance_metrics()
    print(f"  Performance metrics: {metrics}")

    print("\nVLA Model wrapper test completed.")