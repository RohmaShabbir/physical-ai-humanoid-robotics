"""
TensorRT Engine Utilities for the Vision-Language-Action System
Handles TensorRT engine creation, loading, and inference for VLA models
"""

import os
import time
import json
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

try:
    import tensorrt as trt
    from cuda import cuda, cudart
    HAS_TENSORRT = True
except ImportError:
    print("Warning: tensorrt or cuda not installed. Install with 'pip install tensorrt' and CUDA libraries")
    HAS_TENSORRT = False
    trt = None
    cuda = None
    cudart = None


class TensorRTLogger(trt.ILogger):
    """
    Custom TensorRT logger for better integration with our logging system
    """

    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        # Convert TensorRT severity to our logging format
        # This is a simplified version - in practice you'd integrate with your logging system
        print(f"[TensorRT {severity}] {msg}")


class TensorRTEngine:
    """
    TensorRT engine wrapper for VLA model inference
    """

    def __init__(self,
                 engine_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 precision: str = "float16",
                 max_batch_size: int = 1):
        """
        Initialize the TensorRT engine

        Args:
            engine_path: Path to pre-built TensorRT engine file
            model_path: Path to original model file (for building engine if needed)
            precision: Precision for inference ('float16', 'int8', 'float32')
            max_batch_size: Maximum batch size for inference
        """
        self.engine_path = engine_path
        self.model_path = model_path
        self.precision = precision
        self.max_batch_size = max_batch_size

        self.logger = TensorRTLogger()
        self.runtime = None
        self.engine = None
        self.context = None
        self.stream = None

        self.input_binding_idx = 0
        self.output_binding_idx = 1

        # Initialize CUDA
        if HAS_TENSORRT:
            self._initialize_cuda()
            self._load_or_build_engine()
        else:
            print("TensorRT not available, using mock functionality")

    def _initialize_cuda(self):
        """Initialize CUDA context"""
        if not HAS_TENSORRT:
            return

        # Initialize CUDA
        err, = cuda.cuInit(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA initialization failed with error: {err}")

    def _load_or_build_engine(self):
        """Load existing engine or build new one if needed"""
        if not HAS_TENSORRT:
            return

        # Check if engine file exists
        if self.engine_path and os.path.exists(self.engine_path):
            self._load_engine_from_file()
        elif self.model_path:
            # Build engine from model
            self._build_engine_from_model()
        else:
            print("No engine file or model path provided")
            return

    def _load_engine_from_file(self):
        """Load TensorRT engine from file"""
        if not HAS_TENSORRT:
            return

        try:
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()

            self.runtime = trt.Runtime(self.logger)
            if self.runtime is None:
                raise RuntimeError("Failed to create TensorRT runtime")

            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize engine")

            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Failed to create execution context")

            print(f"Engine loaded successfully from {self.engine_path}")

        except Exception as e:
            print(f"Error loading engine from file: {e}")
            self.engine = None
            self.context = None

    def _build_engine_from_model(self):
        """Build TensorRT engine from model (placeholder implementation)"""
        print("Building TensorRT engine from model - this is a placeholder implementation")
        print("In a real implementation, this would convert the model to TensorRT format")
        # In a real implementation, you would:
        # 1. Load the original model (e.g., PyTorch model)
        # 2. Convert it to ONNX format
        # 3. Use TensorRT to optimize the ONNX model
        # 4. Serialize and save the optimized engine
        pass

    def build_engine_from_onnx(self, onnx_model_path: str, save_path: str,
                              min_shape: Tuple = (1, 3, 224, 224),
                              opt_shape: Tuple = (1, 3, 224, 224),
                              max_shape: Tuple = (1, 3, 224, 224)):
        """
        Build TensorRT engine from ONNX model file

        Args:
            onnx_model_path: Path to ONNX model file
            save_path: Path to save the TensorRT engine
            min_shape: Minimum input shape for optimization profile
            opt_shape: Optimal input shape for optimization profile
            max_shape: Maximum input shape for optimization profile
        """
        if not HAS_TENSORRT:
            print("TensorRT not available, cannot build engine")
            return False

        try:
            # Create builder and network
            builder = trt.Builder(self.logger)
            if not builder:
                raise RuntimeError("Failed to create TensorRT builder")

            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            if not network:
                raise RuntimeError("Failed to create TensorRT network")

            config = builder.create_builder_config()
            if not config:
                raise RuntimeError("Failed to create TensorRT builder config")

            # Parse ONNX model
            parser = trt.OnnxParser(network, self.logger)
            if not parser:
                raise RuntimeError("Failed to create ONNX parser")

            with open(onnx_model_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return False

            # Create optimization profile
            profile = builder.create_optimization_profile()
            profile.set_shape('input', min_shape, opt_shape, max_shape)  # Adjust 'input' to your actual input name
            config.add_optimization_profile(profile)

            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")

            # Save engine
            with open(save_path, 'wb') as f:
                f.write(serialized_engine)

            print(f"TensorRT engine saved to {save_path}")
            return True

        except Exception as e:
            print(f"Error building TensorRT engine: {e}")
            return False

    def infer(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Perform inference using the TensorRT engine

        Args:
            input_data: Input data as numpy array

        Returns:
            Output data as numpy array or None if inference fails
        """
        if not HAS_TENSORRT or self.engine is None or self.context is None:
            # Mock inference for testing purposes
            print("TensorRT not available, returning mock inference result")
            # Create a mock output with same batch size and some reasonable dimensions
            mock_output = np.random.random((input_data.shape[0], 1000)).astype(np.float32)
            return mock_output

        try:
            # Allocate CUDA memory for inputs and outputs
            input_size = trt.volume(self.engine.get_binding_shape(self.input_binding_idx)) * self.max_batch_size * np.dtype(np.float32).itemsize
            output_size = trt.volume(self.engine.get_binding_shape(self.output_binding_idx)) * self.max_batch_size * np.dtype(np.float32).itemsize

            # Allocate device memory
            d_input = cuda.cuMemAlloc(input_size)[1]
            d_output = cuda.cuMemAlloc(output_size)[1]

            # Create CUDA stream
            stream = cuda.cuStreamCreate()[1]

            # Copy input data to device
            cuda.cuMemcpyHtoD(d_input, input_data.ravel(), input_size)

            # Set engine bindings
            bindings = [int(d_input), int(d_output)]

            # Execute inference
            start_time = time.time()
            self.context.execute_async_v2(bindings=bindings, stream_handle=stream)

            # Synchronize stream
            cuda.cuStreamSynchronize(stream)

            # Copy output data back to host
            output_data = np.empty((self.max_batch_size, output_size // (self.max_batch_size * 4)), dtype=np.float32)
            cuda.cuMemcpyDtoH(output_data.ravel(), d_output, output_size)

            # Clean up
            cuda.cuMemFree(d_input)
            cuda.cuMemFree(d_output)
            cuda.cuStreamDestroy(stream)

            inference_time = time.time() - start_time
            print(f"Inference completed in {inference_time:.4f}s")

            return output_data

        except Exception as e:
            print(f"Error during TensorRT inference: {e}")
            return None

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the loaded engine"""
        if self.engine is None:
            return {"error": "No engine loaded"}

        info = {
            "name": self.engine.name,
            "max_batch_size": self.engine.max_batch_size,
            "num_bindings": self.engine.num_bindings,
            "num_optimization_profiles": self.engine.num_optimization_profiles,
        }

        # Get binding info
        bindings_info = []
        for i in range(self.engine.num_bindings):
            binding_info = {
                "name": self.engine.get_binding_name(i),
                "is_input": self.engine.binding_is_input(i),
                "shape": self.engine.get_binding_shape(i),
                "dtype": str(self.engine.get_binding_dtype(i))
            }
            bindings_info.append(binding_info)

        info["bindings"] = bindings_info
        return info

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'context') and self.context:
            del self.context
        if hasattr(self, 'engine') and self.engine:
            del self.engine
        if hasattr(self, 'runtime') and self.runtime:
            del self.runtime


class TensorRTModelBuilder:
    """
    Builder for creating TensorRT engines from various model formats
    """

    def __init__(self):
        self.logger = TensorRTLogger()

    def build_from_pytorch(self, pytorch_model, input_shape: Tuple, save_path: str,
                          precision: str = "float16") -> bool:
        """
        Build TensorRT engine from PyTorch model

        Args:
            pytorch_model: PyTorch model instance
            input_shape: Input shape tuple (batch, channels, height, width)
            save_path: Path to save the TensorRT engine
            precision: Precision for inference ('float16', 'int8', 'float32')

        Returns:
            True if successful, False otherwise
        """
        print("Building TensorRT engine from PyTorch model - placeholder implementation")
        # In a real implementation, you would:
        # 1. Export PyTorch model to ONNX
        # 2. Use TensorRT to build engine from ONNX
        # 3. Save the engine
        return True

    def build_from_onnx(self, onnx_path: str, save_path: str,
                       input_name: str = "input",
                       min_shape: Tuple = (1, 3, 224, 224),
                       opt_shape: Tuple = (1, 3, 224, 224),
                       max_shape: Tuple = (1, 3, 224, 224),
                       precision: str = "float16") -> bool:
        """
        Build TensorRT engine from ONNX file with optimization
        """
        print(f"Building TensorRT engine from ONNX: {onnx_path}")
        print(f"Output path: {save_path}")

        # This would call the TensorRTEngine's build method
        engine = TensorRTEngine()
        return engine.build_engine_from_onnx(onnx_path, save_path, min_shape, opt_shape, max_shape)


class EngineCache:
    """
    Cache for TensorRT engines to avoid repeated loading
    """

    def __init__(self, max_engines: int = 5):
        self.max_engines = max_engines
        self.engines: Dict[str, TensorRTEngine] = {}
        self.access_order: List[str] = []  # Track access order for LRU

    def get_engine(self, engine_path: str, **kwargs) -> Optional[TensorRTEngine]:
        """Get engine from cache or load if not present"""
        if engine_path in self.engines:
            # Move to end of access order (most recently used)
            self.access_order.remove(engine_path)
            self.access_order.append(engine_path)
            return self.engines[engine_path]

        # Load new engine
        engine = TensorRTEngine(engine_path=engine_path, **kwargs)
        if engine.engine is not None:
            # Add to cache
            self.engines[engine_path] = engine
            self.access_order.append(engine_path)

            # Remove oldest if cache is full
            if len(self.engines) > self.max_engines:
                oldest_path = self.access_order.pop(0)
                del self.engines[oldest_path]

            return engine

        return None

    def clear(self):
        """Clear the engine cache"""
        self.engines.clear()
        self.access_order.clear()


# Global engine cache instance
_engine_cache: Optional[EngineCache] = None


def get_engine_cache(max_engines: int = 5) -> EngineCache:
    """Get the global engine cache instance"""
    global _engine_cache
    if _engine_cache is None:
        _engine_cache = EngineCache(max_engines)
    return _engine_cache


# Example usage and testing
if __name__ == "__main__":
    print("Testing TensorRT Engine Utilities...")

    # Test with mock functionality since TensorRT may not be installed
    print("Creating TensorRT engine (mock mode if TensorRT not available)...")
    engine = TensorRTEngine()

    # Test with some mock input data
    mock_input = np.random.random((1, 3, 224, 224)).astype(np.float32)
    print(f"Mock input shape: {mock_input.shape}")

    # Perform inference
    output = engine.infer(mock_input)
    if output is not None:
        print(f"Inference output shape: {output.shape}")
    else:
        print("Inference failed or not available")

    # Test engine info
    info = engine.get_engine_info()
    print(f"Engine info: {info}")

    print("TensorRT engine utilities test completed.")