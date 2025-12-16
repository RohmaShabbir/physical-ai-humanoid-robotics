"""
Speech Recognition Module for the Vision-Language-Action System
Implements Faster-Whisper for real-time speech recognition on Jetson hardware
"""

import os
import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
import threading
from dataclasses import dataclass

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    print("Warning: faster_whisper not installed. Install with 'pip install faster-whisper'")
    HAS_FASTER_WHISPER = False


@dataclass
class RecognitionResult:
    """Data class for speech recognition results"""
    text: str
    confidence: float
    language: str
    processing_time: float
    is_success: bool
    error_message: Optional[str] = None


class SpeechRecognition:
    """
    Speech recognition class using Faster-Whisper for real-time processing
    """

    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        """
        Initialize the speech recognition module

        Args:
            model_size: Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large-v3')
            device: Device to run the model on ('cuda' or 'cpu')
            compute_type: Compute type ('float16', 'int8', etc.)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None
        self.is_initialized = False
        self.initialization_lock = threading.Lock()

        # Performance tracking
        self.total_processing_time = 0.0
        self.processed_segments = 0

        # Load model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Whisper model"""
        if not HAS_FASTER_WHISPER:
            print("Cannot initialize speech recognition: faster_whisper not available")
            return

        try:
            start_time = time.time()
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                # Enable local download if not available
                download_root=os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            )
            init_time = time.time() - start_time
            print(f"Model {self.model_size} loaded in {init_time:.2f}s")
            self.is_initialized = True
        except Exception as e:
            print(f"Error initializing Whisper model: {e}")
            self.is_initialized = False

    def transcribe_audio(self, audio_data: np.ndarray,
                        language: Optional[str] = None,
                        beam_size: int = 5) -> RecognitionResult:
        """
        Transcribe audio data to text using the Whisper model

        Args:
            audio_data: Audio data as numpy array (should be in the right format for Whisper)
            language: Language code (e.g., 'en', 'es', etc.) or None for auto-detection
            beam_size: Beam size for decoding

        Returns:
            RecognitionResult containing the transcription and metadata
        """
        if not self.is_initialized or self.model is None:
            return RecognitionResult(
                text="",
                confidence=0.0,
                language="",
                processing_time=0.0,
                is_success=False,
                error_message="Model not initialized"
            )

        start_time = time.time()
        try:
            # Transcribe the audio
            segments, info = self.model.transcribe(
                audio_data,
                language=language,
                beam_size=beam_size,
                vad_filter=True  # Use voice activity detection
            )

            # Combine all segments into one text
            text = " ".join([segment.text for segment in segments])

            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.processed_segments += 1

            # Estimate confidence based on the transcription process
            # In a real implementation, we would use actual confidence scores
            confidence = min(0.95, len(text) / 10.0) if text else 0.0  # Simple heuristic

            return RecognitionResult(
                text=text.strip(),
                confidence=min(confidence, 1.0),
                language=info.language,
                processing_time=processing_time,
                is_success=True
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return RecognitionResult(
                text="",
                confidence=0.0,
                language="",
                processing_time=processing_time,
                is_success=False,
                error_message=str(e)
            )

    def transcribe_from_file(self, audio_file_path: str,
                           language: Optional[str] = None,
                           beam_size: int = 5) -> RecognitionResult:
        """
        Transcribe audio from a file

        Args:
            audio_file_path: Path to the audio file
            language: Language code or None for auto-detection
            beam_size: Beam size for decoding

        Returns:
            RecognitionResult containing the transcription and metadata
        """
        if not self.is_initialized or self.model is None:
            return RecognitionResult(
                text="",
                confidence=0.0,
                language="",
                processing_time=0.0,
                is_success=False,
                error_message="Model not initialized"
            )

        start_time = time.time()
        try:
            segments, info = self.model.transcribe(
                audio_file_path,
                language=language,
                beam_size=beam_size,
                vad_filter=True
            )

            text = " ".join([segment.text for segment in segments])
            processing_time = time.time() - start_time

            confidence = min(0.95, len(text) / 10.0) if text else 0.0

            return RecognitionResult(
                text=text.strip(),
                confidence=min(confidence, 1.0),
                language=info.language,
                processing_time=processing_time,
                is_success=True
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return RecognitionResult(
                text="",
                confidence=0.0,
                language="",
                processing_time=processing_time,
                is_success=False,
                error_message=str(e)
            )

    def get_average_processing_time(self) -> float:
        """Get the average processing time per transcription"""
        if self.processed_segments == 0:
            return 0.0
        return self.total_processing_time / self.processed_segments

    def warm_up(self) -> bool:
        """
        Warm up the model with a short audio sample to ensure it's ready
        This can help with initial latency issues
        """
        if not self.is_initialized:
            return False

        # Create a short silent audio array for warming up
        # 1 second of silence at 16kHz
        sample_rate = 16000
        silent_audio = np.zeros(sample_rate, dtype=np.float32)

        try:
            result = self.transcribe_audio(silent_audio)
            return result.is_success
        except:
            return False


class StreamingSpeechRecognition:
    """
    Streaming speech recognition for continuous audio input
    """

    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        self.speech_recognizer = SpeechRecognition(model_size, device, compute_type)
        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.audio_queue = []
        self.results_callback = None
        self.stream_lock = threading.Lock()

    def set_results_callback(self, callback_func):
        """Set a callback function to receive recognition results"""
        self.results_callback = callback_func

    def start_streaming(self):
        """Start the streaming recognition process"""
        if self.is_streaming:
            return

        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.stream_thread.start()

    def stop_streaming(self):
        """Stop the streaming recognition process"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()

    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """Add an audio chunk to the processing queue"""
        with self.stream_lock:
            self.audio_queue.append(audio_chunk)

    def _streaming_worker(self):
        """Internal worker for processing streaming audio"""
        while self.is_streaming:
            with self.stream_lock:
                if self.audio_queue:
                    audio_chunk = self.audio_queue.pop(0)
                else:
                    audio_chunk = None

            if audio_chunk is not None:
                result = self.speech_recognizer.transcribe_audio(audio_chunk)
                if self.results_callback and result.is_success and result.text.strip():
                    self.results_callback(result)

            # Small sleep to prevent busy waiting
            time.sleep(0.01)

    def is_active(self) -> bool:
        """Check if streaming is active"""
        return self.is_streaming


# Example usage and testing
if __name__ == "__main__":
    # Test the speech recognition module
    print("Initializing speech recognition...")
    recognizer = SpeechRecognition(model_size="large-v3", device="cuda", compute_type="float16")

    # Test with a silent audio array (in practice, you'd use real audio data)
    if recognizer.is_initialized:
        print("Model initialized successfully")

        # Warm up the model
        if recognizer.warm_up():
            print("Model warmed up successfully")
        else:
            print("Model warmup failed")
    else:
        print("Model initialization failed - using mock functionality")

        # Test with mock data
        import numpy as np
        mock_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = recognizer.transcribe_audio(mock_audio)
        print(f"Mock transcription result: {result}")