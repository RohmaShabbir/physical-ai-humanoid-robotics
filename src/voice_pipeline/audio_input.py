"""
Audio Input Handler for the Vision-Language-Action System
Handles microphone input and audio preprocessing for speech recognition
"""

import threading
import time
import queue
from typing import Optional, Callable, List
import numpy as np

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    print("Warning: pyaudio not installed. Install with 'pip install pyaudio'")
    HAS_PYAUDIO = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    print("Warning: sounddevice not installed. Install with 'pip install sounddevice'")
    HAS_SOUNDDEVICE = False


class AudioInput:
    """
    Audio input handler for capturing audio from microphone
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 device_index: Optional[int] = None,
                 threshold: int = 400,
                 silence_duration: float = 1.0):
        """
        Initialize the audio input handler

        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Size of audio chunks to read
            device_index: Index of audio input device (None for default)
            threshold: Threshold for voice activity detection
            silence_duration: Duration of silence to stop recording (seconds)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.threshold = threshold
        self.silence_duration = silence_duration

        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.audio_queue = queue.Queue()
        self.audio_callback: Optional[Callable[[np.ndarray], None]] = None

        # Voice activity detection
        self.vad_enabled = True
        self.silence_start_time = None
        self.is_speaking = False

        # Audio stream
        self.stream = None
        self.audio_interface = None

        # Initialize audio interface
        self._initialize_audio_interface()

    def _initialize_audio_interface(self):
        """Initialize the audio interface based on available libraries"""
        if HAS_SOUNDDEVICE:
            self.audio_interface = "sounddevice"
            print("Using sounddevice for audio input")
        elif HAS_PYAUDIO:
            self.audio_interface = "pyaudio"
            self._init_pyaudio()
            print("Using pyaudio for audio input")
        else:
            print("Warning: No audio library available. Audio input will not work.")
            self.audio_interface = None

    def _init_pyaudio(self):
        """Initialize PyAudio interface"""
        if not HAS_PYAUDIO:
            return

        try:
            self.pyaudio_instance = pyaudio.PyAudio()
        except Exception as e:
            print(f"Error initializing PyAudio: {e}")
            self.pyaudio_instance = None

    def set_audio_callback(self, callback: Callable[[np.ndarray], None]):
        """Set a callback function to receive audio chunks"""
        self.audio_callback = callback

    def start_recording(self):
        """Start recording audio from the microphone"""
        if self.is_recording:
            return

        if self.audio_interface is None:
            print("Cannot start recording: No audio interface available")
            return

        self.is_recording = True
        self.silence_start_time = None
        self.is_speaking = False

        if self.audio_interface == "sounddevice":
            self._start_recording_sounddevice()
        elif self.audio_interface == "pyaudio":
            self._start_recording_pyaudio()

    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False

        if self.audio_interface == "sounddevice":
            if self.stream and self.stream.active:
                self.stream.stop()
                self.stream.close()
        elif self.audio_interface == "pyaudio" and self.pyaudio_instance:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()

    def _start_recording_sounddevice(self):
        """Start recording using sounddevice"""
        def audio_callback(indata, frames, time, status):
            if not self.is_recording:
                return

            # Convert to numpy array and flatten
            audio_data = indata[:, 0].copy()  # Take first channel if stereo

            # Voice activity detection
            if self.vad_enabled:
                rms = np.sqrt(np.mean(audio_data ** 2))
                is_speech = rms > self.threshold

                if is_speech:
                    self.is_speaking = True
                    self.silence_start_time = None
                else:
                    if self.is_speaking:
                        if self.silence_start_time is None:
                            self.silence_start_time = time.currentTime
                        elif (time.currentTime - self.silence_start_time) > self.silence_duration:
                            # Extended silence detected, could trigger end of speech
                            self.is_speaking = False
            else:
                is_speech = True  # If VAD disabled, treat all audio as speech

            if is_speech or not self.vad_enabled:
                if self.audio_callback:
                    self.audio_callback(audio_data)

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype='float32',
                device=self.device_index,
                callback=audio_callback
            )
            self.stream.start()
        except Exception as e:
            print(f"Error starting audio stream with sounddevice: {e}")

    def _start_recording_pyaudio(self):
        """Start recording using PyAudio"""
        if not HAS_PYAUDIO or not self.pyaudio_instance:
            return

        def audio_callback_pyaudio(in_data, frame_count, time_info, status):
            if not self.is_recording:
                return (None, pyaudio.paAbort)

            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Voice activity detection
            if self.vad_enabled:
                rms = np.sqrt(np.mean(audio_data ** 2))
                is_speech = rms > (self.threshold / 32768.0)  # Adjust threshold for float32

                if is_speech:
                    self.is_speaking = True
                    self.silence_start_time = None
                else:
                    if self.is_speaking:
                        current_time = time.time()
                        if self.silence_start_time is None:
                            self.silence_start_time = current_time
                        elif (current_time - self.silence_start_time) > self.silence_duration:
                            # Extended silence detected
                            self.is_speaking = False
            else:
                is_speech = True

            if is_speech or not self.vad_enabled:
                if self.audio_callback:
                    self.audio_callback(audio_data)

            return (None, pyaudio.paContinue)

        try:
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=audio_callback_pyaudio,
                input_device_index=self.device_index
            )
            self.stream.start_stream()
        except Exception as e:
            print(f"Error starting audio stream with PyAudio: {e}")

    def get_available_devices(self) -> List[Dict]:
        """Get a list of available audio input devices"""
        devices = []

        if self.audio_interface == "sounddevice":
            devices_info = sd.query_devices()
            for i, device in enumerate(devices_info):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device['name'],
                        'max_input_channels': device['max_input_channels'],
                        'default_samplerate': device['default_samplerate']
                    })
        elif self.audio_interface == "pyaudio" and self.pyaudio_instance:
            for i in range(self.pyaudio_instance.get_device_count()):
                device_info = self.pyaudio_instance.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'max_input_channels': device_info['maxInputChannels'],
                        'default_samplerate': device_info['defaultSampleRate']
                    })

        return devices

    def is_active(self) -> bool:
        """Check if the audio input is currently active"""
        if not self.is_recording:
            return False

        if self.audio_interface == "sounddevice":
            return self.stream is not None and self.stream.active
        elif self.audio_interface == "pyaudio":
            return (self.stream is not None and
                   self.pyaudio_instance is not None and
                   self.stream.is_active())
        return False

    def __del__(self):
        """Cleanup audio resources"""
        self.stop_recording()
        if self.audio_interface == "pyaudio" and self.pyaudio_instance:
            self.pyaudio_instance.terminate()


class AudioPreprocessor:
    """
    Audio preprocessing for speech recognition
    """

    def __init__(self):
        """Initialize the audio preprocessor"""
        pass

    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio data to have values between -1 and 1
        """
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data

    def apply_gain(self, audio_data: np.ndarray, gain: float = 1.0) -> np.ndarray:
        """
        Apply gain to audio data
        """
        return audio_data * gain

    def high_pass_filter(self, audio_data: np.ndarray,
                        cutoff_freq: float = 100.0,
                        sample_rate: int = 16000) -> np.ndarray:
        """
        Apply a simple high-pass filter to remove low-frequency noise
        """
        # Simple first-order high-pass filter
        # y[n] = x[n] - x[n-1] + 0.995 * y[n-1]
        filtered = np.zeros_like(audio_data)
        if len(audio_data) == 0:
            return filtered

        filtered[0] = audio_data[0]
        for i in range(1, len(audio_data)):
            filtered[i] = audio_data[i] - audio_data[i-1] + 0.995 * filtered[i-1]

        return filtered

    def noise_gate(self, audio_data: np.ndarray,
                   threshold: float = 0.01,
                   sample_rate: int = 16000) -> np.ndarray:
        """
        Apply a noise gate to suppress quiet sounds
        """
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < threshold:
            # If below threshold, return silence
            return np.zeros_like(audio_data)
        return audio_data

    def preprocess_audio(self, audio_data: np.ndarray,
                        sample_rate: int = 16000) -> np.ndarray:
        """
        Apply a series of preprocessing steps to audio data
        """
        # Normalize
        processed = self.normalize_audio(audio_data)

        # Apply high-pass filter to remove low-frequency noise
        processed = self.high_pass_filter(processed, sample_rate=sample_rate)

        # Apply noise gate
        processed = self.noise_gate(processed, sample_rate=sample_rate)

        return processed

    def resample_audio(self, audio_data: np.ndarray,
                      original_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio from original sample rate to target sample rate
        Note: This is a simple implementation; for production use, consider using librosa
        """
        if original_sr == target_sr:
            return audio_data

        # Calculate the new length
        original_length = len(audio_data)
        new_length = int(original_length * target_sr / original_sr)

        # Simple resampling using numpy
        time_original = np.linspace(0, 1, original_length)
        time_new = np.linspace(0, 1, new_length)
        resampled = np.interp(time_new, time_original, audio_data)

        return resampled


class AudioBuffer:
    """
    Audio buffer for accumulating audio chunks
    """

    def __init__(self, max_duration: float = 30.0, sample_rate: int = 16000):
        """
        Initialize audio buffer

        Args:
            max_duration: Maximum duration in seconds to keep in buffer
            sample_rate: Sample rate of audio data
        """
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()

    def add_audio_chunk(self, chunk: np.ndarray):
        """Add an audio chunk to the buffer"""
        with self.lock:
            # Append new chunk
            self.buffer = np.concatenate([self.buffer, chunk])

            # Keep only the most recent audio within max duration
            if len(self.buffer) > self.max_samples:
                excess_samples = len(self.buffer) - self.max_samples
                self.buffer = self.buffer[excess_samples:]

    def get_audio_segment(self, duration: float = 5.0) -> np.ndarray:
        """
        Get the most recent audio segment of specified duration
        """
        with self.lock:
            required_samples = int(duration * self.sample_rate)
            if len(self.buffer) == 0:
                return np.array([], dtype=np.float32)

            if len(self.buffer) <= required_samples:
                return self.buffer.copy()
            else:
                return self.buffer[-required_samples:].copy()

    def clear(self):
        """Clear the audio buffer"""
        with self.lock:
            self.buffer = np.array([], dtype=np.float32)

    def get_duration(self) -> float:
        """Get the current duration of buffered audio"""
        with self.lock:
            return len(self.buffer) / self.sample_rate if self.sample_rate > 0 else 0.0

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self.lock:
            return len(self.buffer) == 0


# Example usage and testing
if __name__ == "__main__":
    print("Testing Audio Input Handler...")

    # Create audio input
    audio_input = AudioInput(
        sample_rate=16000,
        chunk_size=1024,
        threshold=400,
        silence_duration=1.0
    )

    # Print available devices
    devices = audio_input.get_available_devices()
    print(f"Available audio devices: {len(devices)}")
    for device in devices[:5]:  # Show first 5 devices
        print(f"  {device['index']}: {device['name']}")

    # Test audio preprocessing
    preprocessor = AudioPreprocessor()
    test_audio = np.random.random(1000).astype(np.float32)
    processed_audio = preprocessor.preprocess_audio(test_audio)
    print(f"Original audio shape: {test_audio.shape}")
    print(f"Processed audio shape: {processed_audio.shape}")

    # Test audio buffer
    buffer = AudioBuffer(max_duration=10.0, sample_rate=16000)
    test_chunk = np.random.random(1600).astype(np.float32)  # 0.1 seconds of audio
    buffer.add_audio_chunk(test_chunk)
    segment = buffer.get_audio_segment(duration=0.05)  # 50ms segment
    print(f"Retrieved segment shape: {segment.shape}")

    print("Audio input handler test completed.")