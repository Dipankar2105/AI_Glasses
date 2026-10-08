from typing import List, Optional
from .core import HALDevice, HALError, HALState

class AudioCaptureFrame:
    """Hardware-independent representation of captured audio."""
    def __init__(self, pcm_data: List[float], timestamp: int, seq_num: int):
        self.pcm_data = pcm_data
        self.timestamp = timestamp
        self.seq_num = seq_num

class AudioPlaybackFrame:
    """Hardware-independent representation of audio to be played."""
    def __init__(self, pcm_data: List[float], timestamp: int, seq_num: int):
        self.pcm_data = pcm_data
        self.timestamp = timestamp
        self.seq_num = seq_num

class AudioHAL(HALDevice):
    """
    Audio Hardware Abstraction Layer interface.
    Contract: 16000 Hz, 16-bit signed PCM, Mono.
    """
    def __init__(self):
        super().__init__()
        # Mock buffers for testing
        self._mock_capture_queue = []
        self._mock_playback_queue = []
        
    def read_capture_frame(self) -> tuple[HALError, Optional[AudioCaptureFrame]]:
        """Reads a frame from the microphone."""
        if self.state != HALState.RUNNING:
            return HALError.NOT_INITIALIZED, None
        
        if not self._mock_capture_queue:
            return HALError.BUFFER_UNAVAILABLE, None
            
        return HALError.OK, self._mock_capture_queue.pop(0)

    def write_playback_frame(self, frame: AudioPlaybackFrame) -> HALError:
        """Submits a frame to the speaker for playback."""
        if self.state != HALState.RUNNING:
            return HALError.NOT_INITIALIZED
            
        self._mock_playback_queue.append(frame)
        return HALError.OK
        
    def read_reference_frame(self) -> tuple[HALError, Optional[AudioPlaybackFrame]]:
        """
        Retrieves the exact playback frame that corresponds chronologically 
        to the next capture frame. Used exclusively by the AEC.
        """
        if self.state != HALState.RUNNING:
            return HALError.NOT_INITIALIZED, None
            
        if not self._mock_playback_queue:
            return HALError.BUFFER_UNAVAILABLE, None
            
        return HALError.OK, self._mock_playback_queue.pop(0)
