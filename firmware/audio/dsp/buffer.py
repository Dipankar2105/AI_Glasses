class AudioBuffer:
    """
    Represents a block of PCM audio data.
    PROVISIONAL DEFAULTS:
    - sample_rate: 16000 Hz
    - bit_depth: 16-bit
    - channels: 1 (mono)
    - representation: signed integer (int16)
    """
    def __init__(self, data: list, sample_rate: int = 16000):
        self.data = list(data)  # Keep as list of floats for processing
        self.sample_rate = sample_rate
        self.channels = 1

    def copy(self):
        return AudioBuffer(list(self.data), self.sample_rate)

class DSPStage:
    """Base class for all DSP stages."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def process(self, buffer: AudioBuffer) -> AudioBuffer:
        if not self.enabled:
            return buffer
        return self._process_impl(buffer)

    def _process_impl(self, buffer: AudioBuffer) -> AudioBuffer:
        raise NotImplementedError("DSPStage must implement _process_impl")
