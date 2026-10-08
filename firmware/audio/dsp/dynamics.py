from .buffer import AudioBuffer, DSPStage

class Gain(DSPStage):
    """Applies a constant linear gain to the signal."""
    def __init__(self, gain_factor: float = 1.0, enabled: bool = True):
        super().__init__(enabled)
        self.gain_factor = gain_factor

    def _process_impl(self, buffer: AudioBuffer) -> AudioBuffer:
        buffer.data = [x * self.gain_factor for x in buffer.data]
        return buffer

class Limiter(DSPStage):
    """
    Hard clipper/limiter.
    Ensures signal stays within [-threshold, threshold].
    """
    def __init__(self, threshold: float = 32767.0, enabled: bool = True):
        super().__init__(enabled)
        self.threshold = threshold

    def _process_impl(self, buffer: AudioBuffer) -> AudioBuffer:
        out = []
        for x in buffer.data:
            if x > self.threshold:
                out.append(self.threshold)
            elif x < -self.threshold:
                out.append(-self.threshold)
            else:
                out.append(x)
        buffer.data = out
        return buffer
