from .buffer import AudioBuffer, DSPStage

class DCBlocker(DSPStage):
    """
    Removes DC offset using a simple first-order IIR high-pass filter.
    y[n] = x[n] - x[n-1] + R * y[n-1]
    """
    def __init__(self, r: float = 0.995, enabled: bool = True):
        super().__init__(enabled)
        self.r = r
        self.prev_x = 0.0
        self.prev_y = 0.0

    def _process_impl(self, buffer: AudioBuffer) -> AudioBuffer:
        out = []
        for x in buffer.data:
            y = x - self.prev_x + self.r * self.prev_y
            out.append(y)
            self.prev_x = x
            self.prev_y = y
        buffer.data = out
        return buffer

class HighPassFilter(DSPStage):
    """
    Simple first-order high-pass filter.
    alpha = RC / (RC + dt)
    y[n] = alpha * (y[n-1] + x[n] - x[n-1])
    """
    def __init__(self, cutoff_freq: float = 80.0, sample_rate: float = 16000.0, enabled: bool = True):
        super().__init__(enabled)
        import math
        dt = 1.0 / sample_rate
        rc = 1.0 / (2.0 * math.pi * cutoff_freq)
        self.alpha = rc / (rc + dt)
        self.prev_x = 0.0
        self.prev_y = 0.0

    def _process_impl(self, buffer: AudioBuffer) -> AudioBuffer:
        out = []
        for x in buffer.data:
            y = self.alpha * (self.prev_y + x - self.prev_x)
            out.append(y)
            self.prev_x = x
            self.prev_y = y
        buffer.data = out
        return buffer
