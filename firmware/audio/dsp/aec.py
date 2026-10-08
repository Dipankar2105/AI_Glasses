import math
from .buffer import AudioBuffer, DSPStage

class NLMSAEC(DSPStage):
    """
    Normalized Least Mean Squares (NLMS) Acoustic Echo Cancellation.
    """
    def __init__(self, filter_length: int = 256, step_size: float = 0.5, 
                 regularization: float = 1e6, enabled: bool = True):
        super().__init__(enabled)
        self.filter_length = filter_length
        self.mu = step_size
        self.epsilon = regularization
        self.weights = [0.0] * filter_length
        self.x_hist = [0.0] * filter_length

    def _process_impl(self, buffer: AudioBuffer) -> AudioBuffer:
        # Standard DSPStage process isn't natively suitable for a 2-input AEC.
        # This will return the unmodified buffer to satisfy the base class safely.
        # Use `process_aec` instead.
        return buffer

    def process_aec(self, mic_buffer: AudioBuffer, ref_buffer: AudioBuffer) -> AudioBuffer:
        """
        Process the microphone signal using the reference (speaker) signal.
        mic_buffer: The signal from the microphone (near-end + echo + noise)
        ref_buffer: The signal sent to the speaker (reference)
        """
        if not self.enabled:
            return AudioBuffer(list(mic_buffer.data))

        mic_data = mic_buffer.data
        ref_data = ref_buffer.data
        N = min(len(mic_data), len(ref_data))
        out_data = [0.0] * N
        
        for i in range(N):
            # Shift reference history
            self.x_hist.pop()
            self.x_hist.insert(0, ref_data[i])
            
            # Calculate estimated echo: y(n) = w^T * x(n)
            y = sum(w * x for w, x in zip(self.weights, self.x_hist))
            
            # Calculate error (which is our AEC output)
            e = mic_data[i] - y
            out_data[i] = e
            
            # Calculate reference power: ||x(n)||^2
            power = sum(x * x for x in self.x_hist)
            
            # Update filter coefficients: w(n+1) = w(n) + [mu / (power + epsilon)] * e(n) * x(n)
            norm_mu = self.mu / (power + self.epsilon)
            
            # Double-talk detection (DTD) is mathematically omitted in this basic NLMS,
            # but setting mu lower or tracking cross-correlation can simulate it.
            # Here we use standard NLMS.
            for j in range(self.filter_length):
                self.weights[j] += norm_mu * e * self.x_hist[j]
                
        return AudioBuffer(out_data)
