import math
from .buffer import AudioBuffer
from .aec import NLMSAEC

class GeigelDTD:
    """
    Deterministic Geigel Double-Talk Detector.
    Compares the current microphone amplitude to the maximum reference amplitude 
    over the echo tail length.
    """
    def __init__(self, filter_length: int = 256, threshold: float = 0.5, hold_time_samples: int = 800):
        self.filter_length = filter_length
        self.threshold = threshold
        self.hold_time_samples = hold_time_samples
        
        self.hold_counter = 0
        self.ref_history = [0.0] * filter_length
        
    def check(self, mic_sample: float, ref_sample: float) -> bool:
        self.ref_history.pop()
        self.ref_history.insert(0, abs(ref_sample))
        
        # Max of reference over the tail length
        max_ref = max(self.ref_history)
        
        # Geigel logic: if near-end energy is high relative to reference history, 
        # it's highly likely near-end speech is present (double talk).
        if abs(mic_sample) >= self.threshold * max_ref:
            self.hold_counter = self.hold_time_samples
            
        if self.hold_counter > 0:
            self.hold_counter -= 1
            return True
        return False

class NLMSAECWithDTD(NLMSAEC):
    """
    NLMS AEC extended with a Geigel Double-Talk Detector.
    Inherits from the baseline NLMSAEC without modifying it.
    """
    def __init__(self, filter_length: int = 256, step_size: float = 0.5, 
                 regularization: float = 1e6, enabled: bool = True,
                 dtd_threshold: float = 0.5, dtd_hold: int = 800):
        super().__init__(filter_length, step_size, regularization, enabled)
        self.dtd = GeigelDTD(filter_length, dtd_threshold, dtd_hold)
        
        # Tracking for metrics
        self.dt_detected_frames = 0
        
    def process_aec(self, mic_buffer: AudioBuffer, ref_buffer: AudioBuffer) -> AudioBuffer:
        if not self.enabled:
            return AudioBuffer(list(mic_buffer.data))

        mic_data = mic_buffer.data
        ref_data = ref_buffer.data
        N = min(len(mic_data), len(ref_data))
        out_data = [0.0] * N
        
        dt_count = 0
        
        for i in range(N):
            # Shift reference history
            self.x_hist.pop()
            self.x_hist.insert(0, ref_data[i])
            
            # Check DTD
            is_dt = self.dtd.check(mic_data[i], ref_data[i])
            if is_dt:
                dt_count += 1
            
            # Calculate estimated echo: y(n) = w^T * x(n)
            y = sum(w * x for w, x in zip(self.weights, self.x_hist))
            
            # Calculate error
            e = mic_data[i] - y
            out_data[i] = e
            
            # Only adapt if NO double-talk
            if not is_dt:
                power = sum(x * x for x in self.x_hist)
                norm_mu = self.mu / (power + self.epsilon)
                for j in range(self.filter_length):
                    self.weights[j] += norm_mu * e * self.x_hist[j]
                    
        self.dt_detected_frames = dt_count
        return AudioBuffer(out_data)
