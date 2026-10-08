import math
from .buffer import AudioBuffer, DSPStage

class AGC(DSPStage):
    """
    Automatic Gain Control (AGC) using a feed-forward envelope detector.
    """
    def __init__(self, target_rms: float = 4000.0, max_gain: float = 10.0, min_gain: float = 0.1, 
                 attack_time: float = 0.05, release_time: float = 0.2, 
                 noise_threshold: float = 100.0, sample_rate: float = 16000.0, 
                 enabled: bool = True):
        super().__init__(enabled)
        self.target_rms = target_rms
        self.max_gain = max_gain
        self.min_gain = min_gain
        self.noise_threshold = noise_threshold
        
        # Calculate coefficients
        # alpha = exp(-1 / (fs * t))
        self.alpha_attack = math.exp(-1.0 / (sample_rate * attack_time))
        self.alpha_release = math.exp(-1.0 / (sample_rate * release_time))
        
        self.envelope = 0.0
        self.current_gain = 1.0

    def _process_impl(self, buffer: AudioBuffer) -> AudioBuffer:
        out = []
        for x in buffer.data:
            abs_x = abs(x)
            
            # Envelope detection
            if abs_x > self.envelope:
                self.envelope = self.alpha_attack * self.envelope + (1.0 - self.alpha_attack) * abs_x
            else:
                self.envelope = self.alpha_release * self.envelope + (1.0 - self.alpha_release) * abs_x
            
            # Target gain calculation
            if self.envelope < self.noise_threshold:
                desired_gain = 1.0 # Do not amplify silence/noise
            else:
                # RMS is approximately envelope / sqrt(2) for sine, but we simplify
                desired_gain = self.target_rms / max(self.envelope, 1.0)
                
            # Clamp desired gain
            desired_gain = max(self.min_gain, min(self.max_gain, desired_gain))
            
            # Smooth the gain applied (optional, but helps avoid clicking)
            # For this simple AGC, we just use the smoothed envelope's desired gain directly
            self.current_gain = desired_gain
            
            # Apply gain
            y = x * self.current_gain
            
            # Hard clip safety (AGC should ideally prevent this, but it's a safety net)
            if y > 32767.0:
                y = 32767.0
            elif y < -32767.0:
                y = -32767.0
                
            out.append(y)
            
        buffer.data = out
        return buffer
