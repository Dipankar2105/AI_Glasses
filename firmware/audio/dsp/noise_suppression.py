import math
import cmath
import time
from .buffer import AudioBuffer, DSPStage

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def ifft(x):
    N = len(x)
    x_conj = [c.conjugate() for c in x]
    X_conj = fft(x_conj)
    return [(c.conjugate() / N).real for c in X_conj]

class SpectralNoiseSuppression(DSPStage):
    """
    Classical Spectral Subtraction noise suppression.
    Uses pure-Python FFT for evaluation.
    """
    def __init__(self, frame_size: int = 256, overlap_factor: int = 2, 
                 noise_estimation_frames: int = 10, alpha: float = 2.0, 
                 spectral_floor: float = 0.05, enabled: bool = True):
        super().__init__(enabled)
        # Force frame_size to power of 2 for pure python FFT
        self.frame_size = next_power_of_2(frame_size)
        self.hop_size = self.frame_size // overlap_factor
        self.noise_estimation_frames = noise_estimation_frames
        self.alpha = alpha
        self.spectral_floor = spectral_floor
        
        # Hann window
        self.window = [0.5 * (1 - math.cos(2 * math.pi * i / (self.frame_size - 1))) for i in range(self.frame_size)]
        
        self.noise_profile = [0.0] * self.frame_size
        self.is_profile_estimated = False
        
        # Performance tracking
        self.processing_times = []

    def _process_impl(self, buffer: AudioBuffer) -> AudioBuffer:
        data = list(buffer.data)
        N = len(data)
        if N < self.frame_size:
            return buffer
        out_data = [0.0] * N
        
        frames_processed = 0
        
        start_idx = 0
        while start_idx + self.frame_size <= N:
            t0 = time.perf_counter()
            
            # Extract frame and window
            frame = data[start_idx : start_idx + self.frame_size]
            windowed = [frame[i] * self.window[i] for i in range(self.frame_size)]
            
            # FFT
            complex_spec = fft([complex(x, 0) for x in windowed])
            mags = [abs(c) for c in complex_spec]
            phases = [cmath.phase(c) for c in complex_spec]
            
            # Noise estimation (first N frames)
            if not self.is_profile_estimated:
                for i in range(self.frame_size):
                    self.noise_profile[i] += mags[i] / self.noise_estimation_frames
                
                frames_processed += 1
                if frames_processed >= self.noise_estimation_frames:
                    self.is_profile_estimated = True
                    
                # During estimation, just pass through (or output silence). We pass through.
                out_frame = frame 
            else:
                # Spectral subtraction
                out_mags = [0.0] * self.frame_size
                for i in range(self.frame_size):
                    # Power/Magnitude subtraction
                    sub = mags[i] - self.alpha * self.noise_profile[i]
                    out_mags[i] = max(sub, mags[i] * self.spectral_floor)
                    
                # IFFT
                out_complex = [cmath.rect(out_mags[i], phases[i]) for i in range(self.frame_size)]
                out_frame = ifft(out_complex)
                
            # Overlap-add
            for i in range(self.frame_size):
                out_data[start_idx + i] += out_frame[i]
                
            t1 = time.perf_counter()
            self.processing_times.append(t1 - t0)
            
            start_idx += self.hop_size
            
        # Optional: scale output by overlap factor to normalize window gain
        # For Hann with 50% overlap, sum of windows is constant 1.0 (excluding boundaries)
        
        buffer.data = out_data
        return buffer
