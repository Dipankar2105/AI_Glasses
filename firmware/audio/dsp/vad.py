import math
from .buffer import AudioBuffer, DSPStage

class VAD(DSPStage):
    """
    Lightweight deterministic energy-based Voice Activity Detection (VAD).
    Operates on frames and uses temporal smoothing (hangover).
    """
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 10,
                 energy_threshold: float = 500.0,
                 attack_frames: int = 1, hangover_frames: int = 20,
                 enabled: bool = True):
        super().__init__(enabled)
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.energy_threshold = energy_threshold
        
        self.attack_frames = attack_frames
        self.hangover_frames = hangover_frames
        
        self.active_count = 0
        self.inactive_count = 0
        self.speech_active = False
        
        # Diagnostic history
        self.history = []

    def _process_impl(self, buffer: AudioBuffer) -> AudioBuffer:
        # We will not mutate the audio data, VAD acts as an analyzer here.
        # Alternatively it could be used as a gate, but for evaluation we just want states.
        out_data = list(buffer.data)
        
        self.history = []
        n_frames = len(out_data) // self.frame_size
        
        for i in range(n_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = out_data[start:end]
            
            # Calculate RMS energy for the frame
            sq_sum = sum(x * x for x in frame)
            rms = math.sqrt(sq_sum / self.frame_size) if self.frame_size > 0 else 0
            
            raw_active = rms > self.energy_threshold
            
            # Temporal logic
            if raw_active:
                self.active_count += 1
                self.inactive_count = 0
                if self.active_count >= self.attack_frames:
                    self.speech_active = True
            else:
                self.inactive_count += 1
                self.active_count = 0
                if self.inactive_count >= self.hangover_frames:
                    self.speech_active = False
                    
            self.history.append({
                "frame": i,
                "rms": rms,
                "raw_active": raw_active,
                "speech_active": self.speech_active
            })
            
        return buffer
