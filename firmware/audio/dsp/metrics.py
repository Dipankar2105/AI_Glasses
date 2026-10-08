import math
from .buffer import AudioBuffer

def calculate_rms(buffer: AudioBuffer) -> float:
    if not buffer.data: return 0.0
    sq_sum = sum(x * x for x in buffer.data)
    return math.sqrt(sq_sum / len(buffer.data))

def calculate_peak(buffer: AudioBuffer) -> float:
    if not buffer.data: return 0.0
    return max(abs(x) for x in buffer.data)

def calculate_crest_factor(buffer: AudioBuffer) -> float:
    rms = calculate_rms(buffer)
    if rms == 0: return 0.0
    return calculate_peak(buffer) / rms

def calculate_dc_offset(buffer: AudioBuffer) -> float:
    if not buffer.data: return 0.0
    return sum(buffer.data) / len(buffer.data)

def calculate_clipping_percentage(buffer: AudioBuffer, threshold: float = 32767.0) -> float:
    if not buffer.data: return 0.0
    clipped = sum(1 for x in buffer.data if abs(x) >= threshold)
    return (clipped / len(buffer.data)) * 100.0
