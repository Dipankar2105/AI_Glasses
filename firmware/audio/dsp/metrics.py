"""DSP numerical metrics for NextSight Audio subsystem."""

import math
from .buffer import AudioBuffer


def calculate_rms(buffer: AudioBuffer) -> float:
    """Calculate Root Mean Square (RMS) energy."""
    if not buffer.data:
        return 0.0
    sq_sum = sum(x * x for x in buffer.data)
    return math.sqrt(sq_sum / len(buffer.data))


def calculate_peak(buffer: AudioBuffer) -> float:
    """Calculate maximum absolute peak amplitude."""
    if not buffer.data:
        return 0.0
    return max(abs(x) for x in buffer.data)


def calculate_crest_factor(buffer: AudioBuffer) -> float:
    """Calculate Peak to RMS ratio."""
    rms = calculate_rms(buffer)
    if rms == 0.0:
        return 0.0
    return calculate_peak(buffer) / rms


def calculate_dc_offset(buffer: AudioBuffer) -> float:
    """Calculate mean DC offset."""
    if not buffer.data:
        return 0.0
    return sum(buffer.data) / len(buffer.data)


def calculate_clipping_percentage(buffer: AudioBuffer, threshold: float = 32767.0) -> float:
    """Calculate percentage of samples at or exceeding threshold."""
    if not buffer.data:
        return 0.0
    clipped = sum(1 for x in buffer.data if abs(x) >= threshold)
    return (clipped / len(buffer.data)) * 100.0


def calculate_snr(signal_buffer: AudioBuffer, noise_buffer: AudioBuffer) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) in dB.
    SNR = 10 * log10(P_signal / P_noise)
    """
    rms_s = calculate_rms(signal_buffer)
    rms_n = calculate_rms(noise_buffer)
    if rms_n <= 1e-12:
        return 999.0 if rms_s > 0 else 0.0
    if rms_s <= 1e-12:
        return -999.0
    return 20.0 * math.log10(rms_s / rms_n)


def calculate_erle(echo_mic_buffer: AudioBuffer, residual_buffer: AudioBuffer) -> float:
    """
    Calculate Echo Return Loss Enhancement (ERLE) in dB.
    ERLE = 20 * log10(RMS(echo_mic) / RMS(residual))
    """
    rms_echo = calculate_rms(echo_mic_buffer)
    rms_resid = calculate_rms(residual_buffer)
    if rms_resid <= 1e-12:
        return 999.0 if rms_echo > 0 else 0.0
    if rms_echo <= 1e-12:
        return 0.0
    return 20.0 * math.log10(rms_echo / rms_resid)
