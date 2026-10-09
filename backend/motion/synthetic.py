import math
import numpy as np
from typing import List

from backend.motion.contracts import IMUSample, TouchSample

def generate_stationary_trace(
    duration_s: float = 1.0,
    sample_rate_hz: float = 50.0,
    noise_std: float = 0.01
) -> List[IMUSample]:
    """Generates synthetic stationary IMU data with zero-mean Gaussian noise."""
    samples = []
    num_samples = int(duration_s * sample_rate_hz)
    dt = 1.0 / sample_rate_hz

    for i in range(num_samples):
        t = i * dt
        samples.append(IMUSample(
            timestamp=t,
            ax=float(np.random.normal(0.0, noise_std)),
            ay=float(np.random.normal(0.0, noise_std)),
            az=float(1.0 + np.random.normal(0.0, noise_std)), # ~1g gravity on Z
            gx=float(np.random.normal(0.0, noise_std * 5.0)),
            gy=float(np.random.normal(0.0, noise_std * 5.0)),
            gz=float(np.random.normal(0.0, noise_std * 5.0)),
        ))
    return samples


def generate_nod_trace(
    duration_s: float = 1.0,
    sample_rate_hz: float = 50.0,
    nod_peak_deg_s: float = 60.0
) -> List[IMUSample]:
    """
    Generates a deterministic head nod motion trace (pitch oscillation on gy).
    """
    samples = []
    num_samples = int(duration_s * sample_rate_hz)
    dt = 1.0 / sample_rate_hz

    for i in range(num_samples):
        t = i * dt
        # Bi-phasic sine wave centered around t = 0.3s to 0.7s
        if 0.25 <= t <= 0.65:
            # 2.5 Hz full cycle
            phase = 2.0 * math.pi * 2.5 * (t - 0.25)
            gy = nod_peak_deg_s * math.sin(phase)
            # Accel vertical tilt component
            az = 1.0 - 0.15 * math.sin(phase)
            ay = 0.25 * math.sin(phase)
        else:
            gy = 0.0
            ay = 0.0
            az = 1.0

        samples.append(IMUSample(
            timestamp=t,
            ax=0.0,
            ay=ay,
            az=az,
            gx=0.0,
            gy=gy,
            gz=0.0
        ))
    return samples


def generate_shake_trace(
    duration_s: float = 1.0,
    sample_rate_hz: float = 50.0,
    shake_peak_deg_s: float = 70.0
) -> List[IMUSample]:
    """
    Generates a deterministic head shake motion trace (yaw oscillation on gz).
    """
    samples = []
    num_samples = int(duration_s * sample_rate_hz)
    dt = 1.0 / sample_rate_hz

    for i in range(num_samples):
        t = i * dt
        if 0.25 <= t <= 0.65:
            phase = 2.0 * math.pi * 2.5 * (t - 0.25)
            gz = shake_peak_deg_s * math.sin(phase)
            ax = 0.20 * math.sin(phase)
        else:
            gz = 0.0
            ax = 0.0

        samples.append(IMUSample(
            timestamp=t,
            ax=ax,
            ay=0.0,
            az=1.0,
            gx=0.0,
            gy=0.0,
            gz=gz
        ))
    return samples


def generate_slow_head_movement_trace(
    duration_s: float = 2.0,
    sample_rate_hz: float = 50.0
) -> List[IMUSample]:
    """
    Generates slow, gentle head rotation (< 15 deg/s) that should NOT trigger gestures.
    """
    samples = []
    num_samples = int(duration_s * sample_rate_hz)
    dt = 1.0 / sample_rate_hz

    for i in range(num_samples):
        t = i * dt
        # Slow 0.25 Hz sweep with low amplitude
        gz = 12.0 * math.sin(2.0 * math.pi * 0.25 * t)
        gy = 8.0 * math.cos(2.0 * math.pi * 0.25 * t)
        samples.append(IMUSample(
            timestamp=t,
            ax=0.02 * math.sin(t),
            ay=0.02 * math.cos(t),
            az=1.0,
            gx=0.0,
            gy=gy,
            gz=gz
        ))
    return samples


# --- Touch Trace Generators ---

def generate_touch_tap_trace(
    press_time_s: float = 0.1,
    hold_duration_s: float = 0.15,
    sample_rate_hz: float = 100.0
) -> List[TouchSample]:
    """Generates a single tap touch trace."""
    samples = []
    dt = 1.0 / sample_rate_hz
    for i in range(int(1.0 * sample_rate_hz)):
        t = i * dt
        is_pressed = press_time_s <= t <= (press_time_s + hold_duration_s)
        samples.append(TouchSample(timestamp=t, is_pressed=is_pressed))
    return samples


def generate_touch_double_tap_trace(
    first_tap_time_s: float = 0.1,
    tap_duration_s: float = 0.10,
    gap_duration_s: float = 0.12,
    sample_rate_hz: float = 100.0
) -> List[TouchSample]:
    """Generates a double tap touch trace."""
    samples = []
    dt = 1.0 / sample_rate_hz
    t1_end = first_tap_time_s + tap_duration_s
    t2_start = t1_end + gap_duration_s
    t2_end = t2_start + tap_duration_s

    for i in range(int(1.0 * sample_rate_hz)):
        t = i * dt
        is_pressed = (first_tap_time_s <= t <= t1_end) or (t2_start <= t <= t2_end)
        samples.append(TouchSample(timestamp=t, is_pressed=is_pressed))
    return samples


def generate_touch_long_press_trace(
    press_time_s: float = 0.1,
    hold_duration_s: float = 0.75,
    sample_rate_hz: float = 100.0
) -> List[TouchSample]:
    """Generates a long press touch trace."""
    samples = []
    dt = 1.0 / sample_rate_hz
    for i in range(int(1.2 * sample_rate_hz)):
        t = i * dt
        is_pressed = press_time_s <= t <= (press_time_s + hold_duration_s)
        samples.append(TouchSample(timestamp=t, is_pressed=is_pressed))
    return samples


def generate_touch_glitch_noise_trace(
    sample_rate_hz: float = 100.0
) -> List[TouchSample]:
    """Generates high frequency capacitive spikes (< 15ms) that should be filtered by debounce."""
    samples = []
    dt = 1.0 / sample_rate_hz
    for i in range(int(0.5 * sample_rate_hz)):
        t = i * dt
        # 10ms spike at t=0.1
        is_pressed = (0.10 <= t <= 0.11)
        samples.append(TouchSample(timestamp=t, is_pressed=is_pressed))
    return samples
