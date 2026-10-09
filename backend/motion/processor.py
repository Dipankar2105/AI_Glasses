import math
from typing import List, Optional, Tuple
from collections import deque

from backend.motion.contracts import IMUSample, HeadGestureType, GestureEvent

class MotionProcessor:
    """
    Deterministic real-time motion and gesture detection engine for 6-DOF IMU data.
    Features:
      - Exponential smoothing filter for high-frequency noise rejection.
      - Gravity-compensated dynamic acceleration estimation.
      - Stationary vs moving classification.
      - Pattern-based intentional gesture detection (Nod, Shake, Tilt).
      - Cooldown-based false-trigger suppression.
    """
    def __init__(
        self,
        smoothing_alpha: float = 0.4,
        history_window_seconds: float = 1.0,
        nod_threshold_deg_s: float = 40.0,
        shake_threshold_deg_s: float = 45.0,
        cooldown_seconds: float = 0.6,
        stationary_accel_threshold_g: float = 0.08,
        stationary_gyro_threshold_deg_s: float = 15.0
    ):
        self.alpha = smoothing_alpha
        self.window_seconds = history_window_seconds
        self.nod_threshold = nod_threshold_deg_s
        self.shake_threshold = shake_threshold_deg_s
        self.cooldown_seconds = cooldown_seconds
        self.stationary_accel_th = stationary_accel_threshold_g
        self.stationary_gyro_th = stationary_gyro_threshold_deg_s

        self._history: deque[IMUSample] = deque()
        self._last_smoothed: Optional[IMUSample] = None
        self._last_gesture_time: float = -999.0
        self._last_timestamp: float = -1.0

    def reset(self) -> None:
        """Resets all internal filters and historical state."""
        self._history.clear()
        self._last_smoothed = None
        self._last_gesture_time = -999.0
        self._last_timestamp = -1.0

    def _smooth_sample(self, raw: IMUSample) -> IMUSample:
        """Applies single-pole low-pass exponential smoothing."""
        if self._last_smoothed is None:
            self._last_smoothed = raw
            return raw

        s = IMUSample(
            timestamp=raw.timestamp,
            ax=self.alpha * raw.ax + (1.0 - self.alpha) * self._last_smoothed.ax,
            ay=self.alpha * raw.ay + (1.0 - self.alpha) * self._last_smoothed.ay,
            az=self.alpha * raw.az + (1.0 - self.alpha) * self._last_smoothed.az,
            gx=self.alpha * raw.gx + (1.0 - self.alpha) * self._last_smoothed.gx,
            gy=self.alpha * raw.gy + (1.0 - self.alpha) * self._last_smoothed.gy,
            gz=self.alpha * raw.gz + (1.0 - self.alpha) * self._last_smoothed.gz,
        )
        self._last_smoothed = s
        return s

    def process_sample(self, raw_sample: IMUSample) -> Optional[GestureEvent]:
        """
        Processes a single IMU sample and returns a GestureEvent if an intentional
        gesture (Nod, Shake, Tilt) is recognized.
        """
        # Validate monotonic timestamp
        if raw_sample.timestamp <= self._last_timestamp:
            # Reject out-of-order or duplicate timestamp sample
            return None
        self._last_timestamp = raw_sample.timestamp

        smoothed = self._smooth_sample(raw_sample)
        self._history.append(smoothed)

        # Evict samples older than window_seconds
        cutoff = smoothed.timestamp - self.window_seconds
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()

        # Check cooldown
        if (smoothed.timestamp - self._last_gesture_time) < self.cooldown_seconds:
            return None

        # Analyze motion window
        gesture = self._detect_gesture()
        if gesture is not None:
            self._last_gesture_time = smoothed.timestamp
            return gesture

        return None

    def is_stationary(self) -> bool:
        """Determines if the head is currently stationary within the recent window."""
        if len(self._history) < 3:
            return True

        recent = list(self._history)[-5:]
        for s in recent:
            accel_mag = math.sqrt(s.ax**2 + s.ay**2 + s.az**2)
            dynamic_accel = abs(accel_mag - 1.0)
            gyro_mag = math.sqrt(s.gx**2 + s.gy**2 + s.gz**2)

            if dynamic_accel > self.stationary_accel_th or gyro_mag > self.stationary_gyro_th:
                return False
        return True

    def _detect_gesture(self) -> Optional[GestureEvent]:
        """Inspects historical buffer for bi-phasic oscillation patterns."""
        if len(self._history) < 5:
            return None

        samples = list(self._history)
        t_start = samples[0].timestamp
        t_end = samples[-1].timestamp

        # 1. Check Pitch (Nod): Look for positive pitch peak then negative pitch peak
        gy_values = [s.gy for s in samples]
        max_gy = max(gy_values)
        min_gy = min(gy_values)

        if max_gy > self.nod_threshold and min_gy < -self.nod_threshold:
            # Check temporal ordering of peaks
            max_idx = gy_values.index(max_gy)
            min_idx = gy_values.index(min_gy)
            t_diff = abs(samples[max_idx].timestamp - samples[min_idx].timestamp)

            if 0.10 <= t_diff <= 0.65:
                # Calculate confidence based on peak separation and magnitude
                conf = min(1.0, (max_gy - min_gy) / (2.0 * self.nod_threshold))
                return GestureEvent(
                    gesture_type=HeadGestureType.NOD.value,
                    confidence=round(conf, 2),
                    start_time=t_start,
                    end_time=t_end,
                    metadata={"peak_positive": max_gy, "peak_negative": min_gy, "peak_gap_s": round(t_diff, 3)}
                )

        # 2. Check Yaw (Shake): Look for positive yaw peak then negative yaw peak
        gz_values = [s.gz for s in samples]
        max_gz = max(gz_values)
        min_gz = min(gz_values)

        if max_gz > self.shake_threshold and min_gz < -self.shake_threshold:
            max_idx = gz_values.index(max_gz)
            min_idx = gz_values.index(min_gz)
            t_diff = abs(samples[max_idx].timestamp - samples[min_idx].timestamp)

            if 0.10 <= t_diff <= 0.65:
                conf = min(1.0, (max_gz - min_gz) / (2.0 * self.shake_threshold))
                return GestureEvent(
                    gesture_type=HeadGestureType.SHAKE.value,
                    confidence=round(conf, 2),
                    start_time=t_start,
                    end_time=t_end,
                    metadata={"peak_positive": max_gz, "peak_negative": min_gz, "peak_gap_s": round(t_diff, 3)}
                )

        # 3. Check Roll (Head Tilt): Sustained lateral roll rate or sustained lateral tilt
        gx_values = [s.gx for s in samples]
        max_gx = max(gx_values)
        min_gx = min(gx_values)
        if (max_gx > 60.0 or min_gx < -60.0) and (t_end - t_start) >= 0.35:
            conf = min(1.0, max(abs(max_gx), abs(min_gx)) / 80.0)
            return GestureEvent(
                gesture_type=HeadGestureType.TILT_HEAD.value,
                confidence=round(conf, 2),
                start_time=t_start,
                end_time=t_end,
                metadata={"max_roll_rate": max(abs(max_gx), abs(min_gx))}
            )

        return None
