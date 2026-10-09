from enum import Enum
from typing import Optional

from backend.motion.contracts import TouchSample, TouchGestureType, GestureEvent

class TouchState(str, Enum):
    IDLE = "IDLE"
    PRESSED = "PRESSED"
    WAITING_FOR_DOUBLE_TAP = "WAITING_FOR_DOUBLE_TAP"

class TouchProcessor:
    """
    Deterministic touch interaction state machine for capacitive sensors.
    Supports:
      - Noise debouncing (30ms threshold).
      - Single Tap (< 350ms press).
      - Double Tap (2 taps within 300ms inter-tap window).
      - Long Press (>= 600ms continuous press).
      - Missing/duplicate event suppression.
    """
    def __init__(
        self,
        debounce_seconds: float = 0.03,
        short_tap_max_seconds: float = 0.35,
        long_press_min_seconds: float = 0.60,
        double_tap_max_gap_seconds: float = 0.30
    ):
        self.debounce_sec = debounce_seconds
        self.short_tap_max = short_tap_max_seconds
        self.long_press_min = long_press_min_seconds
        self.double_tap_gap = double_tap_max_gap_seconds

        self.state = TouchState.IDLE
        self._press_start_time: float = 0.0
        self._last_release_time: float = -999.0
        self._last_sample_time: float = -1.0
        self._is_pressed_debounced: bool = False
        self._long_press_fired: bool = False

    def reset(self) -> None:
        """Resets the touch state machine."""
        self.state = TouchState.IDLE
        self._press_start_time = 0.0
        self._last_release_time = -999.0
        self._last_sample_time = -1.0
        self._is_pressed_debounced = False
        self._long_press_fired = False
        self._double_tap_fired = False

    def process_sample(self, sample: TouchSample) -> Optional[GestureEvent]:
        """
        Updates state with incoming capacitive sample and emits GestureEvent when complete.
        """
        if sample.timestamp <= self._last_sample_time:
            # Monotonicity check
            return None
        self._last_sample_time = sample.timestamp

        t = sample.timestamp
        pressed = sample.is_pressed

        # 1. State: IDLE
        if self.state == TouchState.IDLE:
            if pressed:
                self.state = TouchState.PRESSED
                self._press_start_time = t
                self._long_press_fired = False
                self._double_tap_fired = False
            return None

        # 2. State: PRESSED
        elif self.state == TouchState.PRESSED:
            press_duration = t - self._press_start_time

            # Check long press trigger while holding
            if pressed:
                if press_duration >= self.long_press_min and not self._long_press_fired and not self._double_tap_fired:
                    self._long_press_fired = True
                    return GestureEvent(
                        gesture_type=TouchGestureType.LONG_PRESS.value,
                        confidence=1.0,
                        start_time=self._press_start_time,
                        end_time=t,
                        metadata={"hold_duration_s": round(press_duration, 3)}
                    )
                return None
            else:
                # Released
                if self._double_tap_fired:
                    self.state = TouchState.IDLE
                    self._double_tap_fired = False
                    return None

                if press_duration < self.debounce_sec:
                    # Spurious glitch / noise below debounce threshold
                    self.state = TouchState.IDLE
                    return None

                if self._long_press_fired:
                    # Already handled long press
                    self.state = TouchState.IDLE
                    return None

                if press_duration <= self.short_tap_max:
                    # Valid short press, now transition to wait for potential second tap
                    self.state = TouchState.WAITING_FOR_DOUBLE_TAP
                    self._last_release_time = t
                    return None
                else:
                    self.state = TouchState.IDLE
                    return None

        # 3. State: WAITING_FOR_DOUBLE_TAP
        elif self.state == TouchState.WAITING_FOR_DOUBLE_TAP:
            gap = t - self._last_release_time

            if pressed:
                # Second press occurred within double tap gap
                if gap <= self.double_tap_gap:
                    self.state = TouchState.PRESSED
                    self._press_start_time = t
                    self._double_tap_fired = True
                    return GestureEvent(
                        gesture_type=TouchGestureType.DOUBLE_TAP.value,
                        confidence=1.0,
                        start_time=self._press_start_time,
                        end_time=t,
                        metadata={"inter_tap_gap_s": round(gap, 3)}
                    )
                else:
                    # Too late for double tap; treat as new first press
                    self.state = TouchState.PRESSED
                    self._press_start_time = t
                    self._long_press_fired = False
                    self._double_tap_fired = False
                    # Emit previous single tap first
                    return GestureEvent(
                        gesture_type=TouchGestureType.TAP.value,
                        confidence=1.0,
                        start_time=self._press_start_time,
                        end_time=self._last_release_time,
                        metadata={"tap_duration_s": round(self._last_release_time - self._press_start_time, 3)}
                    )
            else:
                # Still released; check if timeout expired to resolve single tap
                if gap > self.double_tap_gap:
                    self.state = TouchState.IDLE
                    return GestureEvent(
                        gesture_type=TouchGestureType.TAP.value,
                        confidence=1.0,
                        start_time=self._press_start_time,
                        end_time=self._last_release_time,
                        metadata={"tap_duration_s": round(self._last_release_time - self._press_start_time, 3)}
                    )
                return None

        return None

    def tick(self, current_time: float) -> Optional[GestureEvent]:
        """
        Advances timer. Call when no new sensor events are arriving to flush pending single taps.
        """
        if self.state == TouchState.WAITING_FOR_DOUBLE_TAP:
            gap = current_time - self._last_release_time
            if gap > self.double_tap_gap:
                self.state = TouchState.IDLE
                return GestureEvent(
                    gesture_type=TouchGestureType.TAP.value,
                    confidence=1.0,
                    start_time=self._press_start_time,
                    end_time=self._last_release_time,
                    metadata={"tap_duration_s": round(self._last_release_time - self._press_start_time, 3)}
                )
        return None
