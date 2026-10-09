from backend.motion.contracts import (
    SensorStatus,
    HeadGestureType,
    TouchGestureType,
    InteractionIntent,
    IMUSample,
    TouchSample,
    GestureEvent,
    InteractionResult
)
from backend.motion.processor import MotionProcessor
from backend.motion.touch import TouchProcessor, TouchState
from backend.motion.dispatcher import (
    InteractionDispatcher,
    get_interaction_dispatcher,
    reset_interaction_dispatcher
)
from backend.motion.synthetic import (
    generate_stationary_trace,
    generate_nod_trace,
    generate_shake_trace,
    generate_slow_head_movement_trace,
    generate_touch_tap_trace,
    generate_touch_double_tap_trace,
    generate_touch_long_press_trace,
    generate_touch_glitch_noise_trace
)

__all__ = [
    "SensorStatus",
    "HeadGestureType",
    "TouchGestureType",
    "InteractionIntent",
    "IMUSample",
    "TouchSample",
    "GestureEvent",
    "InteractionResult",
    "MotionProcessor",
    "TouchProcessor",
    "TouchState",
    "InteractionDispatcher",
    "get_interaction_dispatcher",
    "reset_interaction_dispatcher",
    "generate_stationary_trace",
    "generate_nod_trace",
    "generate_shake_trace",
    "generate_slow_head_movement_trace",
    "generate_touch_tap_trace",
    "generate_touch_double_tap_trace",
    "generate_touch_long_press_trace",
    "generate_touch_glitch_noise_trace"
]
