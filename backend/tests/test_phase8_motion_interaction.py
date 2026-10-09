import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.motion.contracts import (
    IMUSample,
    TouchSample,
    HeadGestureType,
    TouchGestureType,
    InteractionIntent
)
from backend.motion.processor import MotionProcessor
from backend.motion.touch import TouchProcessor, TouchState
from backend.motion.dispatcher import InteractionDispatcher
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
from backend.services.vision_service import VisionService
from backend.conversation.session import SessionManager


# --- 1. Contract & Validation Tests ---

def test_imu_sample_validation_valid():
    s = IMUSample(timestamp=1.0, ax=0.0, ay=0.0, az=1.0, gx=0.0, gy=0.0, gz=0.0)
    assert s.timestamp == 1.0
    assert s.az == 1.0

def test_imu_sample_validation_invalid_timestamp():
    with pytest.raises(Exception):
        IMUSample(timestamp=-1.0, ax=0.0, ay=0.0, az=1.0, gx=0.0, gy=0.0, gz=0.0)

def test_imu_sample_validation_out_of_range_accel():
    with pytest.raises(Exception):
        IMUSample(timestamp=1.0, ax=25.0, ay=0.0, az=1.0, gx=0.0, gy=0.0, gz=0.0)


# --- 2. Motion Processing & Gesture Detection Tests ---

def test_stationary_noise_suppression():
    processor = MotionProcessor()
    trace = generate_stationary_trace(duration_s=1.5, noise_std=0.015)
    
    events = []
    for s in trace:
        ev = processor.process_sample(s)
        if ev:
            events.append(ev)

    assert len(events) == 0, f"Expected 0 gestures on stationary noise, got {len(events)}"
    assert processor.is_stationary() is True

def test_slow_head_movement_rejection():
    processor = MotionProcessor()
    trace = generate_slow_head_movement_trace(duration_s=2.0)

    events = []
    for s in trace:
        ev = processor.process_sample(s)
        if ev:
            events.append(ev)

    assert len(events) == 0, f"Expected 0 gestures on slow head movement, got {len(events)}"

def test_intentional_head_nod_detection():
    processor = MotionProcessor()
    trace = generate_nod_trace(duration_s=1.0, nod_peak_deg_s=65.0)

    events = []
    for s in trace:
        ev = processor.process_sample(s)
        if ev:
            events.append(ev)

    assert len(events) == 1
    assert events[0].gesture_type == HeadGestureType.NOD.value
    assert events[0].confidence >= 0.5
    assert "peak_positive" in events[0].metadata

def test_intentional_head_shake_detection():
    processor = MotionProcessor()
    trace = generate_shake_trace(duration_s=1.0, shake_peak_deg_s=75.0)

    events = []
    for s in trace:
        ev = processor.process_sample(s)
        if ev:
            events.append(ev)

    assert len(events) == 1
    assert events[0].gesture_type == HeadGestureType.SHAKE.value
    assert events[0].confidence >= 0.5

def test_motion_cooldown_suppresses_repeat():
    processor = MotionProcessor(cooldown_seconds=1.2)
    trace = generate_nod_trace(duration_s=1.0)

    events = []
    # Feed first nod trace
    for s in trace:
        ev = processor.process_sample(s)
        if ev:
            events.append(ev)

    # Immediately feed second nod within cooldown window (0.4s shift < 0.8s cooldown)
    for s in trace:
        s_shifted = IMUSample(
            timestamp=s.timestamp + 0.4,
            ax=s.ax, ay=s.ay, az=s.az,
            gx=s.gx, gy=s.gy, gz=s.gz
        )
        ev = processor.process_sample(s_shifted)
        if ev:
            events.append(ev)

    assert len(events) == 1, "Expected cooldown to suppress immediate second nod"

def test_out_of_order_timestamp_rejected():
    processor = MotionProcessor()
    s1 = IMUSample(timestamp=1.0, ax=0, ay=0, az=1, gx=0, gy=0, gz=0)
    s2 = IMUSample(timestamp=0.8, ax=0, ay=0, az=1, gx=0, gy=0, gz=0) # Out of order
    
    assert processor.process_sample(s1) is None
    assert processor.process_sample(s2) is None


# --- 3. Silent Touch Interaction Tests ---

def test_touch_glitch_debounce_suppression():
    touch = TouchProcessor(debounce_seconds=0.03)
    trace = generate_touch_glitch_noise_trace()

    events = []
    for s in trace:
        ev = touch.process_sample(s)
        if ev:
            events.append(ev)
        # Advance trailing timer
        tick_ev = touch.tick(s.timestamp)
        if tick_ev:
            events.append(tick_ev)

    assert len(events) == 0, f"Expected glitch to be debounced, got {len(events)}"

def test_touch_single_tap():
    touch = TouchProcessor(debounce_seconds=0.03, double_tap_max_gap_seconds=0.25)
    trace = generate_touch_tap_trace(press_time_s=0.1, hold_duration_s=0.12)

    events = []
    for s in trace:
        ev = touch.process_sample(s)
        if ev:
            events.append(ev)
        tick_ev = touch.tick(s.timestamp)
        if tick_ev:
            events.append(tick_ev)

    assert len(events) == 1
    assert events[0].gesture_type == TouchGestureType.TAP.value

def test_touch_double_tap():
    touch = TouchProcessor(debounce_seconds=0.03, double_tap_max_gap_seconds=0.30)
    trace = generate_touch_double_tap_trace(first_tap_time_s=0.1, tap_duration_s=0.08, gap_duration_s=0.12)

    events = []
    for s in trace:
        ev = touch.process_sample(s)
        if ev:
            events.append(ev)
        tick_ev = touch.tick(s.timestamp)
        if tick_ev:
            events.append(tick_ev)

    assert len(events) == 1
    assert events[0].gesture_type == TouchGestureType.DOUBLE_TAP.value

def test_touch_long_press():
    touch = TouchProcessor(long_press_min_seconds=0.60)
    trace = generate_touch_long_press_trace(press_time_s=0.1, hold_duration_s=0.75)

    events = []
    for s in trace:
        ev = touch.process_sample(s)
        if ev:
            events.append(ev)

    assert len(events) == 1
    assert events[0].gesture_type == TouchGestureType.LONG_PRESS.value


# --- 4. Safe Interaction Dispatcher Tests ---

@pytest.mark.anyio
async def test_dispatcher_mappings():
    disp = InteractionDispatcher()
    
    from backend.motion.contracts import GestureEvent
    nod_ev = GestureEvent(gesture_type=HeadGestureType.NOD.value, start_time=0, end_time=1)
    shake_ev = GestureEvent(gesture_type=HeadGestureType.SHAKE.value, start_time=0, end_time=1)
    dtap_ev = GestureEvent(gesture_type=TouchGestureType.DOUBLE_TAP.value, start_time=0, end_time=1)

    assert disp.map_gesture_to_intent(nod_ev) == InteractionIntent.CONFIRM
    assert disp.map_gesture_to_intent(shake_ev) == InteractionIntent.DISMISS_OR_CANCEL
    assert disp.map_gesture_to_intent(dtap_ev) == InteractionIntent.TRIGGER_SCENE_ANALYSIS

@pytest.mark.anyio
async def test_dispatcher_scene_analysis_execution():
    vision_svc = VisionService()
    session_mgr = SessionManager()
    disp = InteractionDispatcher(vision_service=vision_svc, session_manager=session_mgr)

    from backend.motion.contracts import GestureEvent
    dtap_ev = GestureEvent(gesture_type=TouchGestureType.DOUBLE_TAP.value, start_time=0, end_time=1)
    
    res = await disp.dispatch_gesture(dtap_ev)
    assert res.success is True
    assert res.intent == InteractionIntent.TRIGGER_SCENE_ANALYSIS
    assert res.target_service == "VisionService"

@pytest.mark.anyio
async def test_dispatcher_cooldown_suppression():
    disp = InteractionDispatcher(intent_cooldown_seconds=1.0)

    from backend.motion.contracts import GestureEvent
    nod_ev = GestureEvent(gesture_type=HeadGestureType.NOD.value, start_time=0, end_time=1)

    res1 = await disp.dispatch_gesture(nod_ev)
    assert res1.success is True

    # Immediate second dispatch
    res2 = await disp.dispatch_gesture(nod_ev)
    assert res2.success is False
    assert res2.action_taken == "COOLDOWN_SUPPRESSED"
