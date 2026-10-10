"""Unit and integration tests for Phase 11: Performance, Reliability, and Fault Tolerance."""

import time
import json
import pytest
from backend.reliability.retry import RetryPolicy
from backend.reliability.circuit_breaker import CircuitBreaker, CircuitState
from backend.reliability.rate_limiter import RateLimiter
from backend.reliability.orchestrator import SystemIntegrationOrchestrator
from backend.protocol.contracts import PacketType
from backend.protocol.framing import ProtocolFraming
from backend.power.contracts import BatteryTelemetry, ThermalTelemetry
from firmware.system.reliability import RetryPolicy as FirmwareRetryPolicy


def test_retry_policy_success():
    """Verify RetryPolicy executes immediately on success."""
    policy = RetryPolicy(max_retries=3, initial_backoff_s=0.001)
    call_count = 0

    def work():
        nonlocal call_count
        call_count += 1
        return "SUCCESS"

    res = policy.execute(work)
    assert res == "SUCCESS"
    assert call_count == 1


def test_retry_policy_retries_and_recovers():
    """Verify RetryPolicy retries upon transient errors and returns once successful."""
    policy = RetryPolicy(max_retries=4, initial_backoff_s=0.001, jitter=False)
    attempts = 0

    def flaky_work():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("Transient network drop")
        return "RECOVERED"

    res = policy.execute(flaky_work)
    assert res == "RECOVERED"
    assert attempts == 3


def test_retry_policy_exhaustion():
    """Verify RetryPolicy raises RuntimeError when maximum retries are exhausted."""
    policy = RetryPolicy(max_retries=2, initial_backoff_s=0.001, jitter=False)

    def failing_work():
        raise TimeoutError("Service timed out")

    with pytest.raises(RuntimeError, match="Max retries"):
        policy.execute(failing_work)


def test_circuit_breaker_transitions():
    """Verify CircuitBreaker CLOSED -> OPEN -> HALF_OPEN -> CLOSED state transitions."""
    cb = CircuitBreaker(name="test_service", failure_threshold=2, recovery_timeout_s=0.05)
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute(current_time=100.0) is True

    # 1st failure
    cb.record_failure(current_time=100.0)
    assert cb.state == CircuitState.CLOSED

    # 2nd failure -> Trips circuit to OPEN
    cb.record_failure(current_time=100.1)
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute(current_time=100.1) is False

    # Check during recovery timeout -> Still OPEN
    assert cb.can_execute(current_time=100.12) is False

    # Check after recovery timeout (0.05s) -> Transitions to HALF_OPEN
    assert cb.can_execute(current_time=100.20) is True
    assert cb.state == CircuitState.HALF_OPEN

    # Success in HALF_OPEN -> Recovers to CLOSED
    cb.record_success(current_time=100.21)
    assert cb.state == CircuitState.CLOSED


def test_rate_limiter_token_bucket():
    """Verify RateLimiter permits bursts up to capacity and throttles over-limit requests."""
    limiter = RateLimiter(rate_per_second=10.0, burst_capacity=5)

    # First 5 tokens consumed immediately
    for _ in range(5):
        assert limiter.allow_request(tokens_requested=1.0, current_time=10.0) is True

    # 6th token denied at same timestamp
    assert limiter.allow_request(tokens_requested=1.0, current_time=10.0) is False

    # After 0.2s, 2 tokens replenished (10 * 0.2)
    assert limiter.allow_request(tokens_requested=1.0, current_time=10.2) is True
    assert limiter.allow_request(tokens_requested=1.0, current_time=10.2) is True
    assert limiter.allow_request(tokens_requested=1.0, current_time=10.2) is False


def test_orchestrator_end_to_end_scenarios():
    """Test full system integration across telemetry, gestures, power gating, and invalid packets."""
    orchestrator = SystemIntegrationOrchestrator()

    # Scenario 1: Telemetry processing
    telem_payload = json.dumps({"battery_pct": 80.0, "temp_c": 32.0}).encode("utf-8")
    telem_packet = ProtocolFraming.encode_message(
        packet_type=PacketType.TELEMETRY,
        payload=telem_payload,
        sequence_number=1,
        timestamp_ms=1000,
    )
    res_telem = orchestrator.process_incoming_device_packet(telem_packet)
    assert res_telem["status"] == "PROCESSED"
    assert res_telem["battery_status"] == "NORMAL"
    assert res_telem["thermal_status"] == "NORMAL"

    # Scenario 2: Tap Gesture -> Confirmation
    tap_payload = json.dumps({"gesture": "TAP", "confidence": 0.95}).encode("utf-8")
    tap_packet = ProtocolFraming.encode_message(
        packet_type=PacketType.TOUCH_EVENT,
        payload=tap_payload,
        sequence_number=2,
        timestamp_ms=2000,
    )
    res_tap = orchestrator.process_incoming_device_packet(tap_packet)
    assert res_tap["status"] == "DISPATCHED"
    assert res_tap["intent"] == "CONFIRM"
    assert res_tap["success"] is True

    # Scenario 3: Double Tap with Vision Service Ready
    dt_payload = json.dumps({"gesture": "DOUBLE_TAP", "confidence": 0.98}).encode("utf-8")
    dt_packet = ProtocolFraming.encode_message(
        packet_type=PacketType.TOUCH_EVENT,
        payload=dt_payload,
        sequence_number=3,
        timestamp_ms=3000,
    )
    res_dt = orchestrator.process_incoming_device_packet(dt_packet)
    assert res_dt["status"] == "DISPATCHED"
    assert res_dt["intent"] == "TRIGGER_SCENE_ANALYSIS"

    # Scenario 4: Double Tap when Battery is LOW -> Power policy blocks vision capture
    low_bat_payload = json.dumps({"battery_pct": 12.0, "temp_c": 30.0}).encode("utf-8")
    low_bat_packet = ProtocolFraming.encode_message(
        packet_type=PacketType.TELEMETRY,
        payload=low_bat_payload,
        sequence_number=4,
        timestamp_ms=4000,
    )
    orchestrator.process_incoming_device_packet(low_bat_packet)

    dt_blocked = orchestrator.process_incoming_device_packet(dt_packet)
    assert dt_blocked["status"] == "BLOCKED_BY_POWER_POLICY"
    assert "conserve low battery" in dt_blocked["reason"]

    # Scenario 5: Corrupted Packet Rejection
    corrupted_packet = b"\xAA\x55\x01\x00\x00\x01\x00\x00\x00\x00\x00\x04BAD_CRC_BYTES"
    res_corrupted = orchestrator.process_incoming_device_packet(corrupted_packet)
    assert res_corrupted["status"] == "REJECTED"
    assert "error" in res_corrupted


def test_firmware_reliability_wrapper_compatibility():
    """Verify firmware.system.reliability.RetryPolicy wrapper."""
    r = FirmwareRetryPolicy()
    f = lambda: True
    assert r.execute(f) is True


def test_orchestrator_strict_telemetry_gate_scenario():
    """Verify that SystemIntegrationOrchestrator with require_verified_telemetry=True enforces the strict safety gate."""
    from backend.power.policy import PowerThermalPolicyManager

    strict_power = PowerThermalPolicyManager(require_verified_telemetry=True)
    orchestrator = SystemIntegrationOrchestrator(power_manager=strict_power)

    dt_payload = json.dumps({"gesture": "DOUBLE_TAP", "confidence": 0.98}).encode("utf-8")
    dt_packet_uninit = ProtocolFraming.encode_message(
        packet_type=PacketType.TOUCH_EVENT,
        payload=dt_payload,
        sequence_number=1,
        timestamp_ms=1000,
    )

    # 1. Uninitialized telemetry -> Double tap blocked by policy
    res1 = orchestrator.process_incoming_device_packet(dt_packet_uninit)
    assert res1["status"] == "BLOCKED_BY_POWER_POLICY"
    assert "uninitialized" in res1["reason"].lower()

    # 2. Ingest fresh valid telemetry at t=2000ms
    telem_payload = json.dumps({"battery_pct": 85.0, "temp_c": 32.0}).encode("utf-8")
    telem_packet = ProtocolFraming.encode_message(
        packet_type=PacketType.TELEMETRY,
        payload=telem_payload,
        sequence_number=2,
        timestamp_ms=2000,
    )
    res_telem = orchestrator.process_incoming_device_packet(telem_packet)
    assert res_telem["status"] == "PROCESSED"
    assert res_telem["battery_status"] == "NORMAL"

    # 3. Double tap at t=2100ms -> Vision capture permitted & dispatched
    dt_packet_fresh = ProtocolFraming.encode_message(
        packet_type=PacketType.TOUCH_EVENT,
        payload=dt_payload,
        sequence_number=3,
        timestamp_ms=2100,
    )
    res2 = orchestrator.process_incoming_device_packet(dt_packet_fresh)
    assert res2["status"] == "DISPATCHED"
    assert res2["intent"] == "TRIGGER_SCENE_ANALYSIS"

    # 4. Double tap at t=60000ms (>30s later without new telemetry) -> Stale telemetry blocks capture
    dt_packet_stale = ProtocolFraming.encode_message(
        packet_type=PacketType.TOUCH_EVENT,
        payload=dt_payload,
        sequence_number=4,
        timestamp_ms=60000,
    )
    res3 = orchestrator.process_incoming_device_packet(dt_packet_stale)
    assert res3["status"] == "BLOCKED_BY_POWER_POLICY"
    assert "stale" in res3["reason"].lower()

