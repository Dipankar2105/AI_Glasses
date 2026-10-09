"""Automated integration tests verifying the NextSight local system demonstration path."""

import json
import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.protocol.contracts import PacketType
from backend.protocol.framing import ProtocolFraming
from backend.protocol.device_sim import SimulatedESP32Device
from backend.reliability.orchestrator import SystemIntegrationOrchestrator
from backend.reliability.circuit_breaker import CircuitBreaker, CircuitState


@pytest.fixture
def api_client():
    return TestClient(app)


@pytest.fixture
def orchestrator():
    return SystemIntegrationOrchestrator()


def test_demo_stage1_backend_health_and_capabilities(api_client):
    """Verify health, readiness, and capabilities endpoints."""
    # Health check
    res_h = api_client.get("/health")
    assert res_h.status_code == 200
    assert res_h.json()["status"] == "ok"

    # Readiness check
    res_r = api_client.get("/ready")
    assert res_r.status_code == 200
    assert res_r.json()["status"] == "READY"

    # Capabilities check
    res_c = api_client.get("/api/v1/capabilities")
    assert res_c.status_code == 200
    caps = res_c.json()
    assert "fastapi_backend_server" in caps["implemented"]
    assert "physical_camera_sensor" in caps["unavailable"]


def test_demo_stage2_device_simulator_framing(orchestrator):
    """Verify simulated device connection, framing, and telemetry ingest."""
    sim = SimulatedESP32Device(device_id="test-glasses-demo")
    sim.connect()
    assert sim.is_connected is True

    # Send heartbeat
    hb_bytes = sim.send_heartbeat()
    msg = ProtocolFraming.decode_message(hb_bytes)
    assert msg.is_valid is True
    assert msg.header.magic == 0xAA55
    assert msg.header.packet_type == PacketType.HEARTBEAT

    # Send telemetry
    telem_bytes = sim.send_telemetry(battery_pct=88.0, temp_c=33.0)
    res = orchestrator.process_incoming_device_packet(telem_bytes)
    assert res["status"] == "PROCESSED"
    assert res["battery_status"] == "NORMAL"


def test_demo_stage3_interaction_dispatching(orchestrator):
    """Verify motion/touch events are routed to correct services."""
    # Tap -> Confirm
    tap_packet = ProtocolFraming.encode_message(
        packet_type=PacketType.TOUCH_EVENT,
        payload=json.dumps({"gesture": "TAP", "confidence": 1.0}).encode("utf-8"),
        sequence_number=1,
    )
    res_tap = orchestrator.process_incoming_device_packet(tap_packet)
    assert res_tap["status"] == "DISPATCHED"
    assert res_tap["intent"] == "CONFIRM"
    assert res_tap["success"] is True

    # Double Tap -> Vision Scene Analysis
    dt_packet = ProtocolFraming.encode_message(
        packet_type=PacketType.TOUCH_EVENT,
        payload=json.dumps({"gesture": "DOUBLE_TAP", "confidence": 1.0}).encode("utf-8"),
        sequence_number=2,
    )
    res_dt = orchestrator.process_incoming_device_packet(dt_packet)
    assert res_dt["status"] == "DISPATCHED"
    assert res_dt["intent"] == "TRIGGER_SCENE_ANALYSIS"


def test_demo_stage4_power_gating(orchestrator):
    """Verify low-battery telemetry halts heavy vision workloads."""
    sim = SimulatedESP32Device()
    sim.connect()

    # Drop battery to 10%
    low_bat_packet = sim.send_telemetry(battery_pct=10.0, temp_c=36.0)
    orchestrator.process_incoming_device_packet(low_bat_packet)

    # Attempt vision trigger
    dt_packet = ProtocolFraming.encode_message(
        packet_type=PacketType.TOUCH_EVENT,
        payload=json.dumps({"gesture": "DOUBLE_TAP", "confidence": 1.0}).encode("utf-8"),
    )
    blocked_res = orchestrator.process_incoming_device_packet(dt_packet)
    assert blocked_res["status"] == "BLOCKED_BY_POWER_POLICY"


def test_demo_stage5_mcp_and_conversation(api_client):
    """Verify MCP direct execution and conversation message integration."""
    # Direct MCP tool call
    mcp_res = api_client.post("/api/v1/mcp/tools/call", json={"name": "get_system_status", "arguments": {}})
    assert mcp_res.status_code == 200
    assert mcp_res.json()["is_error"] is False

    # Conversation message with tool invocation
    conv_res = api_client.post(
        "/api/v1/conversation/message",
        json={"message": "System status check", "tool_to_invoke": "get_system_status", "tool_arguments": {}},
    )
    assert conv_res.status_code == 200
    data = conv_res.json()
    assert data["llm_status"] == "PROVIDER_UNAVAILABLE"
    assert len(data["tool_executions"]) == 1


def test_demo_stage6_fault_tolerance(orchestrator):
    """Verify corrupt packet handling and circuit breaker trip."""
    # Bad magic
    bad_bytes = b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x04DATA\x00\x00\x00\x00"
    res = orchestrator.process_incoming_device_packet(bad_bytes)
    assert res["status"] == "REJECTED"

    # Circuit breaker
    cb = CircuitBreaker(name="test_circuit", failure_threshold=2, recovery_timeout_s=0.05)
    cb.record_failure(1.0)
    cb.record_failure(1.0)
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute(1.0) is False
