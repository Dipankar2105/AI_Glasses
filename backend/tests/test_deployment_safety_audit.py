"""
Deployment Safety & Telemetry Verification Audit Test Suite.
Verifies that unauthorized vision capture is strictly blocked under all telemetry edge cases
(uninitialized, invalid, missing, stale, critical, out-of-order, direct capture, and legacy wrappers).
"""

import time
import json
import pytest
from backend.power.contracts import (
    OperatingState,
    BatteryStatus,
    ThermalStatus,
    WorkloadPriority,
    BatteryTelemetry,
    ThermalTelemetry,
)
from backend.power.policy import PowerThermalPolicyManager
from backend.protocol.contracts import PacketType
from backend.protocol.framing import ProtocolFraming
from backend.reliability.orchestrator import SystemIntegrationOrchestrator
from firmware.system.power import PowerManager as FirmwarePowerManager
from firmware.system.thermal import ThermalManager as FirmwareThermalManager


class TestDeploymentSafetyGate:
    """Rigorous audit tests for hardware-readiness telemetry gating."""

    def test_uninitialized_telemetry_blocks_direct_capture_and_state(self):
        """Direct capture request and CAPTURE state transition are strictly blocked when telemetry is uninitialized."""
        manager = PowerThermalPolicyManager(require_verified_telemetry=True)
        
        # Direct workload check
        allowed, reason = manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)
        assert allowed is False
        assert "uninitialized" in reason.lower()

        # Direct state transition request
        transition_ok, trans_reason = manager.request_state(OperatingState.CAPTURE)
        assert transition_ok is False
        assert "uninitialized" in trans_reason.lower()

    def test_missing_single_stream_blocks_capture(self):
        """If only one telemetry stream is provided, vision capture remains strictly blocked."""
        # Only battery provided
        mgr1 = PowerThermalPolicyManager(require_verified_telemetry=True)
        mgr1.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.9, percentage=80.0, timestamp=10.0), current_time=10.0)
        allowed1, _ = mgr1.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=10.0)
        assert allowed1 is False

        # Only thermal provided
        mgr2 = PowerThermalPolicyManager(require_verified_telemetry=True)
        mgr2.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=32.0, timestamp=10.0), current_time=10.0)
        allowed2, _ = mgr2.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=10.0)
        assert allowed2 is False

    def test_invalid_telemetry_blocks_capture(self):
        """Telemetry marked is_valid=False results in UNKNOWN status and blocks capture."""
        manager = PowerThermalPolicyManager(require_verified_telemetry=True)
        # Valid thermal, invalid battery
        manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=32.0, timestamp=10.0), current_time=10.0)
        manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=0.0, percentage=0.0, is_valid=False, timestamp=10.0), current_time=10.0)
        
        assert manager.battery_status == BatteryStatus.UNKNOWN
        allowed, reason = manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=10.0)
        assert allowed is False
        assert "unknown" in reason.lower()

    def test_stale_telemetry_blocks_capture(self):
        """Telemetry exceeding staleness timeout triggers on-demand STALE status and blocks capture."""
        manager = PowerThermalPolicyManager(
            require_verified_telemetry=True,
            battery_stale_timeout_s=30.0,
            thermal_stale_timeout_s=15.0,
        )
        # Ingest fresh telemetry at t=100.0s
        manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.9, percentage=80.0, timestamp=100.0), current_time=100.0)
        manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=32.0, timestamp=100.0), current_time=100.0)
        
        # Valid at t=105.0s
        allowed, _ = manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=105.0)
        assert allowed is True

        # Stale at t=140.0s (>30s battery, >15s thermal)
        allowed_stale, reason = manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=140.0)
        assert allowed_stale is False
        assert "stale" in reason.lower()

    def test_out_of_order_telemetry_cannot_corrupt_gate(self):
        """Older telemetry packets arriving out of order are rejected and cannot bypass state safety."""
        manager = PowerThermalPolicyManager(require_verified_telemetry=True)
        # Fresh telemetry at t=100.0s
        manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.9, percentage=80.0, timestamp=100.0), current_time=100.0)
        manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=32.0, timestamp=100.0), current_time=100.0)
        assert manager.battery_status == BatteryStatus.NORMAL

        # Out-of-order packet with old timestamp t=50.0s
        status = manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.4, percentage=5.0, timestamp=50.0), current_time=100.0)
        # Must retain the newer valid state
        assert status == BatteryStatus.NORMAL
        assert manager.last_battery.timestamp == 100.0

    def test_critical_telemetry_and_recovery_safety(self):
        """Critical temperature or battery forces shutdown and cannot be bypassed until both sensors recover."""
        manager = PowerThermalPolicyManager(require_verified_telemetry=True)
        manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.9, percentage=80.0, timestamp=10.0), current_time=10.0)
        manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=75.0, timestamp=10.0), current_time=10.0)

        # In critical shutdown
        assert manager.thermal_status == ThermalStatus.CRITICAL
        assert manager.requested_state == OperatingState.CRITICAL_SHUTDOWN
        assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=10.0)[0] is False

        # Attempt to recover with partial fresh reading (only battery fresh at t=20s, thermal still critical)
        manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=4.0, percentage=90.0, timestamp=20.0), current_time=20.0)
        assert manager.requested_state == OperatingState.CRITICAL_SHUTDOWN
        assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=20.0)[0] is False

        # Thermal recovers at t=30s
        manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=35.0, timestamp=30.0), current_time=30.0)
        assert manager.thermal_status == ThermalStatus.NORMAL
        assert manager.requested_state == OperatingState.IDLE
        assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=30.0)[0] is True

    def test_orchestrator_ingress_strict_gate_enforcement(self):
        """End-to-end orchestrator rejects DOUBLE_TAP vision triggers when telemetry is uninitialized, stale, or critical."""
        strict_policy = PowerThermalPolicyManager(
            require_verified_telemetry=True,
            battery_stale_timeout_s=30.0,
            thermal_stale_timeout_s=15.0,
        )
        orchestrator = SystemIntegrationOrchestrator(power_manager=strict_policy)

        dt_payload = json.dumps({"gesture": "DOUBLE_TAP", "confidence": 0.99}).encode("utf-8")
        dt_packet_t1 = ProtocolFraming.encode_message(
            packet_type=PacketType.TOUCH_EVENT,
            payload=dt_payload,
            sequence_number=1,
            timestamp_ms=1000,
        )

        # 1. Uninitialized -> BLOCKED
        res1 = orchestrator.process_incoming_device_packet(dt_packet_t1)
        assert res1["status"] == "BLOCKED_BY_POWER_POLICY"
        assert "uninitialized" in res1["reason"].lower()

        # 2. Supply fresh telemetry at t=2000ms
        telem_payload = json.dumps({"battery_pct": 85.0, "temp_c": 32.0}).encode("utf-8")
        telem_packet = ProtocolFraming.encode_message(
            packet_type=PacketType.TELEMETRY,
            payload=telem_payload,
            sequence_number=2,
            timestamp_ms=2000,
        )
        res_telem = orchestrator.process_incoming_device_packet(telem_packet)
        assert res_telem["status"] == "PROCESSED"

        # 3. Fresh gesture at t=2200ms -> DISPATCHED
        dt_packet_t2 = ProtocolFraming.encode_message(
            packet_type=PacketType.TOUCH_EVENT,
            payload=dt_payload,
            sequence_number=3,
            timestamp_ms=2200,
        )
        res2 = orchestrator.process_incoming_device_packet(dt_packet_t2)
        assert res2["status"] == "DISPATCHED"
        assert res2["intent"] == "TRIGGER_SCENE_ANALYSIS"

        # 4. Stale gesture at t=50000ms (>30s later) -> BLOCKED
        dt_packet_t3 = ProtocolFraming.encode_message(
            packet_type=PacketType.TOUCH_EVENT,
            payload=dt_payload,
            sequence_number=4,
            timestamp_ms=50000,
        )
        res3 = orchestrator.process_incoming_device_packet(dt_packet_t3)
        assert res3["status"] == "BLOCKED_BY_POWER_POLICY"
        assert "stale" in res3["reason"].lower()

    def test_legacy_firmware_wrappers_with_strict_policy(self):
        """Legacy firmware wrappers correctly update and reflect strict policy state."""
        strict_policy = PowerThermalPolicyManager(require_verified_telemetry=True)
        f_pwr = FirmwarePowerManager(policy=strict_policy)
        f_therm = FirmwareThermalManager(policy=strict_policy)

        # Wrappers update underlying policy
        assert f_pwr.check_battery(85) == "ACTIVE"
        assert f_therm.check_temp(32) == "NORMAL"
        assert strict_policy.battery_status == BatteryStatus.NORMAL
        assert strict_policy.thermal_status == ThermalStatus.NORMAL
        assert strict_policy.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True

    @pytest.mark.anyio
    async def test_vision_service_direct_power_authorization_gate(self):
        """Direct invocation of VisionService.process_frame_async is intercepted when power manager is attached."""
        import numpy as np
        from backend.services.vision_service import VisionService
        from backend.vision.frame import VisionFrame

        policy = PowerThermalPolicyManager(require_verified_telemetry=True)
        service = VisionService(power_manager=policy)

        frame = VisionFrame(
            data=np.zeros((50, 50, 3), dtype=np.uint8),
            width=50,
            height=50,
            channels=3,
            pixel_format="RGB",
            numerical_range=(0, 255),
            timestamp=1000,
            seq_num=1,
        )

        # 1. Uninitialized telemetry -> Blocked directly at service boundary
        res1 = await service.process_frame_async(frame)
        assert len(res1.errors) > 0
        assert res1.errors[0]["error"] == "BLOCKED_BY_POWER_POLICY"
        assert "uninitialized" in res1.errors[0]["msg"].lower()

        # 2. Ingest valid telemetry at t=1.0s
        policy.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.9, percentage=80.0, timestamp=1.0), current_time=1.0)
        policy.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=32.0, timestamp=1.0), current_time=1.0)

        # Authorized frame execution at t=1.0s (1000ms)
        res2 = await service.process_frame_async(frame)
        assert len(res2.errors) == 0
        assert res2.detection_result is not None

        # 3. Enter LOW_POWER state -> Blocked directly at service boundary
        policy.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.5, percentage=12.0, timestamp=2.0), current_time=2.0)
        res3 = await service.process_frame_async(frame)
        assert len(res3.errors) > 0
        assert res3.errors[0]["error"] == "BLOCKED_BY_POWER_POLICY"
        assert "low battery" in res3.errors[0]["msg"].lower()

    def test_production_fastapi_app_wiring_fails_closed(self):
        """FastAPI app in production mode wires strict power manager to vision endpoint, failing closed when uninitialized."""
        from fastapi.testclient import TestClient
        from backend.app import create_app
        from backend.config.settings import AppSettings
        from backend.services.vision_service import reset_vision_service
        from backend.power.policy import reset_power_policy_manager

        reset_vision_service(None)
        reset_power_policy_manager(None)

        prod_settings = AppSettings(environment="production")
        app = create_app(settings=prod_settings)
        client = TestClient(app)

        # 1. Uninitialized telemetry -> POST /api/v1/vision/process fails closed
        resp = client.post("/api/v1/vision/process", json={"use_mock_frame": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert len(data["errors"]) > 0
        assert data["errors"][0]["error"] == "BLOCKED_BY_POWER_POLICY"
        assert "uninitialized" in data["errors"][0]["msg"].lower()

        # 2. Ingest valid telemetry into app.state.power_manager
        app.state.power_manager.update_battery_telemetry(
            BatteryTelemetry(voltage_volts=3.9, percentage=80.0, timestamp=time.time()),
            current_time=time.time()
        )
        app.state.power_manager.update_thermal_telemetry(
            ThermalTelemetry(temperature_celsius=32.0, timestamp=time.time()),
            current_time=time.time()
        )

        # Authorized vision processing
        resp2 = client.post("/api/v1/vision/process", json={"use_mock_frame": True})
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["success"] is True
        assert len(data2["errors"]) == 0

        reset_vision_service(None)
        reset_power_policy_manager(None)

    @pytest.mark.anyio
    async def test_mcp_tool_with_strict_power_wiring(self):
        """MCP analyze_vision_frame tool fails closed when strict power policy is active."""
        from backend.mcp.registry import MCPToolRegistry
        from backend.services.vision_service import VisionService, reset_vision_service
        from backend.power.policy import PowerThermalPolicyManager, reset_power_policy_manager

        reset_vision_service(None)
        reset_power_policy_manager(None)

        strict_policy = PowerThermalPolicyManager(require_verified_telemetry=True)
        strict_service = VisionService(power_manager=strict_policy)
        reset_vision_service(strict_service)

        registry = MCPToolRegistry()

        # 1. Uninitialized -> Error reported in tool result payload
        call_res = await registry.execute_tool("analyze_vision_frame", arguments={"use_synthetic_frame": True})
        assert call_res.is_error is False
        payload = call_res.content[0]["data"]
        assert len(payload["errors"]) > 0
        assert payload["errors"][0]["error"] == "BLOCKED_BY_POWER_POLICY"
        assert "uninitialized" in payload["errors"][0]["msg"].lower()

        # Clean up
        reset_vision_service(None)
        reset_power_policy_manager(None)

    @pytest.mark.anyio
    async def test_unauthorized_request_does_not_invoke_vision_orchestrator(self):
        """Verify that when power/thermal policy denies authorization, orchestrator.process is never invoked."""
        from unittest.mock import MagicMock
        import numpy as np
        from backend.services.vision_service import VisionService
        from backend.vision.frame import VisionFrame

        strict_policy = PowerThermalPolicyManager(require_verified_telemetry=True)
        service = VisionService(power_manager=strict_policy)
        
        # Attach spy to orchestrator.process
        mock_process = MagicMock(side_effect=RuntimeError("Orchestrator should not be called when denied!"))
        service.orchestrator.process = mock_process

        frame = VisionFrame(
            data=np.zeros((30, 30, 3), dtype=np.uint8),
            width=30,
            height=30,
            channels=3,
            pixel_format="RGB",
            numerical_range=(0, 255),
            timestamp=1000,
            seq_num=1,
        )

        # Uninitialized request -> denied before reaching orchestrator
        res = await service.process_frame_async(frame)
        assert len(res.errors) > 0
        assert res.errors[0]["error"] == "BLOCKED_BY_POWER_POLICY"
        mock_process.assert_not_called()

    def test_explicit_simulation_configuration_allows_execution(self):
        """Explicit development/simulation configuration allows software-only testing without strict hardware telemetry."""
        from fastapi.testclient import TestClient
        from backend.app import create_app
        from backend.config.settings import AppSettings
        from backend.services.vision_service import reset_vision_service
        from backend.power.policy import reset_power_policy_manager

        reset_vision_service(None)
        reset_power_policy_manager(None)

        dev_settings = AppSettings(environment="development", require_verified_telemetry=False)
        app = create_app(settings=dev_settings)
        client = TestClient(app)

        resp = client.post("/api/v1/vision/process", json={"use_mock_frame": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["errors"]) == 0

        reset_vision_service(None)
        reset_power_policy_manager(None)




