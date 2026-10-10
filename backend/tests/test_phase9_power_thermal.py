"""Unit and integration tests for Phase 9: Power and Thermal Management."""

import pytest
from backend.power.contracts import (
    OperatingState,
    BatteryStatus,
    ThermalStatus,
    PeripheralType,
    WorkloadPriority,
    BatteryTelemetry,
    ThermalTelemetry,
)
from backend.power.model import PowerModel, DEFAULT_PERIPHERAL_PROFILES
from backend.power.policy import PowerThermalPolicyManager
from backend.power.synthetic import generate_battery_discharge_trace, generate_thermal_ramp_trace
from firmware.system.power import PowerManager as FirmwarePowerManager
from firmware.system.thermal import ThermalManager as FirmwareThermalManager


def test_battery_telemetry_validation():
    """Verify numeric range validation on battery contracts."""
    valid_bat = BatteryTelemetry(voltage_volts=3.85, percentage=80.0)
    assert valid_bat.voltage_volts == 3.85
    assert valid_bat.percentage == 80.0

    with pytest.raises(ValueError, match="outside physical range"):
        BatteryTelemetry(voltage_volts=6.5, percentage=50.0)

    with pytest.raises(ValueError, match="within 0-100%"):
        BatteryTelemetry(voltage_volts=3.7, percentage=120.0)


def test_thermal_telemetry_validation():
    """Verify temperature range validation on thermal telemetry."""
    valid_therm = ThermalTelemetry(temperature_celsius=42.5)
    assert valid_therm.temperature_celsius == 42.5

    with pytest.raises(ValueError, match="outside physical sensor limits"):
        ThermalTelemetry(temperature_celsius=200.0)


def test_battery_discharge_hysteresis():
    """Test battery threshold hysteresis and state transitions."""
    manager = PowerThermalPolicyManager(
        low_battery_entry_pct=15.0,
        low_battery_exit_pct=20.0,
        critical_battery_entry_pct=5.0,
        critical_battery_exit_pct=10.0,
    )

    # Initial high battery -> NORMAL
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=4.0, percentage=80.0, timestamp=100.0), current_time=100.0) == BatteryStatus.NORMAL

    # Discharge to 18% -> still NORMAL (above 15%)
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.65, percentage=18.0, timestamp=101.0), current_time=101.0) == BatteryStatus.NORMAL

    # Discharge to 14% -> LOW (triggers low battery)
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.58, percentage=14.0, timestamp=102.0), current_time=102.0) == BatteryStatus.LOW
    assert manager.requested_state == OperatingState.LOW_POWER

    # Slight bounce to 18% -> remains LOW due to hysteresis (needs >20% to exit)
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.65, percentage=18.0, timestamp=103.0), current_time=103.0) == BatteryStatus.LOW

    # Discharge to 4% -> CRITICAL
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.35, percentage=4.0, timestamp=104.0), current_time=104.0) == BatteryStatus.CRITICAL
    assert manager.requested_state == OperatingState.CRITICAL_SHUTDOWN

    # Charging starts, recovers to 8% -> remains CRITICAL (needs >10% to exit)
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.45, percentage=8.0, is_charging=False, timestamp=105.0), current_time=105.0) == BatteryStatus.CRITICAL

    # Recovers to 12% -> transitions to LOW
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.55, percentage=12.0, is_charging=False, timestamp=106.0), current_time=106.0) == BatteryStatus.LOW

    # Recovers to 25% -> returns to NORMAL
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.75, percentage=25.0, is_charging=False, timestamp=107.0), current_time=107.0) == BatteryStatus.NORMAL


def test_thermal_hysteresis_and_emergency():
    """Test thermal thresholds, throttling, emergency shutdown and cooling recovery."""
    manager = PowerThermalPolicyManager(
        warm_entry_temp_c=45.0,
        warm_exit_temp_c=40.0,
        throttle_entry_temp_c=55.0,
        throttle_exit_temp_c=48.0,
        critical_thermal_entry_c=70.0,
        critical_thermal_exit_c=60.0,
    )

    # 35°C -> NORMAL
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=35.0, timestamp=10.0), current_time=10.0) == ThermalStatus.NORMAL

    # 47°C -> WARM
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=47.0, timestamp=11.0), current_time=11.0) == ThermalStatus.WARM

    # 57°C -> HOT_THROTTLED
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=57.0, timestamp=12.0), current_time=12.0) == ThermalStatus.HOT_THROTTLED

    # Cools to 50°C -> still HOT_THROTTLED (needs <48°C to exit)
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=50.0, timestamp=13.0), current_time=13.0) == ThermalStatus.HOT_THROTTLED

    # Cools to 46°C -> drops to WARM (since >=40°C)
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=46.0, timestamp=14.0), current_time=14.0) == ThermalStatus.WARM

    # Thermal spike to 72°C -> CRITICAL
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=72.0, timestamp=15.0), current_time=15.0) == ThermalStatus.CRITICAL
    assert manager.requested_state == OperatingState.CRITICAL_SHUTDOWN


def test_telemetry_staleness():
    """Verify stale telemetry triggers STALE status."""
    manager = PowerThermalPolicyManager(battery_stale_timeout_s=30.0, thermal_stale_timeout_s=15.0)

    # Battery sample timestamped at t=100.0, evaluated at t=140.0 (40s later > 30s)
    bat = BatteryTelemetry(voltage_volts=3.8, percentage=70.0, timestamp=100.0)
    assert manager.update_battery_telemetry(bat, current_time=140.0) == BatteryStatus.STALE

    # Thermal sample timestamped at t=100.0, evaluated at t=120.0 (20s later > 15s)
    therm = ThermalTelemetry(temperature_celsius=38.0, timestamp=100.0)
    assert manager.update_thermal_telemetry(therm, current_time=120.0) == ThermalStatus.STALE


def test_state_transition_validation():
    """Verify allowed state transitions and prohibition of illegal transitions."""
    manager = PowerThermalPolicyManager()

    # IDLE -> LISTENING (Allowed)
    ok, msg = manager.request_state(OperatingState.LISTENING)
    assert ok is True
    assert manager.requested_state == OperatingState.LISTENING

    # LISTENING -> CAPTURE (Illegal direct transition without going through PROCESSING or IDLE)
    ok, msg = manager.request_state(OperatingState.CAPTURE)
    assert ok is False
    assert "Invalid transition" in msg

    # LISTENING -> PROCESSING (Allowed)
    ok, _ = manager.request_state(OperatingState.PROCESSING)
    assert ok is True

    # PROCESSING -> SPEAKING (Allowed)
    ok, _ = manager.request_state(OperatingState.SPEAKING)
    assert ok is True

    # SPEAKING -> IDLE (Allowed)
    ok, _ = manager.request_state(OperatingState.IDLE)
    assert ok is True


def test_workload_prioritization():
    """Test workload permission matrix under normal, low power, and thermal throttled conditions."""
    manager = PowerThermalPolicyManager()

    # In NORMAL state: all workloads permitted
    assert manager.can_execute_workload(WorkloadPriority.SAFETY_CRITICAL)[0] is True
    assert manager.can_execute_workload(WorkloadPriority.TIME_SENSITIVE_AUDIO)[0] is True
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True
    assert manager.can_execute_workload(WorkloadPriority.BACKGROUND_SYNC)[0] is True

    # In LOW_BATTERY condition: Vision capture & background sync blocked
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.5, percentage=12.0, timestamp=1.0), current_time=1.0)
    assert manager.can_execute_workload(WorkloadPriority.SAFETY_CRITICAL)[0] is True
    assert manager.can_execute_workload(WorkloadPriority.TIME_SENSITIVE_AUDIO)[0] is True
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is False
    assert manager.can_execute_workload(WorkloadPriority.BACKGROUND_SYNC)[0] is False

    # In HOT_THROTTLED condition: Vision capture blocked
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.9, percentage=80.0, timestamp=2.0), current_time=2.0)
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=58.0, timestamp=2.0), current_time=2.0)
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is False


def test_power_model_estimates():
    """Test modeled power consumption values and runtime projections."""
    model = PowerModel(battery_capacity_mah=500.0, nominal_voltage_volts=3.7)

    # IDLE state power: only IMU active + peripherals standby
    idle_mw = model.estimate_state_power_mw(OperatingState.IDLE)
    assert 10.0 < idle_mw < 35.0  # Approx ~20mW

    # CAPTURE state power: ESP32 + WiFi + Camera + IMU active
    capture_mw = model.estimate_state_power_mw(OperatingState.CAPTURE)
    assert capture_mw > idle_mw
    assert capture_mw > 500.0  # High consumption with WiFi & Camera

    # Runtime calculation
    runtime_idle = model.estimate_runtime_hours(OperatingState.IDLE, remaining_percentage=100.0)
    runtime_capture = model.estimate_runtime_hours(OperatingState.CAPTURE, remaining_percentage=100.0)
    assert runtime_idle > runtime_capture
    assert runtime_idle > 20.0  # High standby runtime


def test_legacy_firmware_compatibility():
    """Test compatibility with existing firmware.system wrappers."""
    p = FirmwarePowerManager()
    assert p.check_battery(5) == "CRITICAL"
    assert p.check_battery(15) == "LOW_POWER"
    assert p.check_battery(80) == "ACTIVE"

    t = FirmwareThermalManager()
    assert t.check_temp(35) == "NORMAL"
    assert t.check_temp(65) == "WARM"
    assert t.check_temp(85) == "CRITICAL"


def test_battery_recovery_cannot_override_critical_thermal():
    """Safety Invariant: Critical thermal protection must NEVER be overridden by battery recharge."""
    manager = PowerThermalPolicyManager()

    # Trigger critical thermal condition (72°C)
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=72.0, timestamp=1.0), current_time=1.0)
    assert manager.thermal_status == ThermalStatus.CRITICAL
    assert manager.requested_state == OperatingState.CRITICAL_SHUTDOWN

    # Battery discharges to low
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.5, percentage=12.0, timestamp=2.0), current_time=2.0)
    assert manager.battery_status == BatteryStatus.LOW
    assert manager.requested_state == OperatingState.CRITICAL_SHUTDOWN

    # Battery now recharges to 100% while temperature is still critical (72°C)
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=4.2, percentage=100.0, is_charging=True, timestamp=3.0), current_time=3.0)
    assert manager.battery_status == BatteryStatus.FULL
    # MUST remain in CRITICAL_SHUTDOWN
    assert manager.requested_state == OperatingState.CRITICAL_SHUTDOWN
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is False

    # Manual attempt to request IDLE must also be rejected
    ok, msg = manager.request_state(OperatingState.IDLE)
    assert ok is False
    assert "Thermal status is CRITICAL" in msg


def test_battery_recovery_cannot_override_hot_throttled():
    """Safety Invariant: Battery recharge must not clear active thermal throttling restrictions."""
    manager = PowerThermalPolicyManager()

    # Enter thermal throttling (58°C) & low battery (14%)
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=58.0, timestamp=1.0), current_time=1.0)
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.55, percentage=14.0, timestamp=1.0), current_time=1.0)
    assert manager.thermal_status == ThermalStatus.HOT_THROTTLED
    assert manager.battery_status == BatteryStatus.LOW
    assert manager.requested_state == OperatingState.LOW_POWER

    # Battery recovers to 85%
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=4.0, percentage=85.0, timestamp=2.0), current_time=2.0)
    assert manager.battery_status == BatteryStatus.NORMAL

    # State must not be IDLE because thermal throttling remains active
    assert manager.requested_state == OperatingState.LOW_POWER
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is False

    # Direct request to CAPTURE state must be rejected from LOW_POWER
    ok, msg = manager.request_state(OperatingState.CAPTURE)
    assert ok is False

    # Also verify from IDLE state: if manager is IDLE and HOT_THROTTLED, CAPTURE is rejected
    idle_manager = PowerThermalPolicyManager()
    idle_manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=58.0, timestamp=1.0), current_time=1.0)
    assert idle_manager.thermal_status == ThermalStatus.HOT_THROTTLED
    ok2, msg2 = idle_manager.request_state(OperatingState.CAPTURE)
    assert ok2 is False
    assert "HOT_THROTTLED" in msg2


def test_thermal_recovery_preserves_low_battery_restriction():
    """Safety Invariant: Cooling down must not bypass an active low-battery restriction."""
    manager = PowerThermalPolicyManager()

    # Set low battery (12%)
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.5, percentage=12.0, timestamp=1.0), current_time=1.0)
    assert manager.battery_status == BatteryStatus.LOW
    assert manager.requested_state == OperatingState.LOW_POWER

    # Thermal spike to critical (75°C) -> CRITICAL_SHUTDOWN
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=75.0, timestamp=2.0), current_time=2.0)
    assert manager.thermal_status == ThermalStatus.CRITICAL
    assert manager.requested_state == OperatingState.CRITICAL_SHUTDOWN

    # Thermal cools down to 32°C (NORMAL) while battery remains 12% (LOW)
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=32.0, timestamp=3.0), current_time=3.0)
    assert manager.thermal_status == ThermalStatus.NORMAL
    # Must recover to LOW_POWER, NOT to IDLE!
    assert manager.requested_state == OperatingState.LOW_POWER
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is False


def test_full_recovery_from_critical_shutdown():
    """Verify dual recovery when both thermal and battery conditions return to safe operating levels."""
    manager = PowerThermalPolicyManager()

    # Both critical: 3% battery, 74°C die temp
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.3, percentage=3.0, timestamp=1.0), current_time=1.0)
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=74.0, timestamp=1.0), current_time=1.0)
    assert manager.battery_status == BatteryStatus.CRITICAL
    assert manager.thermal_status == ThermalStatus.CRITICAL
    assert manager.requested_state == OperatingState.CRITICAL_SHUTDOWN

    # Thermal cools first to 30°C, but battery still critical -> remains CRITICAL_SHUTDOWN
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=30.0, timestamp=2.0), current_time=2.0)
    assert manager.thermal_status == ThermalStatus.NORMAL
    assert manager.requested_state == OperatingState.CRITICAL_SHUTDOWN

    # Battery now recharges to 70% -> full recovery to IDLE
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.85, percentage=70.0, timestamp=3.0), current_time=3.0)
    assert manager.battery_status == BatteryStatus.NORMAL
    assert manager.requested_state == OperatingState.IDLE
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True


@pytest.mark.parametrize(
    "bat_pct,temp_c,expected_bat,expected_therm,expected_state,vision_allowed",
    [
        (85.0, 32.0, BatteryStatus.NORMAL, ThermalStatus.NORMAL, OperatingState.IDLE, True),
        (14.0, 32.0, BatteryStatus.LOW, ThermalStatus.NORMAL, OperatingState.LOW_POWER, False),
        (3.0, 32.0, BatteryStatus.CRITICAL, ThermalStatus.NORMAL, OperatingState.CRITICAL_SHUTDOWN, False),
        (85.0, 47.0, BatteryStatus.NORMAL, ThermalStatus.WARM, OperatingState.IDLE, True),
        (85.0, 58.0, BatteryStatus.NORMAL, ThermalStatus.HOT_THROTTLED, OperatingState.IDLE, False),
        (85.0, 72.0, BatteryStatus.NORMAL, ThermalStatus.CRITICAL, OperatingState.CRITICAL_SHUTDOWN, False),
        (14.0, 58.0, BatteryStatus.LOW, ThermalStatus.HOT_THROTTLED, OperatingState.LOW_POWER, False),
        (14.0, 72.0, BatteryStatus.LOW, ThermalStatus.CRITICAL, OperatingState.CRITICAL_SHUTDOWN, False),
        (3.0, 72.0, BatteryStatus.CRITICAL, ThermalStatus.CRITICAL, OperatingState.CRITICAL_SHUTDOWN, False),
    ],
)
def test_power_thermal_matrix_interactions(bat_pct, temp_c, expected_bat, expected_therm, expected_state, vision_allowed):
    """Table-driven verification of multi-variable power and thermal interaction states."""
    manager = PowerThermalPolicyManager()
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.3 + (bat_pct / 100.0) * 0.9, percentage=bat_pct, timestamp=1.0), current_time=1.0)
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=temp_c, timestamp=1.0), current_time=1.0)

    assert manager.battery_status == expected_bat
    assert manager.thermal_status == expected_therm
    assert manager.requested_state == expected_state
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is vision_allowed


def test_missing_and_invalid_telemetry_handling():
    """Verify system safety when telemetry is missing or marked invalid."""
    manager = PowerThermalPolicyManager()

    # Initial snapshot without any telemetry received
    snap = manager.get_snapshot(current_time=50.0)
    assert snap.battery_status == BatteryStatus.NORMAL
    assert snap.thermal_status == ThermalStatus.NORMAL
    assert snap.battery.percentage == 75.0  # Safe default fallback

    # Invalid battery reading
    invalid_bat = BatteryTelemetry(voltage_volts=3.7, percentage=50.0, is_valid=False, timestamp=51.0)
    assert manager.update_battery_telemetry(invalid_bat, current_time=51.0) == BatteryStatus.UNKNOWN

    # Invalid thermal reading
    invalid_therm = ThermalTelemetry(temperature_celsius=35.0, is_valid=False, timestamp=51.0)
    assert manager.update_thermal_telemetry(invalid_therm, current_time=51.0) == ThermalStatus.UNKNOWN


def test_telemetry_staleness_and_recovery_flow():
    """Test telemetry staleness transitions during pending workloads and recovery when fresh telemetry resumes."""
    manager = PowerThermalPolicyManager(battery_stale_timeout_s=30.0, thermal_stale_timeout_s=15.0)

    # Initial healthy telemetry at t=100.0
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=4.0, percentage=80.0, timestamp=100.0), current_time=100.0)
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=35.0, timestamp=100.0), current_time=100.0)
    assert manager.battery_status == BatteryStatus.NORMAL
    assert manager.thermal_status == ThermalStatus.NORMAL
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True

    # Thermal becomes stale at t=120.0 (>15s elapsed)
    therm_stale = ThermalTelemetry(temperature_celsius=35.0, timestamp=100.0)
    assert manager.update_thermal_telemetry(therm_stale, current_time=120.0) == ThermalStatus.STALE

    # Battery becomes stale at t=140.0 (>30s elapsed)
    bat_stale = BatteryTelemetry(voltage_volts=4.0, percentage=80.0, timestamp=100.0)
    assert manager.update_battery_telemetry(bat_stale, current_time=140.0) == BatteryStatus.STALE

    # Fresh thermal telemetry arrives at t=150.0 with critical spike -> triggers CRITICAL_SHUTDOWN
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=72.0, timestamp=150.0), current_time=150.0)
    assert manager.thermal_status == ThermalStatus.CRITICAL
    assert manager.requested_state == OperatingState.CRITICAL_SHUTDOWN
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is False

    # Fresh thermal normalizes at t=160.0 & fresh battery confirms healthy -> full recovery to IDLE
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=35.0, timestamp=160.0), current_time=160.0)
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.9, percentage=75.0, timestamp=160.0), current_time=160.0)
    assert manager.thermal_status == ThermalStatus.NORMAL
    assert manager.battery_status == BatteryStatus.NORMAL
    assert manager.requested_state == OperatingState.IDLE
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True


def test_repeated_low_and_recovered_battery_cycles():
    """Verify repeated oscillation between low battery and recovery does not leave inconsistent state."""
    manager = PowerThermalPolicyManager(low_battery_entry_pct=15.0, low_battery_exit_pct=20.0)

    # Cycle 1: Discharges to 14% -> LOW_POWER
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.58, percentage=14.0, timestamp=1.0), current_time=1.0)
    assert manager.battery_status == BatteryStatus.LOW
    assert manager.requested_state == OperatingState.LOW_POWER
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is False

    # Cycle 1: Recharges to 25% -> IDLE
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.75, percentage=25.0, timestamp=2.0), current_time=2.0)
    assert manager.battery_status == BatteryStatus.NORMAL
    assert manager.requested_state == OperatingState.IDLE
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True

    # Cycle 2: Discharges to 13% -> LOW_POWER
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.56, percentage=13.0, timestamp=3.0), current_time=3.0)
    assert manager.battery_status == BatteryStatus.LOW
    assert manager.requested_state == OperatingState.LOW_POWER
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is False

    # Cycle 2: Recharges to 80% -> IDLE
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=4.0, percentage=80.0, timestamp=4.0), current_time=4.0)
    assert manager.battery_status == BatteryStatus.NORMAL
    assert manager.requested_state == OperatingState.IDLE
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True

    # Cycle 3: Critical discharge to 4% -> CRITICAL_SHUTDOWN
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.35, percentage=4.0, timestamp=5.0), current_time=5.0)
    assert manager.battery_status == BatteryStatus.CRITICAL
    assert manager.requested_state == OperatingState.CRITICAL_SHUTDOWN

    # Cycle 3: Partial recharge to 12% (above critical exit >10%, but <=20% low exit) -> LOW status, stays in shutdown until normal recovery
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.55, percentage=12.0, timestamp=6.0), current_time=6.0)
    assert manager.battery_status == BatteryStatus.LOW

    # Cycle 3: Full recharge to 85% -> returns to IDLE
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=4.05, percentage=85.0, timestamp=7.0), current_time=7.0)
    assert manager.battery_status == BatteryStatus.NORMAL
    assert manager.requested_state == OperatingState.IDLE
    assert manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True


def test_battery_boundary_threshold_precision():
    """Verify precise behavior exactly at, immediately above, and immediately below battery thresholds."""
    manager = PowerThermalPolicyManager(
        low_battery_entry_pct=15.0,
        low_battery_exit_pct=20.0,
        critical_battery_entry_pct=5.0,
        critical_battery_exit_pct=10.0,
    )

    # 1. Low entry threshold (15.0%)
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.61, percentage=15.1, timestamp=1.0), current_time=1.0) == BatteryStatus.NORMAL
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.60, percentage=15.0, timestamp=2.0), current_time=2.0) == BatteryStatus.LOW
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.59, percentage=14.9, timestamp=3.0), current_time=3.0) == BatteryStatus.LOW

    # 2. Low recovery threshold (20.0%, requires >20.0% to exit)
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.69, percentage=19.9, timestamp=4.0), current_time=4.0) == BatteryStatus.LOW
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.70, percentage=20.0, timestamp=5.0), current_time=5.0) == BatteryStatus.LOW
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.71, percentage=20.1, timestamp=6.0), current_time=6.0) == BatteryStatus.NORMAL

    # 3. Critical entry threshold (5.0%)
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.55, percentage=12.0, timestamp=7.0), current_time=7.0)
    assert manager.battery_status == BatteryStatus.LOW
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.41, percentage=5.1, timestamp=8.0), current_time=8.0) == BatteryStatus.LOW
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.40, percentage=5.0, timestamp=9.0), current_time=9.0) == BatteryStatus.CRITICAL

    # 4. Critical recovery threshold (10.0%, requires >10.0% to exit)
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.49, percentage=9.9, timestamp=10.0), current_time=10.0) == BatteryStatus.CRITICAL
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.50, percentage=10.0, timestamp=11.0), current_time=11.0) == BatteryStatus.CRITICAL
    assert manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.51, percentage=10.1, timestamp=12.0), current_time=12.0) == BatteryStatus.LOW


def test_thermal_boundary_threshold_precision():
    """Verify precise behavior exactly at, immediately above, and immediately below thermal thresholds."""
    manager = PowerThermalPolicyManager(
        warm_entry_temp_c=45.0,
        warm_exit_temp_c=40.0,
        throttle_entry_temp_c=55.0,
        throttle_exit_temp_c=48.0,
        critical_thermal_entry_c=70.0,
        critical_thermal_exit_c=60.0,
    )

    # 1. Warm entry (45.0°C) and recovery (40.0°C)
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=44.9, timestamp=1.0), current_time=1.0) == ThermalStatus.NORMAL
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=45.0, timestamp=2.0), current_time=2.0) == ThermalStatus.WARM
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=40.1, timestamp=3.0), current_time=3.0) == ThermalStatus.WARM
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=40.0, timestamp=4.0), current_time=4.0) == ThermalStatus.WARM
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=39.9, timestamp=5.0), current_time=5.0) == ThermalStatus.NORMAL

    # 2. Throttle entry (55.0°C) and recovery (48.0°C)
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=54.9, timestamp=6.0), current_time=6.0) == ThermalStatus.WARM
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=55.0, timestamp=7.0), current_time=7.0) == ThermalStatus.HOT_THROTTLED
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=48.1, timestamp=8.0), current_time=8.0) == ThermalStatus.HOT_THROTTLED
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=48.0, timestamp=9.0), current_time=9.0) == ThermalStatus.HOT_THROTTLED
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=47.9, timestamp=10.0), current_time=10.0) == ThermalStatus.WARM

    # 3. Critical entry (70.0°C) and recovery (60.0°C)
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=69.9, timestamp=11.0), current_time=11.0) == ThermalStatus.HOT_THROTTLED
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=70.0, timestamp=12.0), current_time=12.0) == ThermalStatus.CRITICAL
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=60.1, timestamp=13.0), current_time=13.0) == ThermalStatus.CRITICAL
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=60.0, timestamp=14.0), current_time=14.0) == ThermalStatus.CRITICAL
    assert manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=59.9, timestamp=15.0), current_time=15.0) == ThermalStatus.HOT_THROTTLED


def test_authorization_under_missing_stale_and_recovering_telemetry():
    """Verify workload authorization across all 12 missing, stale, invalid, and recovering telemetry scenarios."""

    # Scenario 1: A newly initialized policy with no real telemetry
    m1 = PowerThermalPolicyManager()
    assert m1.battery_status == BatteryStatus.NORMAL
    assert m1.thermal_status == ThermalStatus.NORMAL
    assert m1.requested_state == OperatingState.IDLE
    assert m1.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True
    assert m1.request_state(OperatingState.CAPTURE)[0] is True

    # Scenario 2: Battery telemetry missing while temperature telemetry is valid
    m2 = PowerThermalPolicyManager()
    m2.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=34.0, timestamp=10.0), current_time=10.0)
    assert m2.last_battery is None
    assert m2.last_thermal is not None
    assert m2.thermal_status == ThermalStatus.NORMAL
    assert m2.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True

    # Scenario 3: Temperature telemetry missing while battery telemetry is valid
    m3 = PowerThermalPolicyManager()
    m3.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.9, percentage=80.0, timestamp=10.0), current_time=10.0)
    assert m3.last_thermal is None
    assert m3.last_battery is not None
    assert m3.battery_status == BatteryStatus.NORMAL
    assert m3.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True

    # Scenario 4: Both telemetry streams missing
    m4 = PowerThermalPolicyManager()
    snap = m4.get_snapshot(current_time=10.0)
    assert snap.battery_status == BatteryStatus.NORMAL
    assert snap.thermal_status == ThermalStatus.NORMAL
    assert m4.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True

    # Scenario 5: Invalid battery readings
    m5 = PowerThermalPolicyManager()
    inv_bat = BatteryTelemetry(voltage_volts=3.7, percentage=50.0, is_valid=False, timestamp=10.0)
    assert m5.update_battery_telemetry(inv_bat, current_time=10.0) == BatteryStatus.UNKNOWN

    # Scenario 6: Invalid temperature readings
    m6 = PowerThermalPolicyManager()
    inv_therm = ThermalTelemetry(temperature_celsius=35.0, is_valid=False, timestamp=10.0)
    assert m6.update_thermal_telemetry(inv_therm, current_time=10.0) == ThermalStatus.UNKNOWN

    # Scenario 7: Battery telemetry becoming stale while a vision request is pending
    m7 = PowerThermalPolicyManager(battery_stale_timeout_s=30.0)
    m7.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.9, percentage=80.0, timestamp=100.0), current_time=100.0)
    assert m7.battery_status == BatteryStatus.NORMAL
    # Evaluating staleness at t=140.0s (>30s) during workload check
    assert m7.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=140.0)[0] is True
    assert m7.battery_status == BatteryStatus.STALE

    # Scenario 8: Temperature telemetry becoming stale while a vision request is pending
    m8 = PowerThermalPolicyManager(thermal_stale_timeout_s=15.0)
    m8.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=35.0, timestamp=100.0), current_time=100.0)
    assert m8.thermal_status == ThermalStatus.NORMAL
    # Evaluating staleness at t=120.0s (>15s) during workload check
    assert m8.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=120.0)[0] is True
    assert m8.thermal_status == ThermalStatus.STALE

    # Scenario 9: Both readings becoming stale
    m9 = PowerThermalPolicyManager(battery_stale_timeout_s=30.0, thermal_stale_timeout_s=15.0)
    m9.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.9, percentage=80.0, timestamp=100.0), current_time=100.0)
    m9.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=35.0, timestamp=100.0), current_time=100.0)
    m9.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=150.0)
    assert m9.battery_status == BatteryStatus.STALE
    assert m9.thermal_status == ThermalStatus.STALE

    # Scenario 10: Fresh valid readings arriving after stale telemetry
    m10 = PowerThermalPolicyManager(battery_stale_timeout_s=30.0, thermal_stale_timeout_s=15.0)
    m10.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.9, percentage=80.0, timestamp=100.0), current_time=100.0)
    m10.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=140.0)
    assert m10.battery_status == BatteryStatus.STALE
    # Fresh telemetry arrives at t=150.0
    m10.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.95, percentage=82.0, timestamp=150.0), current_time=150.0)
    assert m10.battery_status == BatteryStatus.NORMAL
    assert m10.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is True

    # Scenario 11: A stale reading following a previously critical reading
    m11 = PowerThermalPolicyManager(battery_stale_timeout_s=30.0)
    # Critical battery event
    m11.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.3, percentage=4.0, timestamp=100.0), current_time=100.0)
    assert m11.battery_status == BatteryStatus.CRITICAL
    assert m11.requested_state == OperatingState.CRITICAL_SHUTDOWN
    # Time elapses to t=140.0s without fresh telemetry
    m11.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=140.0)
    assert m11.battery_status == BatteryStatus.STALE
    # Invariant: State MUST remain in CRITICAL_SHUTDOWN and vision blocked
    assert m11.requested_state == OperatingState.CRITICAL_SHUTDOWN
    assert m11.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is False
    assert m11.request_state(OperatingState.IDLE, current_time=140.0)[0] is False

    # Scenario 12: Fresh but critical telemetry arriving after stale telemetry
    m12 = PowerThermalPolicyManager(thermal_stale_timeout_s=15.0)
    m12.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=35.0, timestamp=100.0), current_time=100.0)
    m12.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=120.0)
    assert m12.thermal_status == ThermalStatus.STALE
    # Fresh critical telemetry arrives at t=130.0s
    m12.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=72.0, timestamp=130.0), current_time=130.0)
    assert m12.thermal_status == ThermalStatus.CRITICAL
    assert m12.requested_state == OperatingState.CRITICAL_SHUTDOWN
    assert m12.can_execute_workload(WorkloadPriority.VISION_CAPTURE)[0] is False


def test_out_of_order_telemetry_rejection():
    """Verify that out-of-order older telemetry packets cannot overwrite newer state."""
    manager = PowerThermalPolicyManager()

    # Newer packet arrives at t=100.0 (Battery=80%, Temp=35°C)
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.9, percentage=80.0, timestamp=100.0), current_time=100.0)
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=35.0, timestamp=100.0), current_time=100.0)
    assert manager.battery_status == BatteryStatus.NORMAL
    assert manager.thermal_status == ThermalStatus.NORMAL

    # Out-of-order older packet arrives with timestamp t=90.0 (e.g. low battery 14%)
    manager.update_battery_telemetry(BatteryTelemetry(voltage_volts=3.5, percentage=14.0, timestamp=90.0), current_time=100.0)
    # MUST reject the older packet and retain the newer 80% NORMAL status
    assert manager.battery_status == BatteryStatus.NORMAL
    assert manager.last_battery.timestamp == 100.0

    # Out-of-order older thermal packet arrives with timestamp t=80.0 (e.g. critical 75°C)
    manager.update_thermal_telemetry(ThermalTelemetry(temperature_celsius=75.0, timestamp=80.0), current_time=100.0)
    # MUST reject the older packet and retain the newer 35°C NORMAL status
    assert manager.thermal_status == ThermalStatus.NORMAL
    assert manager.last_thermal.timestamp == 100.0


