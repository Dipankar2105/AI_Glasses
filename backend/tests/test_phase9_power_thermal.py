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
