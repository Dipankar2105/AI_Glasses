"""Policy manager for power state transitions, battery thresholds, thermal throttling, and workload scheduling."""

import time
from typing import Optional, Set, Tuple
from backend.power.contracts import (
    OperatingState,
    BatteryStatus,
    ThermalStatus,
    WorkloadPriority,
    BatteryTelemetry,
    ThermalTelemetry,
    SystemPowerSnapshot,
    PeripheralType,
)
from backend.power.model import (
    PowerModel,
    STATE_PERIPHERAL_ACTIVATION,
)


class PowerThermalPolicyManager:
    """Manages system power states, battery/thermal threshold hysteresis, and workload gating."""

    # Valid state transition graph: from_state -> set of allowed target_states
    VALID_TRANSITIONS: dict[OperatingState, Set[OperatingState]] = {
        OperatingState.IDLE: {
            OperatingState.LISTENING,
            OperatingState.CAPTURE,
            OperatingState.PROCESSING,
            OperatingState.LOW_POWER,
            OperatingState.CRITICAL_SHUTDOWN,
        },
        OperatingState.LISTENING: {
            OperatingState.IDLE,
            OperatingState.PROCESSING,
            OperatingState.SPEAKING,
            OperatingState.LOW_POWER,
            OperatingState.CRITICAL_SHUTDOWN,
        },
        OperatingState.CAPTURE: {
            OperatingState.IDLE,
            OperatingState.PROCESSING,
            OperatingState.LOW_POWER,
            OperatingState.CRITICAL_SHUTDOWN,
        },
        OperatingState.PROCESSING: {
            OperatingState.IDLE,
            OperatingState.SPEAKING,
            OperatingState.CAPTURE,
            OperatingState.LOW_POWER,
            OperatingState.CRITICAL_SHUTDOWN,
        },
        OperatingState.SPEAKING: {
            OperatingState.IDLE,
            OperatingState.LISTENING,
            OperatingState.LOW_POWER,
            OperatingState.CRITICAL_SHUTDOWN,
        },
        OperatingState.LOW_POWER: {
            OperatingState.IDLE,
            OperatingState.CRITICAL_SHUTDOWN,
        },
        OperatingState.CRITICAL_SHUTDOWN: {
            OperatingState.IDLE,  # Permissible after battery/thermal recovery
            OperatingState.LOW_POWER,  # Permissible if thermal cools while battery is LOW
        },
    }

    def __init__(
        self,
        power_model: Optional[PowerModel] = None,
        # Battery hysteresis thresholds
        low_battery_entry_pct: float = 15.0,
        low_battery_exit_pct: float = 20.0,
        critical_battery_entry_pct: float = 5.0,
        critical_battery_exit_pct: float = 10.0,
        # Thermal hysteresis thresholds (°C)
        warm_entry_temp_c: float = 45.0,
        warm_exit_temp_c: float = 40.0,
        throttle_entry_temp_c: float = 55.0,
        throttle_exit_temp_c: float = 48.0,
        critical_thermal_entry_c: float = 70.0,
        critical_thermal_exit_c: float = 60.0,
        # Staleness thresholds (seconds)
        battery_stale_timeout_s: float = 30.0,
        thermal_stale_timeout_s: float = 15.0,
        # Strict hardware-readiness telemetry verification gate
        require_verified_telemetry: bool = False,
    ) -> None:
        self.power_model = power_model or PowerModel()
        self.require_verified_telemetry = require_verified_telemetry

        self.low_battery_entry_pct = low_battery_entry_pct
        self.low_battery_exit_pct = low_battery_exit_pct
        self.critical_battery_entry_pct = critical_battery_entry_pct
        self.critical_battery_exit_pct = critical_battery_exit_pct

        self.warm_entry_temp_c = warm_entry_temp_c
        self.warm_exit_temp_c = warm_exit_temp_c
        self.throttle_entry_temp_c = throttle_entry_temp_c
        self.throttle_exit_temp_c = throttle_exit_temp_c
        self.critical_thermal_entry_c = critical_thermal_entry_c
        self.critical_thermal_exit_c = critical_thermal_exit_c

        self.battery_stale_timeout_s = battery_stale_timeout_s
        self.thermal_stale_timeout_s = thermal_stale_timeout_s

        # State tracking
        self.requested_state = OperatingState.IDLE
        self.confirmed_state = OperatingState.IDLE
        self.battery_status = BatteryStatus.NORMAL
        self.thermal_status = ThermalStatus.NORMAL

        self.last_battery: Optional[BatteryTelemetry] = None
        self.last_thermal: Optional[ThermalTelemetry] = None

    def _evaluate_staleness(self, current_time: Optional[float] = None) -> None:
        """Evaluate age of last telemetry against staleness timeouts."""
        now = current_time if current_time is not None else time.time()
        if self.last_battery is not None:
            if not self.last_battery.is_valid:
                self.battery_status = BatteryStatus.UNKNOWN
            elif self.last_battery.timestamp > 0.0 and (now - self.last_battery.timestamp) > self.battery_stale_timeout_s:
                self.battery_status = BatteryStatus.STALE
        if self.last_thermal is not None:
            if not self.last_thermal.is_valid:
                self.thermal_status = ThermalStatus.UNKNOWN
            elif self.last_thermal.timestamp > 0.0 and (now - self.last_thermal.timestamp) > self.thermal_stale_timeout_s:
                self.thermal_status = ThermalStatus.STALE

    def update_battery_telemetry(self, telemetry: BatteryTelemetry, current_time: Optional[float] = None) -> BatteryStatus:
        """Process incoming battery telemetry with hysteresis and staleness checking."""
        now = current_time if current_time is not None else time.time()

        # Reject out-of-order older telemetry if newer telemetry is already recorded
        if self.last_battery is not None and telemetry.timestamp > 0.0 and self.last_battery.timestamp > 0.0:
            if telemetry.timestamp < self.last_battery.timestamp:
                return self.battery_status

        self.last_battery = telemetry

        if not telemetry.is_valid:
            self.battery_status = BatteryStatus.UNKNOWN
            return self.battery_status

        # Staleness check
        if telemetry.timestamp > 0.0 and (now - telemetry.timestamp) > self.battery_stale_timeout_s:
            self.battery_status = BatteryStatus.STALE
            return self.battery_status

        if telemetry.is_charging:
            if telemetry.percentage >= 98.0:
                self.battery_status = BatteryStatus.FULL
            else:
                self.battery_status = BatteryStatus.CHARGING
            return self.battery_status

        # Hysteresis state machine for battery discharge
        pct = telemetry.percentage
        if self.battery_status == BatteryStatus.CRITICAL:
            if pct > self.critical_battery_exit_pct:
                self.battery_status = BatteryStatus.LOW if pct <= self.low_battery_exit_pct else BatteryStatus.NORMAL
        elif self.battery_status == BatteryStatus.LOW:
            if pct <= self.critical_battery_entry_pct:
                self.battery_status = BatteryStatus.CRITICAL
            elif pct > self.low_battery_exit_pct:
                self.battery_status = BatteryStatus.NORMAL
        else:  # NORMAL, FULL, CHARGING, UNKNOWN, STALE
            if pct <= self.critical_battery_entry_pct:
                self.battery_status = BatteryStatus.CRITICAL
            elif pct <= self.low_battery_entry_pct:
                self.battery_status = BatteryStatus.LOW
            else:
                self.battery_status = BatteryStatus.NORMAL

        # Auto-enforce low power or shutdown on critical status
        if self.battery_status == BatteryStatus.CRITICAL:
            self.request_state(OperatingState.CRITICAL_SHUTDOWN, reason="Critical battery discharge", current_time=now)
        elif self.battery_status == BatteryStatus.LOW and self.requested_state not in (OperatingState.LOW_POWER, OperatingState.CRITICAL_SHUTDOWN):
            self.request_state(OperatingState.LOW_POWER, reason="Low battery conservation", current_time=now)
        elif self.battery_status in (BatteryStatus.NORMAL, BatteryStatus.FULL, BatteryStatus.CHARGING) and self.requested_state in (OperatingState.LOW_POWER, OperatingState.CRITICAL_SHUTDOWN):
            if self.thermal_status not in (ThermalStatus.CRITICAL, ThermalStatus.HOT_THROTTLED):
                self.request_state(OperatingState.IDLE, reason="Battery recovered to normal operating level", current_time=now)

        return self.battery_status

    def update_thermal_telemetry(self, telemetry: ThermalTelemetry, current_time: Optional[float] = None) -> ThermalStatus:
        """Process incoming thermal telemetry with hysteresis and emergency safeguards."""
        now = current_time if current_time is not None else time.time()

        # Reject out-of-order older telemetry if newer telemetry is already recorded
        if self.last_thermal is not None and telemetry.timestamp > 0.0 and self.last_thermal.timestamp > 0.0:
            if telemetry.timestamp < self.last_thermal.timestamp:
                return self.thermal_status

        self.last_thermal = telemetry

        if not telemetry.is_valid:
            self.thermal_status = ThermalStatus.UNKNOWN
            return self.thermal_status

        # Staleness check
        if telemetry.timestamp > 0.0 and (now - telemetry.timestamp) > self.thermal_stale_timeout_s:
            self.thermal_status = ThermalStatus.STALE
            return self.thermal_status

        temp = telemetry.temperature_celsius

        # Hysteresis state machine for thermal tracking
        if self.thermal_status == ThermalStatus.CRITICAL:
            if temp < self.critical_thermal_exit_c:
                if temp >= self.throttle_entry_temp_c:
                    self.thermal_status = ThermalStatus.HOT_THROTTLED
                elif temp >= self.warm_entry_temp_c:
                    self.thermal_status = ThermalStatus.WARM
                else:
                    self.thermal_status = ThermalStatus.NORMAL
        elif self.thermal_status == ThermalStatus.HOT_THROTTLED:
            if temp >= self.critical_thermal_entry_c:
                self.thermal_status = ThermalStatus.CRITICAL
            elif temp < self.throttle_exit_temp_c:
                self.thermal_status = ThermalStatus.WARM if temp >= self.warm_entry_temp_c else ThermalStatus.NORMAL
        elif self.thermal_status == ThermalStatus.WARM:
            if temp >= self.critical_thermal_entry_c:
                self.thermal_status = ThermalStatus.CRITICAL
            elif temp >= self.throttle_entry_temp_c:
                self.thermal_status = ThermalStatus.HOT_THROTTLED
            elif temp < self.warm_exit_temp_c:
                self.thermal_status = ThermalStatus.NORMAL
        else:  # NORMAL, UNKNOWN, STALE
            if temp >= self.critical_thermal_entry_c:
                self.thermal_status = ThermalStatus.CRITICAL
            elif temp >= self.throttle_entry_temp_c:
                self.thermal_status = ThermalStatus.HOT_THROTTLED
            elif temp >= self.warm_entry_temp_c:
                self.thermal_status = ThermalStatus.WARM
            else:
                self.thermal_status = ThermalStatus.NORMAL

        # Emergency cutoff if critical thermal condition or recovery handling
        if self.thermal_status == ThermalStatus.CRITICAL:
            self.request_state(OperatingState.CRITICAL_SHUTDOWN, reason=f"Critical die temperature: {temp:.1f}°C", current_time=now)
        elif self.thermal_status in (ThermalStatus.NORMAL, ThermalStatus.WARM) and self.requested_state == OperatingState.CRITICAL_SHUTDOWN:
            if self.battery_status in (BatteryStatus.NORMAL, BatteryStatus.FULL, BatteryStatus.CHARGING):
                self.request_state(OperatingState.IDLE, reason="Thermal conditions normalized", current_time=now)
            elif self.battery_status == BatteryStatus.LOW:
                self.request_state(OperatingState.LOW_POWER, reason="Thermal conditions normalized while battery is low", current_time=now)

        return self.thermal_status

    def request_state(self, target_state: OperatingState, reason: str = "", current_time: Optional[float] = None) -> Tuple[bool, str]:
        """Request a transition to target operating state with precondition and safety checks."""
        if current_time is not None:
            self._evaluate_staleness(current_time)

        # 1. Check if already in target state
        if self.requested_state == target_state:
            return True, f"Already in requested state {target_state}"

        # 2. Block escalation if in critical conditions
        if self.battery_status == BatteryStatus.CRITICAL and target_state != OperatingState.CRITICAL_SHUTDOWN:
            return False, f"Cannot enter {target_state}: Battery is CRITICAL"
        if self.thermal_status == ThermalStatus.CRITICAL and target_state != OperatingState.CRITICAL_SHUTDOWN:
            return False, f"Cannot enter {target_state}: Thermal status is CRITICAL"

        # 3. Block recovery from emergency shutdown without verified fresh healthy telemetry
        if self.requested_state == OperatingState.CRITICAL_SHUTDOWN and target_state != OperatingState.CRITICAL_SHUTDOWN:
            if self.battery_status in (BatteryStatus.CRITICAL, BatteryStatus.STALE, BatteryStatus.UNKNOWN):
                return False, f"Cannot exit CRITICAL_SHUTDOWN without fresh valid battery telemetry (status: {self.battery_status.value})"
            if self.thermal_status in (ThermalStatus.CRITICAL, ThermalStatus.STALE, ThermalStatus.UNKNOWN):
                return False, f"Cannot exit CRITICAL_SHUTDOWN without fresh valid thermal telemetry (status: {self.thermal_status.value})"

        # 4. Check valid transition graph
        allowed_transitions = self.VALID_TRANSITIONS.get(self.requested_state, set())
        if target_state not in allowed_transitions:
            return False, f"Invalid transition from {self.requested_state} to {target_state}"

        # 5. Low-power and thermal throttling restrictions
        if self.battery_status == BatteryStatus.LOW and target_state in (OperatingState.CAPTURE, OperatingState.PROCESSING):
            return False, f"Cannot enter heavy workload state {target_state} during LOW battery"
        if self.thermal_status == ThermalStatus.HOT_THROTTLED and target_state == OperatingState.CAPTURE:
            return False, f"Cannot enter heavy workload state {target_state} during HOT_THROTTLED thermal status"
        if self.requested_state == OperatingState.LOW_POWER and target_state == OperatingState.IDLE:
            if self.battery_status in (BatteryStatus.LOW, BatteryStatus.CRITICAL, BatteryStatus.STALE, BatteryStatus.UNKNOWN):
                return False, f"Cannot exit LOW_POWER to IDLE while battery status is {self.battery_status.value}"
            if self.thermal_status in (ThermalStatus.CRITICAL, ThermalStatus.HOT_THROTTLED, ThermalStatus.STALE, ThermalStatus.UNKNOWN):
                return False, f"Cannot exit LOW_POWER to IDLE while thermal status is {self.thermal_status.value}"

        # 6. Hardware telemetry verification gate
        if self.require_verified_telemetry and target_state in (OperatingState.CAPTURE, OperatingState.PROCESSING):
            if self.last_battery is None or self.last_thermal is None:
                return False, "Hardware telemetry uninitialized; physical state transition blocked"
            if self.battery_status in (BatteryStatus.UNKNOWN, BatteryStatus.STALE):
                return False, f"Battery telemetry is {self.battery_status.value}; physical state transition blocked"
            if self.thermal_status in (ThermalStatus.UNKNOWN, ThermalStatus.STALE):
                return False, f"Thermal telemetry is {self.thermal_status.value}; physical state transition blocked"

        self.requested_state = target_state
        # In software simulation, confirmed state updates immediately upon policy approval
        self.confirmed_state = target_state
        return True, f"Transitioned to {target_state} (Reason: {reason})"

    def confirm_hardware_state(self, hardware_state: OperatingState) -> None:
        """Acknowledge hardware-confirmed operating state."""
        self.confirmed_state = hardware_state

    def can_execute_workload(self, priority: WorkloadPriority, current_time: Optional[float] = None) -> Tuple[bool, str]:
        """Check whether a given workload is permitted under current power/thermal constraints."""
        if current_time is not None:
            self._evaluate_staleness(current_time)

        if self.require_verified_telemetry and priority in (WorkloadPriority.VISION_CAPTURE, WorkloadPriority.BACKGROUND_SYNC):
            if self.last_battery is None or self.last_thermal is None:
                return False, "Hardware telemetry uninitialized; physical high-power workload blocked"
            if self.battery_status in (BatteryStatus.UNKNOWN, BatteryStatus.STALE):
                return False, f"Battery telemetry is {self.battery_status.value}; physical high-power workload blocked"
            if self.thermal_status in (ThermalStatus.UNKNOWN, ThermalStatus.STALE):
                return False, f"Thermal telemetry is {self.thermal_status.value}; physical high-power workload blocked"

        if self.battery_status == BatteryStatus.CRITICAL or self.thermal_status == ThermalStatus.CRITICAL:
            if priority == WorkloadPriority.SAFETY_CRITICAL:
                return True, "Safety critical operation permitted during emergency"
            return False, "System in critical condition; non-essential workloads suspended"

        if self.requested_state == OperatingState.CRITICAL_SHUTDOWN:
            if priority == WorkloadPriority.SAFETY_CRITICAL:
                return True, "Safety critical operation permitted during emergency"
            return False, "System in CRITICAL_SHUTDOWN state; non-essential workloads suspended"

        if priority == WorkloadPriority.SAFETY_CRITICAL:
            return True, "Permitted"

        if priority == WorkloadPriority.TIME_SENSITIVE_AUDIO:
            if self.battery_status == BatteryStatus.LOW:
                return True, "Permitted in low power mode with high priority"
            return True, "Permitted"

        if priority == WorkloadPriority.SENSOR_PROCESSING:
            return True, "Permitted"

        if priority == WorkloadPriority.VISION_CAPTURE:
            if self.battery_status == BatteryStatus.LOW:
                return False, "Vision capture disabled to conserve low battery"
            if self.thermal_status == ThermalStatus.HOT_THROTTLED:
                return False, "Vision capture throttled due to high thermal load"
            if self.requested_state == OperatingState.LOW_POWER:
                return False, "Vision capture disallowed in LOW_POWER state"
            return True, "Permitted"

        if priority == WorkloadPriority.BACKGROUND_SYNC:
            if self.battery_status in (BatteryStatus.LOW, BatteryStatus.CRITICAL):
                return False, "Background sync suspended during low battery"
            if self.thermal_status in (ThermalStatus.HOT_THROTTLED, ThermalStatus.CRITICAL):
                return False, "Background sync suspended during thermal throttling"
            if self.requested_state != OperatingState.IDLE:
                return False, "Background sync deferred during active interaction"
            return True, "Permitted"

        return True, "Permitted"

    def get_snapshot(self, current_time: Optional[float] = None) -> SystemPowerSnapshot:
        """Produce consolidated snapshot of system power, thermal, and peripheral activations."""
        now = current_time if current_time is not None else time.time()

        bat = self.last_battery or BatteryTelemetry(voltage_volts=3.8, percentage=75.0, timestamp=now)
        therm = self.last_thermal or ThermalTelemetry(temperature_celsius=32.0, timestamp=now)

        power_mw = self.power_model.estimate_state_power_mw(self.confirmed_state)

        throttle_factor = 1.0
        if self.thermal_status == ThermalStatus.HOT_THROTTLED:
            throttle_factor = 0.5
        elif self.thermal_status == ThermalStatus.CRITICAL:
            throttle_factor = 0.0

        active_peripherals = STATE_PERIPHERAL_ACTIVATION.get(self.confirmed_state, {}).copy()

        return SystemPowerSnapshot(
            requested_state=self.requested_state,
            confirmed_state=self.confirmed_state,
            battery_status=self.battery_status,
            thermal_status=self.thermal_status,
            battery=bat,
            thermal=therm,
            total_estimated_power_mw=power_mw,
            throttle_factor=throttle_factor,
            active_peripherals=active_peripherals,
            timestamp=now,
        )


_power_policy_instance: Optional[PowerThermalPolicyManager] = None

def get_power_policy_manager(require_verified_telemetry: Optional[bool] = None) -> PowerThermalPolicyManager:
    """Singleton getter for PowerThermalPolicyManager, resolving telemetry strictness from settings if unset."""
    global _power_policy_instance
    if _power_policy_instance is None:
        strict = False
        if require_verified_telemetry is not None:
            strict = require_verified_telemetry
        else:
            try:
                from backend.config.settings import get_settings
                strict = get_settings().is_strict_telemetry_required
            except Exception:
                strict = False
        _power_policy_instance = PowerThermalPolicyManager(require_verified_telemetry=strict)
    return _power_policy_instance

def reset_power_policy_manager(manager: Optional[PowerThermalPolicyManager] = None) -> None:
    """Resets or overrides the PowerThermalPolicyManager singleton."""
    global _power_policy_instance
    _power_policy_instance = manager

