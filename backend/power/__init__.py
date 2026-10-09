"""NextSight Power and Thermal Management Package."""

from backend.power.contracts import (
    OperatingState,
    BatteryStatus,
    ThermalStatus,
    PeripheralType,
    WorkloadPriority,
    BatteryTelemetry,
    ThermalTelemetry,
    PeripheralPowerProfile,
    SystemPowerSnapshot,
)
from backend.power.model import PowerModel, DEFAULT_PERIPHERAL_PROFILES, STATE_PERIPHERAL_ACTIVATION
from backend.power.policy import PowerThermalPolicyManager
from backend.power.synthetic import generate_battery_discharge_trace, generate_thermal_ramp_trace

__all__ = [
    "OperatingState",
    "BatteryStatus",
    "ThermalStatus",
    "PeripheralType",
    "WorkloadPriority",
    "BatteryTelemetry",
    "ThermalTelemetry",
    "PeripheralPowerProfile",
    "SystemPowerSnapshot",
    "PowerModel",
    "DEFAULT_PERIPHERAL_PROFILES",
    "STATE_PERIPHERAL_ACTIVATION",
    "PowerThermalPolicyManager",
    "generate_battery_discharge_trace",
    "generate_thermal_ramp_trace",
]
