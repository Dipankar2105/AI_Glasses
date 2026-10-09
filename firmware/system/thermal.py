"""Firmware thermal manager interface for NextSight smart glasses."""

from backend.power.contracts import ThermalStatus, ThermalTelemetry
from backend.power.policy import PowerThermalPolicyManager


class ThermalManager:
    """Firmware thermal manager wrapper around backend policy engine."""

    def __init__(self, policy: PowerThermalPolicyManager = None):
        self.policy = policy or PowerThermalPolicyManager(critical_thermal_entry_c=80.0, throttle_entry_temp_c=60.0)

    def check_temp(self, temp: float) -> str:
        """Legacy compatibility check returning string status."""
        telemetry = ThermalTelemetry(
            temperature_celsius=float(temp),
            timestamp=0.0,
        )
        status = self.policy.update_thermal_telemetry(telemetry)
        if status == ThermalStatus.CRITICAL:
            return "CRITICAL"
        elif status in (ThermalStatus.HOT_THROTTLED, ThermalStatus.WARM):
            return "WARM"
        return "NORMAL"
