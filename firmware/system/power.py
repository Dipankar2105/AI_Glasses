"""Firmware power manager interface for NextSight smart glasses."""

from backend.power.contracts import OperatingState, BatteryStatus, BatteryTelemetry
from backend.power.policy import PowerThermalPolicyManager


class PowerManager:
    """Firmware power manager wrapper around backend policy engine."""

    def __init__(self, policy: PowerThermalPolicyManager = None):
        self.policy = policy or PowerThermalPolicyManager()
        self.state = "ACTIVE"

    def check_battery(self, level: float) -> str:
        """Legacy compatibility check returning string status."""
        telemetry = BatteryTelemetry(
            voltage_volts=3.3 + (level / 100.0) * 0.9,
            percentage=float(level),
            timestamp=0.0,
        )
        status = self.policy.update_battery_telemetry(telemetry)
        if status == BatteryStatus.CRITICAL:
            self.state = "CRITICAL"
        elif status == BatteryStatus.LOW:
            self.state = "LOW_POWER"
        else:
            self.state = "ACTIVE"
        return self.state
