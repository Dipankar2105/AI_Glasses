"""Power and energy modeling for NextSight smart glasses.

NOTE: All power values are modeled engineering estimates based on component datasheets
(ESP32-S3 Sense, OV3660, MAX98357A, MPU-6050, and 1S LiPo). They are NOT physical lab measurements.
"""

from typing import Dict
from backend.power.contracts import (
    OperatingState,
    PeripheralType,
    PeripheralPowerProfile,
)


# Modeled default component profiles at 3.3V supply rail
DEFAULT_PERIPHERAL_PROFILES: Dict[PeripheralType, PeripheralPowerProfile] = {
    PeripheralType.ESP32_CORE: PeripheralPowerProfile(
        peripheral=PeripheralType.ESP32_CORE,
        standby_current_ma=2.0,    # Light sleep
        active_current_ma=45.0,    # 240MHz dual-core compute
        voltage_volts=3.3,
    ),
    PeripheralType.WIFI: PeripheralPowerProfile(
        peripheral=PeripheralType.WIFI,
        standby_current_ma=0.0,    # Radio off/power-save
        active_current_ma=180.0,   # Tx/Rx burst average
        voltage_volts=3.3,
    ),
    PeripheralType.CAMERA: PeripheralPowerProfile(
        peripheral=PeripheralType.CAMERA,
        standby_current_ma=0.1,    # Sensor standby / clock gated
        active_current_ma=70.0,    # OV3660 active streaming
        voltage_volts=3.3,
    ),
    PeripheralType.MICROPHONE: PeripheralPowerProfile(
        peripheral=PeripheralType.MICROPHONE,
        standby_current_ma=0.01,
        active_current_ma=1.5,     # I2S MEMS mic active
        voltage_volts=3.3,
    ),
    PeripheralType.SPEAKER: PeripheralPowerProfile(
        peripheral=PeripheralType.SPEAKER,
        standby_current_ma=0.01,   # MAX98357A shutdown
        active_current_ma=120.0,   # Nominal speech playback level
        voltage_volts=3.3,
    ),
    PeripheralType.IMU: PeripheralPowerProfile(
        peripheral=PeripheralType.IMU,
        standby_current_ma=0.005,  # MPU-6050 sleep mode
        active_current_ma=3.8,     # 6-axis 100Hz active
        voltage_volts=3.3,
    ),
    PeripheralType.STORAGE: PeripheralPowerProfile(
        peripheral=PeripheralType.STORAGE,
        standby_current_ma=0.2,
        active_current_ma=45.0,    # MicroSD SPI/SDIO write
        voltage_volts=3.3,
    ),
}

# Modeled peripheral activation mapping per operating state
STATE_PERIPHERAL_ACTIVATION: Dict[OperatingState, Dict[PeripheralType, bool]] = {
    OperatingState.IDLE: {
        PeripheralType.ESP32_CORE: False,  # Standby / light sleep
        PeripheralType.WIFI: False,
        PeripheralType.CAMERA: False,
        PeripheralType.MICROPHONE: False,
        PeripheralType.SPEAKER: False,
        PeripheralType.IMU: True,          # Always tracking head gestures
        PeripheralType.STORAGE: False,
    },
    OperatingState.LISTENING: {
        PeripheralType.ESP32_CORE: True,
        PeripheralType.WIFI: True,
        PeripheralType.CAMERA: False,
        PeripheralType.MICROPHONE: True,
        PeripheralType.SPEAKER: False,
        PeripheralType.IMU: True,
        PeripheralType.STORAGE: False,
    },
    OperatingState.CAPTURE: {
        PeripheralType.ESP32_CORE: True,
        PeripheralType.WIFI: True,
        PeripheralType.CAMERA: True,
        PeripheralType.MICROPHONE: False,
        PeripheralType.SPEAKER: False,
        PeripheralType.IMU: True,
        PeripheralType.STORAGE: False,
    },
    OperatingState.PROCESSING: {
        PeripheralType.ESP32_CORE: True,
        PeripheralType.WIFI: True,
        PeripheralType.CAMERA: False,
        PeripheralType.MICROPHONE: False,
        PeripheralType.SPEAKER: False,
        PeripheralType.IMU: True,
        PeripheralType.STORAGE: False,
    },
    OperatingState.SPEAKING: {
        PeripheralType.ESP32_CORE: True,
        PeripheralType.WIFI: True,
        PeripheralType.CAMERA: False,
        PeripheralType.MICROPHONE: False,
        PeripheralType.SPEAKER: True,
        PeripheralType.IMU: True,
        PeripheralType.STORAGE: False,
    },
    OperatingState.LOW_POWER: {
        PeripheralType.ESP32_CORE: False,  # Deep / light sleep duty cycle
        PeripheralType.WIFI: False,
        PeripheralType.CAMERA: False,
        PeripheralType.MICROPHONE: False,
        PeripheralType.SPEAKER: False,
        PeripheralType.IMU: True,          # Low-rate IMU
        PeripheralType.STORAGE: False,
    },
    OperatingState.CRITICAL_SHUTDOWN: {
        PeripheralType.ESP32_CORE: False,
        PeripheralType.WIFI: False,
        PeripheralType.CAMERA: False,
        PeripheralType.MICROPHONE: False,
        PeripheralType.SPEAKER: False,
        PeripheralType.IMU: False,
        PeripheralType.STORAGE: False,
    },
}


class PowerModel:
    """Calculates modeled power consumption and runtime estimates."""

    def __init__(
        self,
        profiles: Dict[PeripheralType, PeripheralPowerProfile] = None,
        battery_capacity_mah: float = 500.0,
        nominal_voltage_volts: float = 3.7,
    ) -> None:
        self.profiles = profiles or DEFAULT_PERIPHERAL_PROFILES
        self.battery_capacity_mah = battery_capacity_mah
        self.nominal_voltage_volts = nominal_voltage_volts

    @property
    def total_energy_capacity_mwh(self) -> float:
        """Total theoretical battery energy capacity in mWh."""
        return self.battery_capacity_mah * self.nominal_voltage_volts

    def estimate_state_power_mw(self, state: OperatingState) -> float:
        """Estimate total power consumption (mW) for a given operating state."""
        activation = STATE_PERIPHERAL_ACTIVATION.get(state, {})
        total_mw = 0.0

        for peripheral, profile in self.profiles.items():
            is_active = activation.get(peripheral, False)
            if is_active:
                total_mw += profile.active_power_mw
            else:
                total_mw += profile.standby_power_mw

        return round(total_mw, 2)

    def estimate_runtime_hours(self, state: OperatingState, remaining_percentage: float) -> float:
        """Estimate remaining operating time (hours) in a continuous state.

        Modeled estimate assuming linear discharge and constant power draw.
        """
        if remaining_percentage <= 0.0:
            return 0.0

        power_mw = self.estimate_state_power_mw(state)
        if power_mw <= 0.0:
            return float("inf")

        usable_energy_mwh = (self.total_energy_capacity_mwh * (remaining_percentage / 100.0)) * 0.9  # 90% efficiency
        return round(usable_energy_mwh / power_mw, 2)
