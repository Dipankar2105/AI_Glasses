"""Typed contracts and data models for Power and Thermal Management.

All units:
- Voltage: Volts (V)
- Current: Milliamperes (mA)
- Power: Milliwatts (mW)
- Energy / Capacity: Milliampere-hours (mAh) / Milliwatt-hours (mWh)
- Temperature: Degrees Celsius (°C)
- Timestamp: POSIX seconds (float)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class OperatingState(str, Enum):
    """High-level system operating states."""
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    CAPTURE = "CAPTURE"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    LOW_POWER = "LOW_POWER"
    CRITICAL_SHUTDOWN = "CRITICAL_SHUTDOWN"


class BatteryStatus(str, Enum):
    """Battery charge and operational status."""
    NORMAL = "NORMAL"
    LOW = "LOW"
    CRITICAL = "CRITICAL"
    CHARGING = "CHARGING"
    FULL = "FULL"
    UNKNOWN = "UNKNOWN"
    STALE = "STALE"


class ThermalStatus(str, Enum):
    """Thermal condition states."""
    NORMAL = "NORMAL"
    WARM = "WARM"
    HOT_THROTTLED = "HOT_THROTTLED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"
    STALE = "STALE"


class PeripheralType(str, Enum):
    """Physical hardware peripherals."""
    ESP32_CORE = "ESP32_CORE"
    CAMERA = "CAMERA"
    MICROPHONE = "MICROPHONE"
    SPEAKER = "SPEAKER"
    IMU = "IMU"
    WIFI = "WIFI"
    STORAGE = "STORAGE"


class WorkloadPriority(str, Enum):
    """Execution priority levels for system workloads."""
    SAFETY_CRITICAL = "SAFETY_CRITICAL"
    TIME_SENSITIVE_AUDIO = "TIME_SENSITIVE_AUDIO"
    SENSOR_PROCESSING = "SENSOR_PROCESSING"
    VISION_CAPTURE = "VISION_CAPTURE"
    BACKGROUND_SYNC = "BACKGROUND_SYNC"


@dataclass(frozen=True)
class BatteryTelemetry:
    """Validated battery telemetry sample."""
    voltage_volts: float
    percentage: float
    is_charging: bool = False
    timestamp: float = 0.0
    is_valid: bool = True
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.is_valid:
            if not (0.0 <= self.voltage_volts <= 5.5):
                raise ValueError(f"Voltage {self.voltage_volts}V is outside physical range (0.0-5.5V)")
            if not (0.0 <= self.percentage <= 100.0):
                raise ValueError(f"Percentage {self.percentage}% must be within 0-100%")


@dataclass(frozen=True)
class ThermalTelemetry:
    """Validated thermal telemetry sample."""
    temperature_celsius: float
    sensor_id: str = "internal_die"
    timestamp: float = 0.0
    is_valid: bool = True
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.is_valid:
            if not (-40.0 <= self.temperature_celsius <= 150.0):
                raise ValueError(f"Temperature {self.temperature_celsius}°C outside physical sensor limits")


@dataclass(frozen=True)
class PeripheralPowerProfile:
    """Modeled estimated power characteristics for a peripheral."""
    peripheral: PeripheralType
    standby_current_ma: float
    active_current_ma: float
    voltage_volts: float = 3.3

    @property
    def standby_power_mw(self) -> float:
        return self.standby_current_ma * self.voltage_volts

    @property
    def active_power_mw(self) -> float:
        return self.active_current_ma * self.voltage_volts


@dataclass
class SystemPowerSnapshot:
    """Current consolidated state of system power, thermal, and peripherals."""
    requested_state: OperatingState
    confirmed_state: OperatingState
    battery_status: BatteryStatus
    thermal_status: ThermalStatus
    battery: BatteryTelemetry
    thermal: ThermalTelemetry
    total_estimated_power_mw: float
    throttle_factor: float = 1.0
    active_peripherals: Dict[PeripheralType, bool] = field(default_factory=dict)
    timestamp: float = 0.0
