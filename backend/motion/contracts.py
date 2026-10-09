from enum import Enum
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator

class SensorStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"
    DEGRADED = "DEGRADED"

class HeadGestureType(str, Enum):
    NONE = "NONE"
    STATIONARY = "STATIONARY"
    NOD = "NOD"
    SHAKE = "SHAKE"
    TILT_HEAD = "TILT_HEAD"

class TouchGestureType(str, Enum):
    NONE = "NONE"
    TAP = "TAP"
    DOUBLE_TAP = "DOUBLE_TAP"
    LONG_PRESS = "LONG_PRESS"

class InteractionIntent(str, Enum):
    NONE = "NONE"
    CONFIRM = "CONFIRM"
    DISMISS_OR_CANCEL = "DISMISS_OR_CANCEL"
    TRIGGER_SCENE_ANALYSIS = "TRIGGER_SCENE_ANALYSIS"
    START_LISTENING = "START_LISTENING"
    TOGGLE_MUTE = "TOGGLE_MUTE"


class IMUSample(BaseModel):
    """
    Standard 6-DOF IMU reading from MPU-6050.
    Coordinate Frame:
      X: Right temple (+X)
      Y: Forward looking (+Y, pitch up/down around Y)
      Z: Upward (+Z, yaw left/right around Z)
    Units:
      Accelerometer: g (1g = ~9.80665 m/s^2)
      Gyroscope: degrees per second (deg/s)
    """
    timestamp: float = Field(..., description="Sample timestamp in seconds (monotonic)")
    ax: float = Field(..., ge=-16.0, le=16.0, description="Accel X axis in g")
    ay: float = Field(..., ge=-16.0, le=16.0, description="Accel Y axis in g")
    az: float = Field(..., ge=-16.0, le=16.0, description="Accel Z axis in g")
    gx: float = Field(..., ge=-2000.0, le=2000.0, description="Gyro X (roll rate) in deg/s")
    gy: float = Field(..., ge=-2000.0, le=2000.0, description="Gyro Y (pitch rate) in deg/s")
    gz: float = Field(..., ge=-2000.0, le=2000.0, description="Gyro Z (yaw rate) in deg/s")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Timestamp must be non-negative")
        return v


class TouchSample(BaseModel):
    """Capacitive touch sensor sample."""
    timestamp: float = Field(..., ge=0.0, description="Sample timestamp in seconds")
    is_pressed: bool = Field(..., description="True if capacitive electrode is pressed")


class GestureEvent(BaseModel):
    """Detected head or touch interaction event."""
    gesture_type: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InteractionResult(BaseModel):
    """Result of an interaction dispatch action."""
    intent: InteractionIntent
    triggered_by: str # e.g. "HEAD_GESTURE:NOD", "TOUCH:DOUBLE_TAP"
    success: bool
    action_taken: str
    target_service: str
    error: Optional[str] = None
    timestamp: float
