"""System runtime lifecycle manager for smart glasses firmware."""

from enum import Enum
from typing import Dict, Optional


class SystemState(str, Enum):
    BOOT = "BOOT"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    STREAMING = "STREAMING"
    DEGRADED = "DEGRADED"
    SHUTDOWN = "SHUTDOWN"


class SystemRuntime:
    """Manages boot lifecycle, peripheral readiness, and event routing."""

    def __init__(self):
        self.state = SystemState.BOOT.value
        self.peripherals: Dict[str, bool] = {
            "camera": False,
            "audio_in": False,
            "audio_out": False,
            "imu": False,
            "touch": False,
            "wifi": False,
        }
        self.error_log = []

    def boot(self) -> str:
        """Execute boot sequence, initializing default peripherals into READY state."""
        self.state = SystemState.INITIALIZING.value
        # In host simulation, mark core subsystems as initialized
        for k in self.peripherals:
            self.peripherals[k] = True
        self.state = SystemState.READY.value
        return self.state

    def process_event(self, event: str) -> str:
        """Handle incoming system events and lifecycle transitions."""
        if event == "SHUTDOWN":
            self.state = SystemState.SHUTDOWN.value
            for k in self.peripherals:
                self.peripherals[k] = False
        elif event == "START_STREAM":
            if self.state in (SystemState.READY.value, SystemState.DEGRADED.value):
                self.state = SystemState.STREAMING.value
        elif event == "STOP_STREAM":
            if self.state == SystemState.STREAMING.value:
                self.state = SystemState.READY.value
        elif event == "PERIPHERAL_ERROR":
            self.state = SystemState.DEGRADED.value
        return self.state
