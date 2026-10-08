from enum import Enum, auto

class HALState(Enum):
    UNINITIALIZED = auto()
    INITIALIZED = auto()
    RUNNING = auto()
    STOPPED = auto()
    DEINITIALIZED = auto()

class HALError(Enum):
    OK = auto()
    NOT_INITIALIZED = auto()
    DEVICE_UNAVAILABLE = auto()
    INVALID_CONFIGURATION = auto()
    BUFFER_UNAVAILABLE = auto()
    OVERRUN = auto()
    UNDERRUN = auto()
    TIMEOUT = auto()
    HARDWARE_ERROR = auto()
    INVALID_STATE_TRANSITION = auto()

class HALDevice:
    """Base class for all Hardware Abstraction Layer devices."""
    def __init__(self):
        self.state = HALState.UNINITIALIZED

    def init(self) -> HALError:
        if self.state not in (HALState.UNINITIALIZED, HALState.DEINITIALIZED):
            return HALError.INVALID_STATE_TRANSITION
        self.state = HALState.INITIALIZED
        return HALError.OK

    def start(self) -> HALError:
        if self.state == HALState.RUNNING:
            return HALError.OK # Idempotent start
        if self.state not in (HALState.INITIALIZED, HALState.STOPPED):
            return HALError.NOT_INITIALIZED
        self.state = HALState.RUNNING
        return HALError.OK

    def stop(self) -> HALError:
        if self.state == HALState.UNINITIALIZED or self.state == HALState.DEINITIALIZED:
            return HALError.NOT_INITIALIZED
        self.state = HALState.STOPPED
        return HALError.OK

    def deinit(self) -> HALError:
        if self.state == HALState.RUNNING:
            return HALError.INVALID_STATE_TRANSITION
        self.state = HALState.DEINITIALIZED
        return HALError.OK
