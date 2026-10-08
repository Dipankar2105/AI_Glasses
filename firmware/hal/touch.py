from enum import Enum, auto
from typing import Optional
from .core import HALDevice, HALError, HALState

class TouchEvent(Enum):
    TOUCH_DOWN = auto()
    TOUCH_UP = auto()
    TAP = auto()
    DOUBLE_TAP = auto()
    LONG_PRESS = auto()

class TouchSample:
    def __init__(self, event: TouchEvent, timestamp: int, seq_num: int):
        self.event = event
        self.timestamp = timestamp
        self.seq_num = seq_num

class TouchHAL(HALDevice):
    def __init__(self):
        super().__init__()
        self._mock_queue = []
        
    def read_event(self) -> tuple[HALError, Optional[TouchSample]]:
        if self.state != HALState.RUNNING:
            return HALError.NOT_INITIALIZED, None
        if not self._mock_queue:
            return HALError.BUFFER_UNAVAILABLE, None
        return HALError.OK, self._mock_queue.pop(0)
