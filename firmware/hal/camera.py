from typing import Optional
from .core import HALDevice, HALError, HALState

class CameraFrame:
    def __init__(self, width: int, height: int, pixel_format: str, data: bytes, timestamp: int, seq_num: int):
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.data = data
        self.timestamp = timestamp
        self.seq_num = seq_num

class CameraHAL(HALDevice):
    def __init__(self):
        super().__init__()
        self._mock_queue = []
        
    def capture_frame(self) -> tuple[HALError, Optional[CameraFrame]]:
        if self.state != HALState.RUNNING:
            return HALError.NOT_INITIALIZED, None
        if not self._mock_queue:
            return HALError.BUFFER_UNAVAILABLE, None
        return HALError.OK, self._mock_queue.pop(0)
