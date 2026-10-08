from typing import List, Optional
from .core import HALDevice, HALError, HALState

class IMUSample:
    def __init__(self, accel_x: float, accel_y: float, accel_z: float, gyro_x: float, gyro_y: float, gyro_z: float, timestamp: int, seq_num: int):
        self.accel_x = accel_x
        self.accel_y = accel_y
        self.accel_z = accel_z
        self.gyro_x = gyro_x
        self.gyro_y = gyro_y
        self.gyro_z = gyro_z
        self.timestamp = timestamp
        self.seq_num = seq_num

class IMUHAL(HALDevice):
    def __init__(self):
        super().__init__()
        self._mock_queue = []
        
    def read_sample(self) -> tuple[HALError, Optional[IMUSample]]:
        if self.state != HALState.RUNNING:
            return HALError.NOT_INITIALIZED, None
        if not self._mock_queue:
            return HALError.BUFFER_UNAVAILABLE, None
        return HALError.OK, self._mock_queue.pop(0)
