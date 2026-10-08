import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../firmware')))
from hal.camera import CameraFrame
import numpy as np

class DeterministicMockCamera:
    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height
        self.seq = 0
        self.time = 0

    def _next_meta(self):
        self.seq += 1
        self.time += 100
        return self.time, self.seq

    def blank_frame(self):
        t, s = self._next_meta()
        data = np.zeros((self.height, self.width), dtype=np.uint8)
        return CameraFrame(self.width, self.height, "GRAYSCALE", data.tobytes(), t, s)
        
    def constant_frame(self, val):
        t, s = self._next_meta()
        data = np.full((self.height, self.width), val, dtype=np.uint8)
        return CameraFrame(self.width, self.height, "GRAYSCALE", data.tobytes(), t, s)

    def gradient_frame(self):
        t, s = self._next_meta()
        x = np.linspace(0, 255, self.width, dtype=np.uint8)
        data = np.tile(x, (self.height, 1))
        return CameraFrame(self.width, self.height, "GRAYSCALE", data.tobytes(), t, s)
        
    def checkerboard_frame(self):
        t, s = self._next_meta()
        data = np.zeros((self.height, self.width), dtype=np.uint8)
        data[1::2, ::2] = 255
        data[::2, 1::2] = 255
        return CameraFrame(self.width, self.height, "GRAYSCALE", data.tobytes(), t, s)
        
    def rgb_frame(self):
        t, s = self._next_meta()
        data = np.full((self.height, self.width, 3), [128, 64, 32], dtype=np.uint8)
        return CameraFrame(self.width, self.height, "RGB", data.tobytes(), t, s)

    def invalid_frame(self):
        t, s = self._next_meta()
        # Malformed dimension data
        return CameraFrame(self.width, self.height, "YUV420", bytes([0]*1), t, s)
