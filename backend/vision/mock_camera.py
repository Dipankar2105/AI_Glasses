import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../firmware')))
from hal.camera import CameraFrame

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
        return CameraFrame(self.width, self.height, "GRAYSCALE", bytes([0]*(self.width*self.height)), t, s)
        
    def constant_frame(self, val):
        t, s = self._next_meta()
        return CameraFrame(self.width, self.height, "GRAYSCALE", bytes([val]*(self.width*self.height)), t, s)

    def gradient_frame(self):
        t, s = self._next_meta()
        arr = bytearray(self.width*self.height)
        for i in range(self.width*self.height):
            arr[i] = i % 256
        return CameraFrame(self.width, self.height, "GRAYSCALE", bytes(arr), t, s)
        
    def checkerboard_frame(self):
        t, s = self._next_meta()
        arr = bytearray(self.width*self.height)
        for y in range(self.height):
            for x in range(self.width):
                arr[y*self.width+x] = 255 if (x+y)%2==0 else 0
        return CameraFrame(self.width, self.height, "GRAYSCALE", bytes(arr), t, s)
        
    def rgb_frame(self):
        t, s = self._next_meta()
        return CameraFrame(self.width, self.height, "RGB", bytes([128]*(self.width*self.height*3)), t, s)

    def invalid_frame(self):
        t, s = self._next_meta()
        return CameraFrame(self.width, self.height, "YUV420", bytes([0]*(self.width*self.height)), t, s) # unsupported format
