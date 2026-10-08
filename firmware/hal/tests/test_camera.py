import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hal.core import HALError
from hal.camera import CameraHAL, CameraFrame

def test_camera():
    cam = CameraHAL()
    cam.init()
    cam.start()
    
    cam._mock_queue.append(CameraFrame(640, 480, "RGB565", b"123", 1000, 1))
    err, frame = cam.capture_frame()
    assert err == HALError.OK
    assert frame.width == 640

if __name__ == "__main__":
    test_camera()
