import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hal.core import HALError
from hal.device_manager import HardwareDeviceManager
from hal.audio import AudioHAL
from hal.imu import IMUHAL
from hal.touch import TouchHAL
from hal.camera import CameraHAL

def test_integration():
    dm = HardwareDeviceManager()
    dm.register("audio", AudioHAL())
    dm.register("imu", IMUHAL())
    dm.register("touch", TouchHAL())
    dm.register("camera", CameraHAL())
    
    assert dm.initialize_all() == HALError.OK
    assert dm.start_all() == HALError.OK
    assert dm.stop_all() == HALError.OK
    assert dm.deinitialize_all() == HALError.OK

if __name__ == "__main__":
    test_integration()
