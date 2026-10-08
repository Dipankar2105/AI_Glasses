import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hal.core import HALError
from hal.device_manager import HardwareDeviceManager
from hal.audio import AudioHAL, AudioCaptureFrame
from hal.imu import IMUHAL, IMUSample
from hal.touch import TouchHAL, TouchSample, TouchEvent
from hal.camera import CameraHAL, CameraFrame

def test_phase3_e2e():
    dm = HardwareDeviceManager()
    audio = AudioHAL()
    imu = IMUHAL()
    touch = TouchHAL()
    cam = CameraHAL()
    
    dm.register("audio", audio)
    dm.register("imu", imu)
    dm.register("touch", touch)
    dm.register("camera", cam)
    
    dm.initialize_all()
    dm.start_all()
    
    # Inject Mock
    audio._mock_capture_queue.append(AudioCaptureFrame([0.0]*160, 1, 1))
    imu._mock_queue.append(IMUSample(0,0,1,0,0,0,1,1))
    touch._mock_queue.append(TouchSample(TouchEvent.TAP, 1, 1))
    cam._mock_queue.append(CameraFrame(640,480,"RGB",b"",1,1))
    
    assert audio.read_capture_frame()[0] == HALError.OK
    assert imu.read_sample()[0] == HALError.OK
    assert touch.read_event()[0] == HALError.OK
    assert cam.capture_frame()[0] == HALError.OK
    
    dm.stop_all()
    dm.deinitialize_all()
    print("PHASE 3 E2E HOST TESTS PASSED.")

if __name__ == "__main__":
    test_phase3_e2e()
