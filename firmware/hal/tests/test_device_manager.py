import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hal.core import HALDevice, HALError
from hal.device_manager import HardwareDeviceManager, DeviceManagerState

class MockDevice(HALDevice):
    pass

class MockFailingDevice(HALDevice):
    def init(self): return HALError.HARDWARE_ERROR

def test_device_manager():
    dm = HardwareDeviceManager()
    d1 = MockDevice()
    d2 = MockDevice()
    
    # Register
    assert dm.register("d1", d1) == HALError.OK
    assert dm.register("d1", d1) == HALError.INVALID_CONFIGURATION # duplicate
    assert dm.register("d2", d2, deps=["d1"]) == HALError.OK
    
    # Init
    assert dm.initialize_all() == HALError.OK
    assert dm.state == DeviceManagerState.INITIALIZED
    
    # Start
    assert dm.start_all() == HALError.OK
    assert dm.state == DeviceManagerState.RUNNING
    
    # Invalid transition
    assert dm.initialize_all() == HALError.INVALID_STATE_TRANSITION
    
    # Stop
    assert dm.stop_all() == HALError.OK
    
    # Deinit
    assert dm.deinitialize_all() == HALError.OK

def test_failing_device():
    dm = HardwareDeviceManager()
    dm.register("fail", MockFailingDevice())
    assert dm.initialize_all() == HALError.HARDWARE_ERROR
    assert dm.state == DeviceManagerState.FAILED

if __name__ == "__main__":
    test_device_manager()
    test_failing_device()
