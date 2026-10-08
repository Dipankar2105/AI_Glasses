import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hal.core import HALError
from hal.touch import TouchHAL, TouchSample, TouchEvent

def test_touch():
    touch = TouchHAL()
    touch.init()
    touch.start()
    
    touch._mock_queue.append(TouchSample(TouchEvent.TAP, 1000, 1))
    err, samp = touch.read_event()
    assert err == HALError.OK
    assert samp.event == TouchEvent.TAP

if __name__ == "__main__":
    test_touch()
