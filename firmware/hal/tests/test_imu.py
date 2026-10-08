import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hal.core import HALError
from hal.imu import IMUHAL, IMUSample

def test_imu():
    imu = IMUHAL()
    imu.init()
    imu.start()
    
    err, samp = imu.read_sample()
    assert err == HALError.BUFFER_UNAVAILABLE
    
    imu._mock_queue.append(IMUSample(0, 0, 1.0, 0, 0, 0, 1000, 1))
    err, samp = imu.read_sample()
    assert err == HALError.OK
    assert samp.seq_num == 1
    assert samp.accel_z == 1.0

if __name__ == "__main__":
    test_imu()
