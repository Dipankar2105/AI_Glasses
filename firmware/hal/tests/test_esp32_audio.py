import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hal.core import HALState, HALError
from hal.audio import AudioCaptureFrame, AudioPlaybackFrame

# Mocking the ESP32 C Adapter behavior for host testing

class ESP32AudioAdapterHostMock:
    def __init__(self):
        self.state = HALState.UNINITIALIZED
        self.capture_seq = 0
        self.playback_seq = 0
        
    def init(self):
        if self.state not in (HALState.UNINITIALIZED, HALState.DEINITIALIZED):
            return HALError.INVALID_STATE_TRANSITION
        self.capture_seq = 0
        self.playback_seq = 0
        self.state = HALState.INITIALIZED
        return HALError.OK

    def start(self):
        if self.state == HALState.RUNNING: return HALError.OK
        if self.state not in (HALState.INITIALIZED, HALState.STOPPED):
            return HALError.NOT_INITIALIZED
        self.state = HALState.RUNNING
        return HALError.OK
        
    def stop(self):
        if self.state in (HALState.UNINITIALIZED, HALState.DEINITIALIZED):
            return HALError.NOT_INITIALIZED
        self.state = HALState.STOPPED
        return HALError.OK

    def int16_to_float(self, sample: int) -> float:
        return float(sample)
        
    def float_to_int16(self, sample: float) -> int:
        if sample > 32767.0: return 32767
        if sample < -32768.0: return -32768
        return int(sample)
        
    def simulate_hardware_rx(self, hw_int16_buffer: list) -> tuple[HALError, AudioCaptureFrame]:
        if self.state != HALState.RUNNING:
            return HALError.NOT_INITIALIZED, None
        
        # Convert to float for DSP
        float_buf = [self.int16_to_float(s) for s in hw_int16_buffer]
        frame = AudioCaptureFrame(float_buf, 1000, self.capture_seq)
        self.capture_seq += 1
        return HALError.OK, frame
        
    def simulate_hardware_tx(self, dsp_float_frame: AudioPlaybackFrame) -> tuple[HALError, list]:
        if self.state != HALState.RUNNING:
            return HALError.NOT_INITIALIZED, None
            
        # Convert float to int16 for Hardware
        hw_buf = [self.float_to_int16(s) for s in dsp_float_frame.pcm_data]
        self.playback_seq = dsp_float_frame.seq_num
        return HALError.OK, hw_buf

def test_esp32_adapter_host():
    adapter = ESP32AudioAdapterHostMock()
    
    # 1. Lifecycle
    assert adapter.simulate_hardware_rx([])[0] == HALError.NOT_INITIALIZED
    assert adapter.init() == HALError.OK
    assert adapter.start() == HALError.OK
    
    # 2. int16 -> float conversion & Sequence increment
    err, cap_frame = adapter.simulate_hardware_rx([0, 32767, -32768])
    assert err == HALError.OK
    assert cap_frame.seq_num == 0
    assert cap_frame.pcm_data == [0.0, 32767.0, -32768.0]
    
    err, cap_frame2 = adapter.simulate_hardware_rx([100])
    assert cap_frame2.seq_num == 1
    
    # 3. float -> int16 conversion & Saturation/Clipping
    play_frame = AudioPlaybackFrame([0.0, 40000.0, -40000.0, 15000.5], 2000, 42)
    err, hw_buf = adapter.simulate_hardware_tx(play_frame)
    assert err == HALError.OK
    assert adapter.playback_seq == 42
    assert hw_buf == [0, 32767, -32768, 15000]

if __name__ == "__main__":
    print("Running SOFTWARE / HOST VALIDATION for ESP32 Adapter...")
    test_esp32_adapter_host()
    print("ESP32 Adapter Host Tests Passed.")
