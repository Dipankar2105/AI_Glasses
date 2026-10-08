import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from hal.core import HALState, HALError
from hal.audio import AudioHAL, AudioCaptureFrame, AudioPlaybackFrame

def test_hal_lifecycle():
    audio = AudioHAL()
    assert audio.state == HALState.UNINITIALIZED
    
    # Invalid transition (start before init)
    err = audio.start()
    assert err == HALError.NOT_INITIALIZED
    
    # Valid Init
    err = audio.init()
    assert err == HALError.OK
    assert audio.state == HALState.INITIALIZED
    
    # Invalid transition (deinit before stop/while running - though currently it's initialized, so it's valid)
    # Let's start first
    err = audio.start()
    assert err == HALError.OK
    assert audio.state == HALState.RUNNING
    
    # Invalid deinit while running
    err = audio.deinit()
    assert err == HALError.INVALID_STATE_TRANSITION
    
    # Valid stop
    err = audio.stop()
    assert err == HALError.OK
    assert audio.state == HALState.STOPPED
    
    # Valid deinit
    err = audio.deinit()
    assert err == HALError.OK
    assert audio.state == HALState.DEINITIALIZED

def test_audio_buffer_contract():
    audio = AudioHAL()
    audio.init()
    audio.start()
    
    # 1. Sample Representation & Format Metadata test
    # DSP internal uses floats. Hardware uses int16. Python mock accepts floats representing the final conversion.
    play_frame = AudioPlaybackFrame(pcm_data=[0.5, -0.5], timestamp=1000, seq_num=1)
    assert isinstance(play_frame.pcm_data[0], float), "HAL to DSP boundary expects floats"
    assert hasattr(play_frame, 'timestamp'), "Frame must contain timestamp metadata"
    assert hasattr(play_frame, 'seq_num'), "Frame must contain sequence metadata"
    
    # Write playback frame
    err = audio.write_playback_frame(play_frame)
    assert err == HALError.OK
    
    # 2. Reference Frame Sequence/Timestamp Association
    err, ref_frame = audio.read_reference_frame()
    assert err == HALError.OK
    assert ref_frame is not None
    assert ref_frame.seq_num == 1, "AEC reference association via sequence numbers"
    assert ref_frame.pcm_data == [0.5, -0.5]
    
    # Read reference frame again (should be empty now - tests Ownership/Queue behavior)
    err, ref_frame = audio.read_reference_frame()
    assert err == HALError.BUFFER_UNAVAILABLE
    assert ref_frame is None
    
    # 3. Capture/Reference Separation
    err, cap_frame = audio.read_capture_frame()
    assert err == HALError.BUFFER_UNAVAILABLE, "Capture and playback queues are entirely independent"
    
    # Mock some captured data
    audio._mock_capture_queue.append(AudioCaptureFrame([0.1, 0.2], 1005, 1))
    err, cap_frame = audio.read_capture_frame()
    assert err == HALError.OK
    assert cap_frame.pcm_data == [0.1, 0.2]

if __name__ == "__main__":
    print("Running SOFTWARE HAL CONTRACT TESTS...")
    test_hal_lifecycle()
    test_audio_buffer_contract()
    print("HAL Tests Passed.")
