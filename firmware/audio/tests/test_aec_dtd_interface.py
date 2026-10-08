import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dsp.buffer import AudioBuffer
from dsp.aec_dtd import NLMSAECWithDTD

def test_interface():
    # Instantiate with known parameters
    aec = NLMSAECWithDTD(filter_length=256, step_size=0.5, regularization=1e6, enabled=True, dtd_threshold=0.5, dtd_hold=800)
    
    # 1. Valid Frame Processing (equal lengths)
    mic_data = [0.05] * 128  # Keep mic below DTD threshold (0.5 * max(ref)=0.1) so adaptation occurs
    ref_data = [0.2] * 128
    out_buf = aec.process_aec(AudioBuffer(mic_data), AudioBuffer(ref_data))
    assert len(out_buf.data) == 128, "Output frame should match input frame size"
    
    # 2. Repeated Frame Processing & Persistent State
    # Verify that processing identical frames results in different outputs due to persistent filter state
    out_buf2 = aec.process_aec(AudioBuffer(mic_data), AudioBuffer(ref_data))
    assert len(out_buf2.data) == 128
    assert out_buf.data != out_buf2.data, "Filter state must persist and evolve between frames"
    
    # 3. Mismatched Frame Lengths (Implementation defined behavior)
    # The Python implementation uses min(len(mic), len(ref))
    mic_long = [0.1] * 256
    ref_short = [0.2] * 128
    out_buf3 = aec.process_aec(AudioBuffer(mic_long), AudioBuffer(ref_short))
    assert len(out_buf3.data) == 128, "Should process up to the shortest buffer length"
    
    # 4. Zero Reference / Zero Mic (Requires fresh state since no reset exists)
    aec_clean = NLMSAECWithDTD(filter_length=256, step_size=0.5, regularization=1e6, enabled=True, dtd_threshold=0.5, dtd_hold=800)
    mic_zero = [0.0] * 128
    ref_zero = [0.0] * 128
    out_buf4 = aec_clean.process_aec(AudioBuffer(mic_zero), AudioBuffer(ref_zero))
    assert all(x == 0.0 for x in out_buf4.data), "Zero input should produce zero output on clean state"
    
    # 5. Invalid numeric values handling (NaN/Inf)
    # Python standard math might raise exception or propagate NaN.
    # The requirement is to verify behavior if defined, or note it if undefined.
    # We will pass NaN and observe propagation.
    mic_nan = [float('nan')] * 128
    ref_nan = [float('nan')] * 128
    out_buf5 = aec.process_aec(AudioBuffer(mic_nan), AudioBuffer(ref_nan))
    assert math.isnan(out_buf5.data[0]), "NaN inputs propagate to output"
    
    # Reset Behavior is currently undefined (no reset method exists).
    # It must be documented as a hardware/future requirement.
    
    print("AEC+DTD Interface Test: PASS")

if __name__ == "__main__":
    test_interface()
