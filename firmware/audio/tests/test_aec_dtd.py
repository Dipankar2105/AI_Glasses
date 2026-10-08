import math
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dsp.buffer import AudioBuffer
from dsp.aec import NLMSAEC
from dsp.aec_dtd import NLMSAECWithDTD
from tests.test_vad import generate_noise, generate_speech_like, SAMPLE_RATE

def calculate_rms(signal):
    if not signal: return 0.0
    return math.sqrt(sum(x*x for x in signal) / len(signal))

def calculate_er(before, after):
    rms_b = calculate_rms(before)
    rms_a = calculate_rms(after)
    if rms_a == 0: return 999.0
    if rms_b == 0: return 0.0
    return 20 * math.log10(rms_b / rms_a)

def calculate_nmse(ref, test):
    if len(ref) != len(test): return 999.0
    num = sum((r - t)**2 for r, t in zip(ref, test))
    den = sum(r**2 for r in ref)
    if den == 0: return 0.0
    return num / den

def delayed_signal(signal, delay_samples, attenuation):
    out = [0.0] * len(signal)
    for i in range(delay_samples, len(signal)):
        out[i] = signal[i - delay_samples] * attenuation
    return out

def run_compare_test(name, mic_signal, ref_signal, expected_near_end=None, echo_signal=None, expected_dt_ratio=0.0):
    aec_baseline = NLMSAEC(step_size=0.5, regularization=1e6)
    aec_dtd = NLMSAECWithDTD(step_size=0.5, regularization=1e6, dtd_threshold=0.5)
    
    out_base = aec_baseline.process_aec(AudioBuffer(mic_signal), AudioBuffer(ref_signal)).data
    out_dtd = aec_dtd.process_aec(AudioBuffer(mic_signal), AudioBuffer(ref_signal)).data
    
    dt_frames = aec_dtd.dt_detected_frames
    dt_ratio = dt_frames / len(mic_signal) if mic_signal else 0.0
    
    er_base = 0.0
    er_dtd = 0.0
    if echo_signal and (expected_near_end is None or all(x == 0 for x in expected_near_end)):
        skip = int(SAMPLE_RATE * 0.2)
        er_base = calculate_er(echo_signal[skip:], out_base[skip:])
        er_dtd = calculate_er(echo_signal[skip:], out_dtd[skip:])
        
    nmse_base = 0.0
    nmse_dtd = 0.0
    if expected_near_end and not all(x == 0 for x in expected_near_end):
        skip = int(SAMPLE_RATE * 0.2)
        nmse_base = calculate_nmse(expected_near_end[skip:], out_base[skip:])
        nmse_dtd = calculate_nmse(expected_near_end[skip:], out_dtd[skip:])
        
    return {
        "name": name,
        "baseline": {
            "er_db": er_base,
            "nmse": nmse_base,
            "rms_out": calculate_rms(out_base),
            "clipping": any(abs(x) > 32767 for x in out_base),
            "stable": not any(math.isnan(x) for x in out_base)
        },
        "with_dtd": {
            "er_db": er_dtd,
            "nmse": nmse_dtd,
            "rms_out": calculate_rms(out_dtd),
            "dt_detected_ratio": dt_ratio,
            "clipping": any(abs(x) > 32767 for x in out_dtd),
            "stable": not any(math.isnan(x) for x in out_dtd)
        }
    }

def test_aec_dtd():
    results = {}
    
    ref = generate_speech_like(1.0)
    
    # TEST 1 - Echo Only
    echo_1 = delayed_signal(ref, 20, 0.4)
    results["TEST_1_ECHO_ONLY"] = run_compare_test("Echo Only", echo_1, ref, echo_signal=echo_1, expected_dt_ratio=0.0)
    
    # TEST 2 - Near-End Only
    near_2 = generate_speech_like(1.0)
    results["TEST_2_NEAR_ONLY"] = run_compare_test("Near-End Only", near_2, [0]*len(ref), expected_near_end=near_2, expected_dt_ratio=1.0)
    
    # TEST 3 - Clean Double-Talk
    near_3 = generate_speech_like(1.0)[100:] + [0]*100
    echo_3 = delayed_signal(ref, 20, 0.4)
    mic_3 = [n + e for n, e in zip(near_3, echo_3)]
    results["TEST_3_DOUBLE_TALK"] = run_compare_test("Double-Talk", mic_3, ref, expected_near_end=near_3, expected_dt_ratio=1.0)
    
    # TEST 4 - Strong Near-End
    near_4 = [n * 2.0 for n in near_3]
    mic_4 = [n + e for n, e in zip(near_4, echo_3)]
    results["TEST_4_STRONG_NEAR"] = run_compare_test("Strong Near-End DT", mic_4, ref, expected_near_end=near_4)
    
    # TEST 5 - Weak Near-End
    near_5 = [n * 0.1 for n in near_3]
    mic_5 = [n + e for n, e in zip(near_5, echo_3)]
    results["TEST_5_WEAK_NEAR"] = run_compare_test("Weak Near-End DT", mic_5, ref, expected_near_end=near_5)
    
    # TEST 6 - DT Begins
    ref_6 = generate_speech_like(1.0)
    near_6 = [0]*int(SAMPLE_RATE*0.5) + generate_speech_like(0.5)
    echo_6 = delayed_signal(ref_6, 20, 0.4)
    mic_6 = [n + e for n, e in zip(near_6, echo_6)]
    results["TEST_6_DT_BEGINS"] = run_compare_test("DT Begins", mic_6, ref_6, expected_near_end=near_6)
    
    # TEST 8 - Silence
    silence = [0.0]*len(ref)
    results["TEST_8_SILENCE"] = run_compare_test("Silence", silence, silence, expected_near_end=silence)
    
    ac = {
        "dtd_improves_dt_nmse": results["TEST_3_DOUBLE_TALK"]["with_dtd"]["nmse"] < results["TEST_3_DOUBLE_TALK"]["baseline"]["nmse"],
        "echo_only_cancels_normally": results["TEST_1_ECHO_ONLY"]["with_dtd"]["er_db"] > 10.0,
        "dt_detected_properly": results["TEST_3_DOUBLE_TALK"]["with_dtd"]["dt_detected_ratio"] > 0.5,
        "silence_stable": results["TEST_8_SILENCE"]["with_dtd"]["stable"]
    }
    
    overall = "CONDITIONAL" if all(ac.values()) else "FAIL"
    
    out = {
        "phase": "2B-4A",
        "component": "AEC_DTD",
        "method": "Geigel DTD + NLMS",
        "tests": results,
        "acceptance_criteria": ac,
        "overall_status": overall
    }
    
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tests/results')), exist_ok=True)
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tests/results/audio-aec-dtd-phase2b4a.json')), 'w') as f:
        json.dump(out, f, indent=4)
        
    print(f"AEC DTD Status: {overall}")

if __name__ == "__main__":
    test_aec_dtd()
