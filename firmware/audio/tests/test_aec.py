import math
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dsp.buffer import AudioBuffer
from dsp.aec import NLMSAEC
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

def run_aec_test(name, mic_signal, ref_signal, aec, expected_near_end=None, echo_signal=None):
    out_buf = aec.process_aec(AudioBuffer(mic_signal), AudioBuffer(ref_signal))
    out = out_buf.data
    
    # Calculate Echo Reduction if echo is known
    er = 0.0
    if echo_signal:
        # In a real test, echo is mixed into mic. The output should be just near_end.
        # But if expected_near_end is 0, then output IS the residual echo.
        # So ER = 20 log10( RMS(echo) / RMS(out) )
        if expected_near_end is None or all(x == 0 for x in expected_near_end):
            # Skip convergence period (first 0.2s)
            skip = int(SAMPLE_RATE * 0.2)
            er = calculate_er(echo_signal[skip:], out[skip:])
            
    # Calculate near-end preservation if expected near end is present
    nmse = 0.0
    if expected_near_end and not all(x == 0 for x in expected_near_end):
        skip = int(SAMPLE_RATE * 0.2)
        nmse = calculate_nmse(expected_near_end[skip:], out[skip:])
        
    # Check numerical stability
    stable = not any(math.isnan(x) or math.isinf(x) for x in out)
    clipping = any(abs(x) > 32767 for x in out)
    
    return {
        "name": name,
        "echo_reduction_db": er,
        "near_end_nmse": nmse,
        "stable": stable,
        "clipping": clipping,
        "rms_out": calculate_rms(out)
    }

def test_aec():
    results = {}
    
    # Generate 1 second reference
    ref_signal = generate_speech_like(1.0)
    
    # TEST A: No Echo (Mic = near end only)
    near_end_a = generate_noise(1000.0, 1.0) # Using noise as near end for variation
    aec_a = NLMSAEC(step_size=0.1)
    results["A_NO_ECHO"] = run_aec_test("No Echo", near_end_a, ref_signal, aec_a, expected_near_end=near_end_a)
    
    # TEST B: Pure Echo
    echo_b = delayed_signal(ref_signal, delay_samples=50, attenuation=0.5)
    mic_b = echo_b
    aec_b = NLMSAEC(step_size=0.5)
    results["B_PURE_ECHO"] = run_aec_test("Pure Echo", mic_b, ref_signal, aec_b, echo_signal=echo_b)
    
    # TEST C: Near-End + Echo
    near_end_c = generate_noise(1000.0, 1.0)
    echo_c = delayed_signal(ref_signal, delay_samples=30, attenuation=0.3)
    mic_c = [n + e for n, e in zip(near_end_c, echo_c)]
    aec_c = NLMSAEC(step_size=0.1) # Slower step size to preserve near end
    results["C_NEAR_END_ECHO"] = run_aec_test("Near-End + Echo", mic_c, ref_signal, aec_c, expected_near_end=near_end_c, echo_signal=echo_c)
    
    # TEST D: Near-End + Echo + Noise
    near_end_d = generate_noise(1000.0, 1.0)
    echo_d = delayed_signal(ref_signal, delay_samples=40, attenuation=0.4)
    background_noise = generate_noise(200.0, 1.0)
    mic_d = [n + e + b for n, e, b in zip(near_end_d, echo_d, background_noise)]
    expected_d = [n + b for n, b in zip(near_end_d, background_noise)]
    aec_d = NLMSAEC(step_size=0.1)
    results["D_NEAR_END_ECHO_NOISE"] = run_aec_test("Near-End + Echo + Noise", mic_d, ref_signal, aec_d, expected_near_end=expected_d, echo_signal=echo_d)
    
    # TEST E: Different Delays
    echo_e1 = delayed_signal(ref_signal, delay_samples=10, attenuation=0.5)
    echo_e2 = delayed_signal(ref_signal, delay_samples=150, attenuation=0.5)
    results["E_DELAY_SHORT"] = run_aec_test("Short Delay", echo_e1, ref_signal, NLMSAEC(step_size=0.5), echo_signal=echo_e1)
    results["E_DELAY_LONG"] = run_aec_test("Long Delay", echo_e2, ref_signal, NLMSAEC(step_size=0.5), echo_signal=echo_e2)
    
    # TEST F: Gain Variation
    echo_f = delayed_signal(ref_signal, delay_samples=50, attenuation=1.5) # Gain > 1
    results["F_HIGH_GAIN"] = run_aec_test("High Gain", echo_f, ref_signal, NLMSAEC(step_size=0.5), echo_signal=echo_f)
    
    # TEST G: Double-Talk (Both speech-like)
    near_talk = generate_speech_like(1.0) # Near end is also speech-like
    # Shift the reference to simulate different speech
    ref_dt = generate_speech_like(1.0)[100:] + [0]*100
    echo_g = delayed_signal(ref_dt, delay_samples=20, attenuation=0.5)
    mic_g = [n + e for n, e in zip(near_talk, echo_g)]
    # Use smaller mu for double talk stability
    results["G_DOUBLE_TALK"] = run_aec_test("Double-Talk", mic_g, ref_dt, NLMSAEC(step_size=0.05), expected_near_end=near_talk, echo_signal=echo_g)
    
    # TEST H: Silence
    silence = [0.0] * int(SAMPLE_RATE * 1.0)
    results["H_SILENCE"] = run_aec_test("Silence", silence, silence, NLMSAEC(), expected_near_end=silence)
    
    # Acceptance Criteria Verification
    ac = {
        "stable": all(r["stable"] for r in results.values()),
        "no_clipping": not any(r["clipping"] for r in results.values()),
        "pure_echo_reduced": results["B_PURE_ECHO"]["echo_reduction_db"] > 10.0,
        "near_end_preserved": results["C_NEAR_END_ECHO"]["near_end_nmse"] < 0.5,
        "delays_converged": results["E_DELAY_SHORT"]["echo_reduction_db"] > 5.0 and results["E_DELAY_LONG"]["echo_reduction_db"] > 5.0
    }
    
    overall_status = "PASS" if all(ac.values()) else "FAIL"
    
    output = {
        "phase": "2B-4",
        "component": "AEC",
        "method": "NLMS",
        "tests": results,
        "acceptance_criteria": ac,
        "overall_status": overall_status,
        "physical_validation": False
    }
    
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tests/results')), exist_ok=True)
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tests/results/audio-aec-phase2b4.json')), 'w') as f:
        json.dump(output, f, indent=4)
        
    print(f"AEC Test Status: {overall_status}")

if __name__ == "__main__":
    test_aec()
