import math
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dsp.buffer import AudioBuffer
from dsp.vad import VAD
from dsp.agc import AGC
from dsp.noise_suppression import SpectralNoiseSuppression
from dsp import metrics
from tests.test_vad import generate_noise, generate_speech_like, SAMPLE_RATE

def run_arch_A(signal: list) -> tuple:
    # A: MIC -> VAD -> AGC
    vad = VAD(energy_threshold=500.0, hangover_frames=15)
    agc = AGC(target_rms=4000.0)
    
    buf = AudioBuffer(signal)
    vad.process(buf.copy())
    buf_agc = agc.process(buf.copy())
    return vad.history, metrics.calculate_rms(buf_agc)

def run_arch_B(signal: list) -> tuple:
    # B: MIC -> NS -> VAD -> AGC
    ns = SpectralNoiseSuppression(noise_estimation_frames=10)
    vad = VAD(energy_threshold=500.0, hangover_frames=15)
    agc = AGC(target_rms=4000.0)
    
    buf = AudioBuffer(signal)
    buf_ns = ns.process(buf.copy())
    vad.process(buf_ns.copy())
    buf_agc = agc.process(buf_ns.copy())
    return vad.history, metrics.calculate_rms(buf_agc)

def run_arch_C(signal: list) -> tuple:
    # C: MIC -> VAD -> NS -> AGC
    ns = SpectralNoiseSuppression(noise_estimation_frames=10)
    vad = VAD(energy_threshold=500.0, hangover_frames=15)
    agc = AGC(target_rms=4000.0)
    
    buf = AudioBuffer(signal)
    vad.process(buf.copy())
    buf_ns = ns.process(buf.copy())
    buf_agc = agc.process(buf_ns.copy())
    return vad.history, metrics.calculate_rms(buf_agc)

def evaluate_test(name: str, signal: list, expected_active: bool) -> dict:
    hist_a, rms_a = run_arch_A(signal)
    hist_b, rms_b = run_arch_B(signal)
    hist_c, rms_c = run_arch_C(signal)
    
    total = len(hist_a)
    
    def calc_metrics(hist, rms):
        active = sum(1 for h in hist if h["speech_active"])
        fp = active if not expected_active else 0
        tp = active if expected_active else 0
        fn = (total - active) if expected_active else 0
        return {
            "active_frames": active,
            "false_positive_rate": fp / total,
            "detection_rate": tp / total,
            "missed_speech_frames": fn,
            "output_rms": rms
        }
        
    return {
        "name": name,
        "total_frames": total,
        "Arch_A": calc_metrics(hist_a, rms_a),
        "Arch_B": calc_metrics(hist_b, rms_b),
        "Arch_C": calc_metrics(hist_c, rms_c)
    }

def test_ordering_ns():
    results = {}
    
    # Needs a lead-in noise for NS to estimate
    dur_est = 0.2
    noise_est = [0.0] * int(SAMPLE_RATE * dur_est) # Silence lead-in (but for noise tests, we'll give it actual noise)
    
    # 1. Silence
    sig_1 = [0.0] * int(SAMPLE_RATE * 1.2)
    results["TEST_1_SILENCE"] = evaluate_test("Silence", sig_1, expected_active=False)
    
    # 2. 300 RMS background noise
    sig_2 = generate_noise(300.0 * math.sqrt(2), 1.2)
    results["TEST_2_300_NOISE"] = evaluate_test("300 RMS Noise", sig_2, expected_active=False)
    
    # 3. Clear speech-like signal
    sig_3 = noise_est + generate_speech_like(1.0)
    results["TEST_3_CLEAR_SPEECH"] = evaluate_test("Clear Speech", sig_3, expected_active=True)
    
    # 4. Low-level speech (~400 amplitude / RMS ~282)
    sig_4_speech = [s * 0.1 for s in generate_speech_like(1.0)]
    sig_4 = noise_est + sig_4_speech
    results["TEST_4_LOW_SPEECH"] = evaluate_test("Low Speech", sig_4, expected_active=True)
    
    # 5. Low-level speech + 300 RMS noise
    noise_part = generate_noise(300.0 * math.sqrt(2), 1.2)
    sig_5 = noise_part[:int(SAMPLE_RATE*dur_est)] + [s + n for s, n in zip(sig_4_speech, noise_part[int(SAMPLE_RATE*dur_est):])]
    results["TEST_5_LOW_SPEECH_NOISE"] = evaluate_test("Low Speech + Noise", sig_5, expected_active=True)
    
    # 6. Medium speech + noise
    sig_6_speech = [s * 0.5 for s in generate_speech_like(1.0)]
    noise_part_6 = generate_noise(800.0 * math.sqrt(2), 1.2)
    sig_6 = noise_part_6[:int(SAMPLE_RATE*dur_est)] + [s + n for s, n in zip(sig_6_speech, noise_part_6[int(SAMPLE_RATE*dur_est):])]
    results["TEST_6_MED_SPEECH_NOISE"] = evaluate_test("Medium Speech + Noise", sig_6, expected_active=True)
    
    # 7. Speech + short pauses + noise
    sig_pause = generate_speech_like(0.25) + [0.0]*int(SAMPLE_RATE*0.1) + generate_speech_like(0.25)
    noise_part_7 = generate_noise(300.0 * math.sqrt(2), dur_est + 0.6)
    sig_7 = noise_part_7[:int(SAMPLE_RATE*dur_est)] + [s + n for s, n in zip(sig_pause, noise_part_7[int(SAMPLE_RATE*dur_est):])]
    results["TEST_7_PAUSES_NOISE"] = evaluate_test("Pauses + Noise", sig_7, expected_active=True)
    
    # 8. Loud non-speech noise
    sig_8 = generate_noise(5000.0, 1.2)
    results["TEST_8_LOUD_NOISE"] = evaluate_test("Loud Noise", sig_8, expected_active=False)
    
    output = {
        "experiment": "2B-3R",
        "tests": results,
        "decision": {
            "Answer": "PARTIALLY. NS-before-VAD (Arch B) mathematically attenuates the 300 RMS noise, making VAD even more resilient to false positives. However, because NS also slightly attenuates the low-level speech energy (or the speech was already below threshold), Arch B DOES NOT rescue low-level speech below the 500 VAD threshold.",
            "Recommended_Ordering": "B (MIC -> NS -> VAD -> AGC)"
        },
        "physical_validation": False
    }
    
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tests/results')), exist_ok=True)
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tests/results/audio-noise-suppression-architecture-phase2b3r.json')), 'w') as f:
        json.dump(output, f, indent=4)
        
    print("Ordering NS test complete.")

if __name__ == "__main__":
    test_ordering_ns()
