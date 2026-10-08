import math
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dsp.buffer import AudioBuffer
from dsp.pipeline import DSPPipeline
from dsp.vad import VAD
from dsp.agc import AGC
from tests.test_dsp import generate_sine

SAMPLE_RATE = 16000

def generate_noise(amplitude: float, duration_sec: float) -> list:
    # Deterministic pseudo-random noise
    n_samples = int(SAMPLE_RATE * duration_sec)
    return [amplitude * math.sin(i * 1234.56) * math.cos(i * 789.01) for i in range(n_samples)]

def generate_speech_like(duration_sec: float) -> list:
    # Complex deterministic signal acting as speech (modulated)
    n_samples = int(SAMPLE_RATE * duration_sec)
    out = []
    for i in range(n_samples):
        t = i / SAMPLE_RATE
        envelope = max(0, math.sin(2 * math.pi * 2 * t)) # 2 Hz modulation
        carrier = math.sin(2 * math.pi * 300 * t) + 0.5 * math.sin(2 * math.pi * 1500 * t)
        out.append(4000.0 * envelope * carrier)
    return out

def analyze_vad(history: list, expected_active_frames_range: tuple) -> dict:
    total = len(history)
    active = sum(1 for h in history if h["speech_active"])
    raw_active = sum(1 for h in history if h["raw_active"])
    
    pass_test = expected_active_frames_range[0] <= active <= expected_active_frames_range[1]
    return {
        "total_frames": total,
        "active_frames": active,
        "raw_active_frames": raw_active,
        "pass": pass_test
    }

def test_vad():
    results = {}
    
    # TEST A - PURE SILENCE
    sig_silence = [0.0] * int(SAMPLE_RATE * 1.0)
    vad_a = VAD(energy_threshold=500.0)
    vad_a.process(AudioBuffer(sig_silence))
    res_a = analyze_vad(vad_a.history, (0, 0))
    results["A_SILENCE"] = res_a
    
    # TEST B - LOW LEVEL NOISE
    sig_noise = generate_noise(200.0, 1.0)
    vad_b = VAD(energy_threshold=500.0)
    vad_b.process(AudioBuffer(sig_noise))
    res_b = analyze_vad(vad_b.history, (0, 0))
    results["B_LOW_NOISE"] = res_b
    
    # TEST C - CLEAR SPEECH-LIKE SIGNAL
    sig_speech = generate_speech_like(1.0)
    vad_c = VAD(energy_threshold=500.0)
    vad_c.process(AudioBuffer(sig_speech))
    res_c = analyze_vad(vad_c.history, (20, 95)) # Should detect a significant chunk
    results["C_SPEECH_LIKE"] = res_c
    
    # TEST D - LOUD NON-SPEECH NOISE
    sig_loud_noise = generate_noise(5000.0, 1.0)
    vad_d = VAD(energy_threshold=500.0)
    vad_d.process(AudioBuffer(sig_loud_noise))
    res_d = analyze_vad(vad_d.history, (90, 100)) # Will falsely detect as speech (energy weakness)
    results["D_LOUD_NOISE_FALSE_POS"] = res_d
    
    # TEST E - SPEECH WITH PAUSES
    # Generate 0.25s speech-like to avoid the 0.25s silence embedded in generate_speech_like
    sig_pause = generate_speech_like(0.25) + [0.0]*int(SAMPLE_RATE*0.1) + generate_speech_like(0.25)
    vad_e = VAD(energy_threshold=500.0, hangover_frames=15) # 150ms hangover > 100ms pause
    vad_e.process(AudioBuffer(sig_pause))
    # Because hangover > pause, it should bridge the gap.
    # We compare to a VAD with NO hangover to prove baseline improvement.
    vad_e_nohang = VAD(energy_threshold=500.0, hangover_frames=1)
    vad_e_nohang.process(AudioBuffer(sig_pause))
    
    transitions_hangover = sum(1 for i in range(1, len(vad_e.history)) if vad_e.history[i]["speech_active"] != vad_e.history[i-1]["speech_active"])
    transitions_nohang = sum(1 for i in range(1, len(vad_e_nohang.history)) if vad_e_nohang.history[i]["speech_active"] != vad_e_nohang.history[i-1]["speech_active"])
    
    results["E_PAUSE_SMOOTHING"] = {
        "transitions_with_hangover": transitions_hangover,
        "transitions_without_hangover": transitions_nohang,
        "pass": transitions_hangover < transitions_nohang
    }
    
    # AGC INTERACTION TEST
    agc = AGC(target_rms=4000.0)
    sig_agc_test = generate_noise(300.0, 1.0) # Low noise
    
    # Without AGC -> Should be silence for VAD
    vad_no_agc = VAD(energy_threshold=500.0)
    vad_no_agc.process(AudioBuffer(sig_agc_test))
    
    # With AGC -> AGC amplifies noise to 4000 RMS -> VAD falsely triggers!
    buf = AudioBuffer(sig_agc_test)
    buf = agc.process(buf)
    vad_with_agc = VAD(energy_threshold=500.0)
    vad_with_agc.process(buf)
    
    results["AGC_INTERACTION"] = {
        "vad_active_no_agc": analyze_vad(vad_no_agc.history, (0, 100))["active_frames"],
        "vad_active_with_agc": analyze_vad(vad_with_agc.history, (0, 100))["active_frames"]
    }
    
    # ACCEPTANCE CRITERIA
    ac = {
        "silence_detected": res_a["pass"],
        "low_noise_ignored": res_b["pass"],
        "speech_detected": res_c["pass"],
        "hangover_improves_fragmentation": results["E_PAUSE_SMOOTHING"]["pass"],
        "agc_interaction_recorded": True
    }
    
    overall = "PASS" if all(ac.values()) else "FAIL"
    
    output = {
        "phase": "2B-2",
        "component": "VAD",
        "audio_contract": {"sample_rate": 16000, "channels": 1, "format": "int16"},
        "tests": results,
        "acceptance_criteria": ac,
        "overall_status": overall
    }
    
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tests/results')), exist_ok=True)
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tests/results/audio_vad_phase2b2.json')), 'w') as f:
        json.dump(output, f, indent=4)
        
    print(f"VAD Test Status: {overall}")

if __name__ == "__main__":
    test_vad()
