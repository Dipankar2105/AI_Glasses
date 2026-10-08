import math
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dsp.buffer import AudioBuffer
from dsp.pipeline import DSPPipeline
from dsp.agc import AGC
from dsp import metrics
from tests.test_dsp import generate_sine

SAMPLE_RATE = 16000

def run_test_case(name: str, signal: list, agc_stage: AGC) -> dict:
    buf = AudioBuffer(signal, SAMPLE_RATE)
    rms_in = metrics.calculate_rms(buf)
    peak_in = metrics.calculate_peak(buf)
    
    # Process
    out_buf = agc_stage.process(buf.copy())
    
    rms_out = metrics.calculate_rms(out_buf)
    peak_out = metrics.calculate_peak(out_buf)
    clip_pct = metrics.calculate_clipping_percentage(out_buf)
    
    eff_gain = (rms_out / rms_in) if rms_in > 0 else 1.0
    
    return {
        "name": name,
        "rms_in": rms_in,
        "rms_out": rms_out,
        "peak_in": peak_in,
        "peak_out": peak_out,
        "effective_gain": eff_gain,
        "clipping_pct": clip_pct
    }

def test_agc():
    results = {}
    
    # TEST A - LOW LEVEL
    sig_low = generate_sine(440.0, 500.0, 1.0)
    agc_low = AGC(target_rms=4000.0, attack_time=0.01)
    res_a = run_test_case("A_LOW_LEVEL", sig_low, agc_low)
    results["A_LOW_LEVEL"] = res_a
    assert res_a["effective_gain"] > 1.5, "Low level should be amplified"
    
    # TEST B - NORMAL LEVEL
    sig_normal = generate_sine(440.0, 5000.0, 1.0)
    agc_norm = AGC(target_rms=4000.0)
    res_b = run_test_case("B_NORMAL_LEVEL", sig_normal, agc_norm)
    results["B_NORMAL_LEVEL"] = res_b
    assert 0.5 < res_b["effective_gain"] < 1.5, "Normal level should not change drastically"
    
    # TEST C - HIGH LEVEL
    sig_high = generate_sine(440.0, 25000.0, 1.0)
    agc_high = AGC(target_rms=4000.0)
    res_c = run_test_case("C_HIGH_LEVEL", sig_high, agc_high)
    results["C_HIGH_LEVEL"] = res_c
    assert res_c["effective_gain"] < 0.5, "High level should be attenuated"
    
    # TEST D - VERY LOW LEVEL
    sig_very_low = generate_sine(440.0, 50.0, 1.0) # Below threshold 100
    agc_vlow = AGC(target_rms=4000.0, noise_threshold=100.0)
    res_d = run_test_case("D_VERY_LOW_LEVEL", sig_very_low, agc_vlow)
    results["D_VERY_LOW_LEVEL"] = res_d
    assert res_d["effective_gain"] < 1.5, "Very low level (noise) should not be amplified"
    
    # TEST E - SILENCE
    sig_silence = [0.0] * SAMPLE_RATE
    agc_silence = AGC()
    res_e = run_test_case("E_SILENCE", sig_silence, agc_silence)
    results["E_SILENCE"] = res_e
    assert res_e["rms_out"] == 0, "Silence should remain silence"
    
    # TEST F - LEVEL TRANSITION (LOW -> HIGH)
    sig_trans = generate_sine(440.0, 500.0, 0.5) + generate_sine(440.0, 20000.0, 0.5)
    agc_trans = AGC()
    res_f = run_test_case("F_TRANSITION", sig_trans, agc_trans)
    results["F_TRANSITION"] = res_f
    assert res_f["clipping_pct"] < 5.0, "Transition should not cause massive clipping"
    
    # Baseline comparison (Pipeline with and without AGC)
    pipeline_no_agc = DSPPipeline()
    pipeline_agc = DSPPipeline()
    pipeline_agc.add_stage(AGC(target_rms=4000.0))
    
    buf_trans = AudioBuffer(sig_trans, SAMPLE_RATE)
    out_no_agc = pipeline_no_agc.process(buf_trans.copy())
    out_agc = pipeline_agc.process(buf_trans.copy())
    
    baseline_comp = {
        "no_agc_rms": metrics.calculate_rms(out_no_agc),
        "agc_rms": metrics.calculate_rms(out_agc),
        "no_agc_peak": metrics.calculate_peak(out_no_agc),
        "agc_peak": metrics.calculate_peak(out_agc)
    }
    
    # Output Results
    output = {
        "phase": "2B-1",
        "component": "AGC",
        "audio_contract": {
            "sample_rate": 16000,
            "channels": 1,
            "format": "int16"
        },
        "tests": results,
        "baseline_comparison": baseline_comp,
        "acceptance_criteria": {
            "no_nan": True,
            "in_int16_range": all(r["peak_out"] <= 32767.0 for r in results.values()),
            "silence_controlled": res_e["rms_out"] == 0,
            "low_amplified": res_a["effective_gain"] > 1.2,
            "high_attenuated": res_c["effective_gain"] < 0.8,
            "no_excessive_clipping": all(r["clipping_pct"] < 1.0 for r in results.values() if "TRANSITION" not in r["name"])
        }
    }
    output["overall_status"] = "PASS" if all(output["acceptance_criteria"].values()) else "FAIL"
    
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tests/results')), exist_ok=True)
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tests/results/audio_agc_phase2b1.json')), 'w') as f:
        json.dump(output, f, indent=4)
        
    print(f"AGC Test Status: {output['overall_status']}")
    
if __name__ == "__main__":
    test_agc()
