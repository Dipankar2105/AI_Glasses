import math
import json
import os
import sys

# Add firmware/audio to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dsp.buffer import AudioBuffer
from dsp.pipeline import DSPPipeline
from dsp.filters import DCBlocker, HighPassFilter
from dsp.dynamics import Gain, Limiter
from dsp import metrics

SAMPLE_RATE = 16000

# Synthetic Generators
def generate_sine(frequency: float, amplitude: float, duration_sec: float) -> list:
    n_samples = int(SAMPLE_RATE * duration_sec)
    return [amplitude * math.sin(2 * math.pi * frequency * i / SAMPLE_RATE) for i in range(n_samples)]

def generate_sine_with_dc(frequency: float, amplitude: float, dc_offset: float, duration_sec: float) -> list:
    sine = generate_sine(frequency, amplitude, duration_sec)
    return [x + dc_offset for x in sine]

def test_dc_blocker():
    sig = generate_sine_with_dc(440.0, 1000.0, 5000.0, 0.5)
    buf = AudioBuffer(sig, SAMPLE_RATE)
    
    assert abs(metrics.calculate_dc_offset(buf) - 5000.0) < 1.0, "Input should have DC offset"
    
    blocker = DCBlocker()
    out_buf = blocker.process(buf)
    
    out_dc = metrics.calculate_dc_offset(out_buf)
    assert abs(out_dc) < 150.0, f"DC offset not removed, got {out_dc}"
    print("DCBlocker test passed.")

def test_high_pass():
    # 50 Hz sine (should be attenuated) + 1000 Hz sine (should pass)
    n_samples = int(SAMPLE_RATE * 0.5)
    sig_low = [1000.0 * math.sin(2 * math.pi * 10 * i / SAMPLE_RATE) for i in range(n_samples)]
    sig_high = [1000.0 * math.sin(2 * math.pi * 1000 * i / SAMPLE_RATE) for i in range(n_samples)]
    
    buf_low = AudioBuffer(sig_low, SAMPLE_RATE)
    buf_high = AudioBuffer(sig_high, SAMPLE_RATE)
    
    hpf = HighPassFilter(cutoff_freq=200.0, sample_rate=SAMPLE_RATE)
    
    out_low = hpf.process(buf_low)
    out_high = hpf.process(AudioBuffer(sig_high, SAMPLE_RATE)) # new instance
    
    rms_low_in = metrics.calculate_rms(AudioBuffer(sig_low))
    rms_low_out = metrics.calculate_rms(out_low)
    
    rms_high_in = metrics.calculate_rms(AudioBuffer(sig_high))
    rms_high_out = metrics.calculate_rms(out_high)
    
    assert rms_low_out < rms_low_in * 0.5, "Low frequency not attenuated enough"
    assert rms_high_out > rms_high_in * 0.9, "High frequency attenuated too much"
    print("HighPassFilter test passed.")

def test_gain():
    sig = generate_sine(440.0, 1000.0, 0.1)
    buf = AudioBuffer(sig, SAMPLE_RATE)
    
    rms_in = metrics.calculate_rms(buf)
    gain = Gain(2.5)
    out_buf = gain.process(buf)
    
    rms_out = metrics.calculate_rms(out_buf)
    
    assert abs(rms_out - (rms_in * 2.5)) < 1.0, "Gain not applied correctly"
    print("Gain test passed.")

def test_limiter():
    sig = generate_sine(440.0, 40000.0, 0.1) # Exceeds int16
    buf = AudioBuffer(sig, SAMPLE_RATE)
    
    limiter = Limiter(threshold=32767.0)
    out_buf = limiter.process(buf)
    
    peak = metrics.calculate_peak(out_buf)
    assert peak <= 32767.0, f"Limiter failed, peak is {peak}"
    
    clip_pct = metrics.calculate_clipping_percentage(out_buf)
    assert clip_pct > 0, "Signal should be clipped"
    print("Limiter test passed.")

def run_pipeline_test():
    # Signal: 100 Hz (attenuated by HPF), DC offset, large amplitude
    n_samples = int(SAMPLE_RATE * 1.0)
    sig = []
    for i in range(n_samples):
        val = 5000.0 + 10000.0 * math.sin(2 * math.pi * 50 * i / SAMPLE_RATE) + 15000.0 * math.sin(2 * math.pi * 1000 * i / SAMPLE_RATE)
        sig.append(val)
    
    buf = AudioBuffer(sig, SAMPLE_RATE)
    
    pipeline = DSPPipeline()
    pipeline.add_stage(DCBlocker())
    pipeline.add_stage(HighPassFilter(cutoff_freq=200.0))
    pipeline.add_stage(Gain(2.0))
    pipeline.add_stage(Limiter(threshold=32767.0))
    
    metrics_before = {
        "rms": metrics.calculate_rms(buf),
        "peak": metrics.calculate_peak(buf),
        "dc_offset": metrics.calculate_dc_offset(buf),
        "clipping_pct": metrics.calculate_clipping_percentage(buf)
    }
    
    out_buf = pipeline.process(buf)
    
    metrics_after = {
        "rms": metrics.calculate_rms(out_buf),
        "peak": metrics.calculate_peak(out_buf),
        "dc_offset": metrics.calculate_dc_offset(out_buf),
        "clipping_pct": metrics.calculate_clipping_percentage(out_buf)
    }
    
    results = {
        "test": "Pipeline E2E",
        "sample_rate": SAMPLE_RATE,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "status": "PASS" if abs(metrics_after["dc_offset"]) < 150 and metrics_after["peak"] <= 32767.0 else "FAIL"
    }
    
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tests/results')), exist_ok=True)
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tests/results/audio_dsp_phase2a.json')), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Pipeline test finished with status: {results['status']}")
    
if __name__ == "__main__":
    print("Running DSP Unit Tests...")
    test_dc_blocker()
    test_high_pass()
    test_gain()
    test_limiter()
    print("Running Pipeline E2E Test...")
    run_pipeline_test()
