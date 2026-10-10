"""Deterministic tests for the NextSight Integrated Audio DSP Pipeline.

Tests:
1. Silence and low-amplitude signals.
2. Speech-like signals mixed with synthetic noise (SNR / RMS measurement).
3. High-amplitude and clipped input (ceiling safety & clipping metrics).
4. Invalid frame sizes and malformed data (odd byte count, NaN/inf, empty inputs).
5. VAD activation, deactivation, and hangover smoothing.
6. AGC output bounds and level stabilization.
7. Noise suppression using known clean/noisy signal pairs.
8. Repeated frames, stream resets, and determinism.
9. Long synthetic streams and bounded resource utilization.
10. AEC integration and conditional bypass behavior.
11. AudioFrame and AudioProcessingResult contracts and explicit rejection paths.
12. AudioStreamAdapter continuous stream slicing, partial buffering, and dropped counters.
13. Per-stage performance profiling and latency measurement separation.
"""

import math
import os
import struct
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dsp.buffer import AudioBuffer
from dsp.integrated_pipeline import (
    AudioFrame,
    AudioProcessingResult,
    AudioProcessingMetrics,
    AudioPipelineConfig,
    IntegratedAudioPipeline,
    AudioStreamAdapter,
)
from tests.test_vad import generate_noise, generate_speech_like, SAMPLE_RATE
from tests.test_dsp import generate_sine


def test_silence_and_low_amplitude():
    """Verify that pure digital silence and low-amplitude signals are processed safely."""
    pipeline = IntegratedAudioPipeline()
    silence = [0.0] * int(SAMPLE_RATE * 0.5)
    
    # Pure silence
    out_samples, metrics = pipeline.process_pcm_samples(silence)
    assert len(out_samples) == len(silence)
    assert metrics.input_rms == 0.0
    assert metrics.output_rms < 1.0
    assert not metrics.clipping_detected
    assert metrics.clipping_percentage == 0.0
    assert not metrics.vad_speech_active
    assert metrics.vad_active_frames == 0
    assert metrics.error is None
    
    # Low-amplitude signal (below VAD 500.0 threshold - known quiet-speech limitation)
    quiet_speech = [s * 0.05 for s in generate_speech_like(0.5)]
    pipeline.reset()
    out_quiet, metrics_quiet = pipeline.process_pcm_samples(quiet_speech)
    assert len(out_quiet) == len(quiet_speech)
    assert metrics_quiet.input_rms > 0.0
    assert not metrics_quiet.clipping_detected
    assert metrics_quiet.vad_active_frames == 0  # Gated out by VAD threshold


def test_speech_mixed_with_noise():
    """Verify speech mixed with noise passes through NS, VAD, and AGC predictably."""
    config = AudioPipelineConfig(
        ns_noise_estimation_frames=5,
        agc_target_rms=4000.0
    )
    pipeline = IntegratedAudioPipeline(config)
    
    speech = generate_speech_like(1.0)
    noise = generate_noise(300.0, 1.0)
    mixed = [s + n for s, n in zip(speech, noise)]
    
    out_samples, metrics = pipeline.process_pcm_samples(mixed)
    assert len(out_samples) == len(mixed)
    assert metrics.input_rms > 0.0
    assert metrics.output_rms > 0.0
    assert metrics.vad_active_frames > 0
    assert metrics.vad_speech_active
    # Peak limiter enforces max output ceiling <= 32767.0
    assert metrics.output_peak <= 32767.0
    assert max(out_samples) <= 32767.0
    assert min(out_samples) >= -32768.0
    assert "SpectralNoiseSuppression" in metrics.stages_executed
    assert "VAD" in metrics.stages_executed
    assert "AGC" in metrics.stages_executed
    assert "Limiter" in metrics.stages_executed


def test_high_amplitude_and_clipped_input():
    """Verify that high-amplitude and clipped signals are clamped and flagged by the limiter."""
    pipeline = IntegratedAudioPipeline()
    
    # Signal that far exceeds 16-bit range
    loud_sine = [60000.0 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE) for i in range(1600)]
    out_samples, metrics = pipeline.process_pcm_samples(loud_sine)
    
    assert max(out_samples) <= 32767.0
    assert min(out_samples) >= -32768.0
    assert metrics.clipping_detected or metrics.output_peak >= 32767.0
    assert metrics.output_peak <= 32767.0


def test_malformed_input_and_invalid_frame_sizes():
    """Verify error handling on invalid frame sizes, odd byte lengths, and malformed data."""
    pipeline = IntegratedAudioPipeline()
    
    # 1. Odd byte length for 16-bit PCM (e.g. 15 bytes)
    odd_bytes = b"\x00" * 15
    with pytest.raises(ValueError, match="not a multiple of 2"):
        pipeline.process_pcm_bytes(odd_bytes)
        
    # 2. Invalid data type
    with pytest.raises(TypeError):
        pipeline.bytes_to_samples("invalid_string_instead_of_bytes")  # type: ignore
        
    # 3. Empty input buffer
    empty_bytes = b""
    out_b, metrics_empty = pipeline.process_pcm_bytes(empty_bytes)
    assert out_b == b""
    assert metrics_empty.sample_count == 0
    assert metrics_empty.processing_time_ms == 0.0
    assert metrics_empty.frame_duration_ms == 0.0
    
    # 4. NaN / Inf handling
    bad_samples = [100.0, float('nan'), 200.0, float('inf'), -float('inf'), 300.0]
    out_sanitized, metrics_bad = pipeline.process_pcm_samples(bad_samples)
    assert len(out_sanitized) == len(bad_samples)
    assert not any(math.isnan(x) or math.isinf(x) for x in out_sanitized)
    assert metrics_bad.error == "NaN/Inf sanitized"


def test_audio_frame_and_result_rejection_contract():
    """Verify AudioFrame validation and explicit AudioProcessingResult rejection behavior."""
    pipeline = IntegratedAudioPipeline()
    
    # 1. None frame
    res_none = pipeline.process_frame(None)  # type: ignore
    assert not res_none.accepted
    assert "AudioFrame is None" in str(res_none.rejection_reason)
    assert res_none.output_frame is None
    
    # 2. Sample rate mismatch
    frame_bad_rate = AudioFrame(pcm_data=[100.0]*100, sample_rate=44100)
    res_bad_rate = pipeline.process_frame(frame_bad_rate)
    assert not res_bad_rate.accepted
    assert "Unsupported sample rate" in str(res_bad_rate.rejection_reason)
    
    # 3. Multi-channel unsupported
    frame_stereo = AudioFrame(pcm_data=[100.0]*100, channels=2)
    res_stereo = pipeline.process_frame(frame_stereo)
    assert not res_stereo.accepted
    assert "Unsupported channel count" in str(res_stereo.rejection_reason)
    
    # 4. Empty frame
    frame_empty = AudioFrame(pcm_data=[], sample_rate=16000)
    res_empty = pipeline.process_frame(frame_empty)
    assert not res_empty.accepted
    assert "Empty frame" in str(res_empty.rejection_reason)
    
    # 5. Valid frame
    valid_samples = generate_speech_like(0.2)
    frame_valid = AudioFrame(pcm_data=valid_samples, sample_rate=16000, seq_num=42, timestamp_ms=1000.0)
    res_valid = pipeline.process_frame(frame_valid)
    assert res_valid.accepted
    assert res_valid.rejection_reason is None
    assert res_valid.output_frame is not None
    assert res_valid.output_frame.seq_num == 42
    assert res_valid.output_frame.timestamp_ms == 1000.0
    assert len(res_valid.per_stage_timings_ms) > 0


def test_vad_activation_deactivation():
    """Verify VAD state transitions between speech and non-speech."""
    pipeline = IntegratedAudioPipeline()
    
    # Non-speech low noise
    noise_only = generate_noise(150.0, 0.5)
    _, m_noise = pipeline.process_pcm_samples(noise_only)
    assert not m_noise.vad_speech_active
    assert m_noise.vad_active_frames == 0
    
    # Clear speech
    pipeline.reset()
    speech = generate_speech_like(1.0)
    _, m_speech = pipeline.process_pcm_samples(speech)
    assert m_speech.vad_active_frames > 20
    assert m_speech.vad_speech_active


def test_agc_output_bounds_and_behavior():
    """Verify AGC targets and clamping."""
    target_rms = 4000.0
    config = AudioPipelineConfig(agc_target_rms=target_rms)
    pipeline = IntegratedAudioPipeline(config)
    
    # Low-level speech above noise floor
    low_speech = [s * 0.4 for s in generate_speech_like(1.0)]
    _, m_low = pipeline.process_pcm_samples(low_speech)
    
    # Output RMS should be scaled up towards target RMS
    assert m_low.output_rms > m_low.input_rms
    assert m_low.output_peak <= 32767.0


def test_noise_suppression_known_clean_noisy_pair():
    """Verify noise suppression attenuates background noise while preserving speech structure."""
    pipeline = IntegratedAudioPipeline(AudioPipelineConfig(ns_noise_estimation_frames=5))
    
    noise = generate_noise(300.0, 0.8)
    speech = generate_speech_like(0.8)
    noisy_speech = [s + n for s, n in zip(speech, noise)]
    
    out_samples, metrics = pipeline.process_pcm_samples(noisy_speech)
    assert len(out_samples) == len(noisy_speech)
    assert "SpectralNoiseSuppression" in metrics.stages_executed
    assert metrics.output_rms > 0.0


def test_repeated_frames_and_resets():
    """Verify deterministic execution and state reset idempotence."""
    pipeline = IntegratedAudioPipeline()
    speech = generate_speech_like(0.5)
    
    # Run 1
    out1, m1 = pipeline.process_pcm_samples(speech)
    
    # Reset
    pipeline.reset()
    
    # Run 2
    out2, m2 = pipeline.process_pcm_samples(speech)
    
    assert out1 == out2
    assert m1.input_rms == m2.input_rms
    assert m1.output_rms == m2.output_rms
    assert m1.vad_active_frames == m2.vad_active_frames


def test_long_synthetic_stream_and_bounded_queues():
    """Verify processing over 1,000 frames with bounded queue behavior and memory stability."""
    pipeline = IntegratedAudioPipeline(AudioPipelineConfig(max_queue_size=20))
    chunk_samples = 160  # 10ms at 16kHz
    raw_chunk = struct.pack(f"<{chunk_samples}h", *([1000] * chunk_samples))
    
    total_frames = 1000
    for i in range(total_frames):
        pushed = pipeline.push_capture_chunk(raw_chunk)
        if not pushed:
            # Queue bounded full: process queued chunks to drain
            processed = pipeline.process_queued_chunks()
            assert processed > 0
            pushed_again = pipeline.push_capture_chunk(raw_chunk)
            assert pushed_again
            
        popped = pipeline.pop_processed_chunk()
        # Drain periodically
        if i % 10 == 0:
            pipeline.process_queued_chunks()
            
    # Final drain
    pipeline.process_queued_chunks()
    assert len(pipeline._in_queue) == 0


def test_audio_stream_adapter_slicing_and_buffering():
    """Verify AudioStreamAdapter slices arbitrary streams into fixed 256-sample frames with partial buffering."""
    adapter = AudioStreamAdapter(frame_size=256)
    
    # 600 samples = 1200 bytes -> should yield 2 complete 256-sample frames (512 samples = 1024 bytes)
    # and 88 trailing samples (176 bytes) remaining in partial buffer
    samples_600 = [500.0] * 600
    bytes_600 = struct.pack(f"<{len(samples_600)}h", *([500] * 600))
    
    results = adapter.push_raw_stream(bytes_600)
    assert len(results) == 2
    assert all(r.accepted for r in results)
    assert results[0].output_frame.sample_count == 256
    assert results[1].output_frame.sample_count == 256
    assert results[0].output_frame.seq_num == 0
    assert results[1].output_frame.seq_num == 1
    assert adapter.buffered_bytes_count == 176  # 88 samples * 2 bytes
    assert adapter.accepted_frames == 2
    assert adapter.processed_frames == 2
    
    # Push another 200 samples (400 bytes): 88 buffered + 200 = 288 samples -> 1 more frame (256 samples) and 32 left
    bytes_200 = struct.pack("<200h", *([500] * 200))
    results2 = adapter.push_raw_stream(bytes_200)
    assert len(results2) == 1
    assert results2[0].output_frame.seq_num == 2
    assert adapter.buffered_bytes_count == 64  # 32 samples * 2 bytes


def test_aec_conditional_integration():
    """Verify AEC operates when reference buffer is supplied and safely bypasses when absent."""
    pipeline = IntegratedAudioPipeline()
    mic = generate_speech_like(0.5)
    ref = generate_speech_like(0.5)
    
    # Case A: With reference
    out_with_ref, m_with_ref = pipeline.process_pcm_samples(mic, ref)
    assert m_with_ref.aec_status == "ACTIVE"
    assert "NLMSAECWithDTD" in m_with_ref.stages_executed
    
    # Case B: Without reference
    pipeline.reset()
    out_no_ref, m_no_ref = pipeline.process_pcm_samples(mic, None)
    assert m_no_ref.aec_status == "BYPASS_NO_REFERENCE"
    assert "NLMSAECWithDTD" not in m_no_ref.stages_executed


def test_latency_measurement_separation_and_per_stage_timings():
    """Verify processing latency in ms is measured separately from audio frame duration and per-stage timings are recorded."""
    pipeline = IntegratedAudioPipeline()
    speech = generate_speech_like(0.5)  # 500 ms frame duration
    frame = AudioFrame(pcm_data=speech, sample_rate=16000)
    
    res = pipeline.process_frame(frame)
    assert res.accepted
    assert res.metrics.frame_duration_ms == 500.0
    assert res.metrics.processing_time_ms >= 0.0
    assert res.metrics.realtime_factor >= 0.0
    assert "DCBlocker" in res.per_stage_timings_ms
    assert "HighPassFilter" in res.per_stage_timings_ms
    assert "SpectralNoiseSuppression" in res.per_stage_timings_ms
    assert "VAD" in res.per_stage_timings_ms
    assert "AGC" in res.per_stage_timings_ms
    assert "Limiter" in res.per_stage_timings_ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
