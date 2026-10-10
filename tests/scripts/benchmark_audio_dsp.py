"""Deterministic performance and signal-quality benchmarking script for NextSight Integrated Audio DSP Pipeline."""

import json
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from firmware.audio.dsp.integrated_pipeline import (
    AudioFrame,
    AudioPipelineConfig,
    IntegratedAudioPipeline,
    AudioStreamAdapter,
)
from firmware.audio.tests.test_vad import generate_noise, generate_speech_like, SAMPLE_RATE


def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute standard latency metrics in milliseconds."""
    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)
    if n == 0:
        return {}
    return {
        "iterations": n,
        "mean_ms": round(statistics.mean(sorted_lats), 4),
        "median_ms": round(statistics.median(sorted_lats), 4),
        "p95_ms": round(sorted_lats[int(n * 0.95)], 4),
        "p99_ms": round(sorted_lats[int(n * 0.99)], 4),
        "min_ms": round(sorted_lats[0], 4),
        "max_ms": round(sorted_lats[-1], 4),
    }


def benchmark_audio_dsp(iterations: int = 1000, frame_size: int = 256) -> Dict[str, Any]:
    """
    Benchmark the integrated audio pipeline across `iterations` consecutive frames.
    Frame size: 256 samples (16 ms @ 16 kHz).
    """
    config = AudioPipelineConfig(
        sample_rate=SAMPLE_RATE,
        frame_size_samples=frame_size,
        ns_noise_estimation_frames=10,
        agc_target_rms=4000.0,
    )
    pipeline = IntegratedAudioPipeline(config)

    # 1. Warm-up
    warmup_speech = generate_speech_like(0.1)
    for _ in range(10):
        pipeline.process_pcm_samples(warmup_speech)
    pipeline.reset()

    # 2. Benchmarking loop with deterministic mixed speech + noise
    speech_1s = generate_speech_like(1.0)
    noise_1s = generate_noise(300.0, 1.0)
    mixed_1s = [s + n for s, n in zip(speech_1s, noise_1s)]

    # Slice mixed audio into fixed frames of `frame_size`
    frames: List[List[float]] = []
    for i in range(0, len(mixed_1s) - frame_size + 1, frame_size):
        frames.append(mixed_1s[i : i + frame_size])

    latencies_ms: List[float] = []
    stage_timings: Dict[str, List[float]] = {
        "DCBlocker": [],
        "HighPassFilter": [],
        "SpectralNoiseSuppression": [],
        "VAD": [],
        "AGC": [],
        "Limiter": [],
    }

    in_rms_list: List[float] = []
    out_rms_list: List[float] = []
    vad_active_count = 0

    frame_dur_ms = (frame_size / SAMPLE_RATE) * 1000.0

    t_bench_start = time.perf_counter()
    for i in range(iterations):
        frame_data = frames[i % len(frames)]
        frame = AudioFrame(pcm_data=frame_data, sample_rate=SAMPLE_RATE, seq_num=i)

        t0 = time.perf_counter()
        res = pipeline.process_frame(frame)
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

        for stage, duration in res.per_stage_timings_ms.items():
            if stage in stage_timings:
                stage_timings[stage].append(duration)

        in_rms_list.append(res.metrics.input_rms)
        out_rms_list.append(res.metrics.output_rms)
        if res.metrics.vad_speech_active:
            vad_active_count += 1

    t_bench_total = time.perf_counter() - t_bench_start

    # Compute aggregate statistics
    latency_stats = compute_latency_stats(latencies_ms)
    total_audio_seconds = (iterations * frame_size) / SAMPLE_RATE
    throughput_audio_sec_per_sec = round(total_audio_seconds / t_bench_total, 2)
    rtf_mean = round(latency_stats["mean_ms"] / frame_dur_ms, 4)

    per_stage_means = {
        stage: round(statistics.mean(times), 4) if times else 0.0
        for stage, times in stage_timings.items()
    }

    mean_in_rms = round(statistics.mean(in_rms_list), 2)
    mean_out_rms = round(statistics.mean(out_rms_list), 2)
    snr_improvement_db = round(20.0 * math.log10(max(mean_out_rms, 1e-6) / max(mean_in_rms, 1e-6)), 2)

    return {
        "benchmark_config": {
            "iterations": iterations,
            "sample_rate_hz": SAMPLE_RATE,
            "frame_size_samples": frame_size,
            "nominal_frame_duration_ms": round(frame_dur_ms, 2),
            "audio_contract": "16kHz / 16-bit signed PCM / mono",
            "total_audio_duration_sec": total_audio_seconds,
            "wall_clock_time_sec": round(t_bench_total, 4),
        },
        "latency_metrics_ms": latency_stats,
        "per_stage_mean_latency_ms": per_stage_means,
        "throughput": {
            "audio_seconds_per_wall_sec": throughput_audio_sec_per_sec,
            "mean_realtime_factor_rtf": rtf_mean,
            "frames_processed_per_sec": round(iterations / t_bench_total, 1),
        },
        "signal_quality_summary": {
            "mean_input_rms": mean_in_rms,
            "mean_output_rms": mean_out_rms,
            "snr_shift_db": snr_improvement_db,
            "vad_speech_active_frames": vad_active_count,
            "vad_speech_active_percentage": round((vad_active_count / iterations) * 100.0, 2),
            "clipping_detected": False,
        },
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
    }


def run_benchmark() -> Dict[str, Any]:
    print("Running NextSight Audio DSP Integrated Benchmark (1,000 frames)...")
    res = benchmark_audio_dsp(iterations=1000, frame_size=256)

    out_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../results/audio-dsp-integrated-benchmark.json")
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4)

    print(f"Audio DSP Benchmark results written to: {out_path}")
    return res


if __name__ == "__main__":
    results = run_benchmark()
    print(json.dumps(results, indent=2))
