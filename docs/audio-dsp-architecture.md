# Audio DSP Architecture

## 1. Audio Data Contract
- **Sample Rate**: 16,000 Hz ($16\text{ kHz}$)
- **Bit Depth**: 16-bit signed integer (`int16`, range $[-32768, 32767]$)
- **Channel Count**: 1 (Mono)
- **Byte Order**: Little-Endian (`<h`)
- **Internal Float Representation**: `AudioBuffer` storing float samples scaled to $[-32768.0, 32767.0]$

---

## 2. Integrated DSP Pipeline Architecture

The NextSight integrated audio processing pipeline (`IntegratedAudioPipeline` in `firmware/audio/dsp/integrated_pipeline.py`) chains all validated DSP stages in a deterministic, frozen architecture sequence:

```
[Speaker Playback Reference] ──────────────────────────┐ (Conditional)
                                                       ▼
[Raw PCM Ingest] ──> [ DC Blocker ] ──> [ HighPass 80Hz ] ──> [ AEC + DTD ] ──> [ Spectral NS ] ──> [ VAD ] ──> [ AGC ] ──> [ Peak Limiter ] ──> [ Output PCM + Metrics ]
```

### Stage Ordering Details:
1. **Input Normalization & Validation**:
   - Accepts raw PCM bytes, float sample lists, or `AudioBuffer` instances.
   - Validates byte length (must be a multiple of 2 for 16-bit PCM).
   - Sanitizes `NaN` and `Inf` floating-point anomalies to 0.0 with warning flag.
   - Handles empty input safely with zeroed metrics.
2. **DC Blocker**:
   - First-order IIR high-pass filter ($y[n] = x[n] - x[n-1] + R \cdot y[n-1]$ with $R = 0.995$) to eliminate microphone hardware DC bias.
3. **High-Pass Filter (HPF)**:
   - First-order RC filter with $f_c = 80.0\text{ Hz}$ at $16\text{ kHz}$ to attenuate wind rumble and low-frequency handling vibrations.
4. **Acoustic Echo Cancellation with Double-Talk Detection (AEC + DTD)**:
   - Normalized Least Mean Squares (NLMS) filter with Geigel Double-Talk Detector (`NLMSAECWithDTD`).
   - Filter length: 256 samples ($16\text{ ms}$ tail), step size $\mu = 0.5$, regularization $\epsilon = 10^6$.
   - Geigel DTD threshold: 0.5, hold time: 800 samples ($50\text{ ms}$).
   - **Conditional Integration**: AEC operates when a valid far-end speaker playback reference frame is provided. When no reference is provided, AEC is safely bypassed and reported as `BYPASS_NO_REFERENCE`.
5. **Spectral Noise Suppression (NS)**:
   - Classical spectral subtraction with pure-Python FFT / IFFT and Hann windowing.
   - FFT size: 256, 50% overlap (hop size 128), $\alpha = 2.0$, spectral floor $= 0.05$.
6. **Voice Activity Detection (VAD)**:
   - Energy-based Voice Activity Detector operating on 10ms sub-frames (160 samples).
   - Energy threshold: 500.0, attack frames: 1, hangover frames: 15 (150ms).
7. **Automatic Gain Control (AGC)**:
   - Feed-forward envelope follower with fast-attack ($50\text{ ms}$) and slow-release ($200\text{ ms}$).
   - Target RMS: 4000.0, Max Gain: 10.0x, Min Gain: 0.1x, Noise Threshold: 100.0.
8. **Peak Limiter**:
   - Hard peak limiter clamping output samples to $[-32767.0, 32767.0]$ to prevent digital wrap-around distortion.
9. **Output Serialization & Structured Metrics**:
   - Serializes clean float samples to 16-bit signed PCM bytes (`<h`).
   - Produces structured `AudioProcessingMetrics`.

---

## 3. Interfaces and API

The pipeline is implemented in `firmware/audio/dsp/integrated_pipeline.py` and exported through `firmware/audio/dsp/pipeline.py` and `firmware/audio/dsp/__init__.py`.

### Primary Methods:
- **`process_pcm_bytes(mic_bytes: bytes, ref_bytes: Optional[bytes] = None) -> Tuple[bytes, AudioProcessingMetrics]`**:
  Accepts raw 16-bit PCM bytes and returns processed PCM bytes along with complete metrics.
- **`process_pcm_samples(mic_samples: List[float], ref_samples: Optional[List[float]] = None) -> Tuple[List[float], AudioProcessingMetrics]`**:
  Accepts float sample arrays and returns processed sample arrays with metrics.
- **`process_buffer(mic_buffer: AudioBuffer, ref_buffer: Optional[AudioBuffer] = None) -> Tuple[AudioBuffer, AudioProcessingMetrics]`**:
  Direct `AudioBuffer` interface.
- **`reset() -> None`**:
  Resets all filter internal states, adaptation histories, envelope followers, and queues.
- **`push_capture_chunk(chunk: bytes) -> bool` / `pop_processed_chunk() -> Optional[bytes]`**:
  Bounded FIFO streaming queue interface preventing memory growth.

---

## 4. Structured Metrics (`AudioProcessingMetrics`)

| Metric Field | Type | Description |
|---|---|---|
| `sample_rate` | `int` | Audio sample rate in Hz (16000) |
| `sample_count` | `int` | Number of samples processed in the frame |
| `frame_duration_ms` | `float` | Physical audio frame duration in milliseconds |
| `processing_time_ms` | `float` | Host CPU processing latency in milliseconds |
| `realtime_factor` | `float` | Processing time divided by audio frame duration |
| `input_rms` / `output_rms` | `float` | Root Mean Square energy before and after DSP chain |
| `input_peak` / `output_peak` | `float` | Maximum absolute amplitude before and after DSP |
| `input_dc_offset` / `output_dc_offset` | `float` | Mean DC offset before and after filtering |
| `clipping_detected` | `bool` | True if output samples hit the ceiling threshold |
| `clipping_percentage` | `float` | Percentage of samples clamped by the limiter |
| `snr_estimate_db` | `Optional[float]` | Approximate signal-to-noise ratio / energy shift in dB |
| `vad_speech_active` | `bool` | Voice activity detection status for the frame |
| `vad_active_frames` / `vad_total_frames` | `int` | Count of active speech sub-frames vs total evaluated |
| `aec_status` | `str` | `"ACTIVE"`, `"BYPASS_NO_REFERENCE"`, or `"DISABLED"` |
| `dt_detected_frames` | `int` | Double-talk detected frames count when AEC is active |
| `stages_executed` | `List[str]` | List of DSP stage names executed in sequence |
| `error` | `Optional[str]` | Error or warning description (e.g. `"NaN/Inf sanitized"`) |

---

## 5. Memory Safety & Bounded Resource Management

- **Bounded Stream Queues**: Ingestion and egress FIFO queues enforce `max_queue_size = 50` chunks (default) to strictly prevent memory leaks or unbounded growth during network/buffer delays.
- **Stateful History Bounds**: All filter histories, DTD buffers, and AEC weights use fixed-length ring buffers or lists proportional to filter length ($N \le 256$ samples, $\sim 3\text{ KB}$ RAM total).
- **Idempotent Reset**: The `reset()` method clears all queues and re-initializes all stages deterministically.

---

## 6. Known Limitations

1. **Quiet Speech Gating (Known Limitation)**: Low-level speech beneath the fixed $500.0$ energy threshold is gated out as silence by the VAD before the AGC can amplify it. Dynamic noise floor tracking will be required for whisper-level speech.
2. **Spectral Noise Estimation Lead-In**: The spectral subtraction NS currently computes its initial noise profile over the first 10 frames ($160\text{ ms}$). Dynamic continuous noise profiling is planned for Phase 12.
3. **Reference Software Model**: This implementation is a software model in pure Python for architectural and numerical verification. Real-time ESP32 hardware execution will require the C++/ESP-DSP SIMD implementation.

---

## 7. Reproduction and Test Commands

To run all focused audio unit and integration tests:
```bash
python -m pytest firmware/audio/tests/ -v
```

To run the complete repository validation suite:
```bash
python run_full_validation.py
```
