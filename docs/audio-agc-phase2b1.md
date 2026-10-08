# Audio AGC Evaluation (Phase 2B-1)

## 1. Objective
To evaluate whether an Automatic Gain Control (AGC) stage provides a measurable, stable improvement to microphone signals without introducing distortion or instability.

## 2. Why AGC is Evaluated
Microphone signals typically vary greatly in amplitude depending on speaker distance and voice projection. An AGC standardizes the amplitude before it reaches the backend AI processing, improving recognition rates.

## 3. AGC Architecture
The AGC is a modular `DSPStage` using a feed-forward envelope detector. It calculates a short-term amplitude envelope and adjusts a running gain multiplier to match a `target_rms`.

## 4. Parameters
- **Target RMS**: 4000.0 (Target volume level)
- **Min/Max Gain**: 0.1 to 10.0 (Bounds the multiplier)
- **Attack Time**: 0.05s (Fast adaptation to loud sounds)
- **Release Time**: 0.2s (Slower recovery to prevent pumping)
- **Noise Threshold**: 100.0 (Prevents runaway amplification of silence)

## 5. Signal-Processing Flow
1. Measure sample absolute value.
2. Update envelope using attack/release IIR smoothing.
3. If envelope < threshold, gain = 1.0.
4. Else, desired_gain = target_rms / envelope.
5. Clamp gain to [min_gain, max_gain].
6. Multiply sample by gain.
7. Hard clip safety to int16 range.

## 6. Test Signals
- **A. Low Level**: Sine wave amplitude 500
- **B. Normal Level**: Sine wave amplitude 5000
- **C. High Level**: Sine wave amplitude 25000
- **D. Very Low Level**: Amplitude 50 (below threshold)
- **E. Silence**: All zeros
- **F. Transition**: 500 -> 20000 amplitude jump

## 7. Metrics
- **RMS (Input/Output)**
- **Peak (Input/Output)**
- **Effective Gain** (RMS out / RMS in)
- **Clipping Percentage**

## 8. Acceptance Criteria
1. No NaN/Inf output.
2. Output within int16 [-32767, 32767].
3. Silence does not cause runaway amplification.
4. Low-level input receives useful amplification.
5. High-level input receives attenuation.
6. Normal-level input does not receive excessive gain changes.

## 9. Results
See `tests/results/audio_agc_phase2b1.json`. All synthetic numerical tests pass. Low signals are boosted by ~5.6x, high signals are attenuated by ~0.2x.

## 10. Baseline Comparison
When subjected to a transition signal (low -> high), the AGC normalizes the output RMS significantly better than the standard pipeline.

## 11. Limitations
- Attack/Release times are purely theoretical right now and require human listening tests to tune out "pumping" artifacts.
- Envelope detection uses peak tracing rather than true block RMS, which is computationally cheaper but slightly less accurate for complex waves.

## 12. Decision
**ACCEPT**. The AGC provides stable numerical containment of audio levels without runaway edge cases. It is suitable for addition to the pipeline.
