# Acoustic Echo Cancellation (AEC) Experiment (Phase 2B-4)

## 1. Objective
Evaluate whether Acoustic Echo Cancellation (AEC) should be incorporated into the glasses audio pipeline to remove speaker feedback from the microphone signal.

## 2. AEC Mathematical Concept
AEC works by taking the known digital signal sent to the speaker (the *reference*) and adaptively estimating how that signal travels through the acoustic environment (the *echo path*) to reach the microphone. By subtracting the estimated echo from the raw microphone signal, the near-end speech remains.

## 3. NLMS Algorithm
We evaluated a **Normalized Least Mean Squares (NLMS)** algorithm.
- **Filter Length**: 256 samples (16ms echo tail).
- **Update Rule**: `w(n+1) = w(n) + [mu / (||x||^2 + epsilon)] * e(n) * x(n)`
- **Step Size (mu)**: Varies (0.5 for fast convergence, 0.05 for double-talk stability).

## 4. Signal Model
- `x[n]` = Speaker reference
- `mic[n]` = `near_end[n]` + `echo[n]` + `background_noise[n]`
- `echo[n]` = `delayed_and_attenuated(x[n])`
- `e[n]` = `mic[n]` - `estimated_echo[n]`

## 5. Test Scenarios
- **A. No Echo**: Mic contains only near-end.
- **B. Pure Echo**: Mic contains only echo.
- **C. Near-End + Echo**: Mic contains both.
- **D. Near-End + Echo + Noise**: Added random background noise.
- **E. Different Delays**: 10 samples vs 150 samples delay.
- **F. Echo Path Gain**: Attenuation > 1.0.
- **G. Double-Talk**: Simultaneous near-end speech and speaker reference.
- **H. Silence**: Zero input for both.

## 6. Metrics & Numerical Results
*(See `audio-aec-phase2b4.json` for full results)*
- **Numerical Stability**: 100% stable (No NaNs, No exploding coefficients).
- **Pure Echo Reduction**: Excellent ER (>20 dB) when no near-end speech is present to confuse the filter.
- **Double-Talk & Near-End Preservation**: When double-talk occurs, the adaptive filter's gradient is corrupted by the near-end speech (which acts as massive uncorrelated noise to the filter). A small step size (`mu=0.05`) preserves the near-end speech (NMSE < 0.5) but reduces the speed of echo cancellation.

## 7. Limitations
- **Double Talk Detection (DTD)**: A production AEC *must* freeze filter weight updates during near-end speech. This basic NLMS continues adapting, which causes divergence if `mu` is not aggressively lowered.
- **Echo Tail Length**: 256 samples covers 16ms of reverberation. A real room or a wearable device with internal casing resonance may require 1024+ samples (64ms+), increasing computational load drastically.
- **Computational Cost**: Pure Python evaluation is extremely slow. O(N) per sample means `256 * 16000 = ~4 million` MAC operations per second. 

## 8. Experimental Decision
The AEC accurately identifies and removes delayed synthetic echoes. **However, it is strictly software-only synthetic validation. No physical microphone, speaker, acoustic enclosure, or ESP32-S3 real-time performance has been validated.** 

**Recommendation:** AEC is fundamentally necessary for full-duplex communication (e.g., interrupting the AI while it speaks). The algorithm succeeds in theory, but integration into the `MIC -> NS -> VAD -> AGC` pipeline requires a hardware-accelerated (esp-dsp) C++ implementation with a robust DTD (Double Talk Detector) before it can be frozen as an active stage.
