# Audio Noise Suppression Architecture Resolution (Phase 2B-3R)

## 1. Objective
Compare three experiment-only pipelines to determine whether placing Noise Suppression (NS) before Voice Activity Detection (VAD) solves the low-level speech problem discovered in Phase 2B-2R.

## 2. Architectures Tested
- **A — CURRENT BASELINE**: MIC -> VAD -> AGC
- **B — PRIMARY CANDIDATE**: MIC -> NS -> VAD -> AGC
- **C — SECONDARY CANDIDATE**: MIC -> VAD -> NS -> AGC

*Note: The existing DSP components were used as Read-Only. No internal thresholds or algorithms were modified.*

## 3. Test Signals
All architectures were subjected to identical deterministic signals:
1. Silence
2. 300 RMS background noise
3. Clear speech-like signal
4. Low-level speech (~400 amplitude / RMS ~282)
5. Low-level speech + 300 RMS noise
6. Medium speech + noise
7. Speech + short pauses + noise
8. Loud non-speech noise

## 4. Quantitative Results & Comparison
*(See `tests/results/audio-noise-suppression-architecture-phase2b3r.json`)*

**Does NS-before-VAD materially improve quiet-speech detection while keeping background-noise false activation low?**
**PARTIALLY**. 

- **Background Noise**: Architecture B (NS first) mathematically attenuates background noise. Like Architecture A, it boasts a 0% false positive rate on the 300 RMS noise test.
- **Low-Level Speech**: Because the fixed VAD threshold remains 500, and NS actually removes some residual energy from the low-level signal (which was already at RMS 282), Architecture B *fails* to rescue low-level speech. The signal remains gated. 
- **Architecture C**: Operates similarly to Architecture A regarding VAD (since VAD receives raw MIC data), meaning low-level speech is still missed.

## 5. Recommended Architecture
**B (MIC -> NS -> VAD -> AGC)**. 

While it does not magically fix the low-level speech gating issue without dynamic thresholding, placing NS before VAD ensures the VAD is always analyzing the cleanest possible signal, providing a mathematically superior noise-floor defense.

## 6. Limitations
- We cannot detect speech whose RMS falls below the fixed 500 threshold unless the threshold is made dynamic or lowered. 
- Physical microphone validation has not yet been performed.
