# Audio VAD / AGC Ordering Resolution (Phase 2B-2R)

## 1. Problem Discovered in Phase 2B-2
When chaining AGC followed by VAD (Architecture A), the AGC aggressively amplifies low-level background noise toward the target RMS. This causes the VAD to falsely trigger on pure noise, as the noise energy crosses the VAD threshold post-amplification.

## 2. Why Ordering Matters
- **Architecture A (MIC -> AGC -> VAD)**: VAD analyzes the *normalized* signal. 
  - *Pro*: Can detect very quiet speech because AGC boosts it.
  - *Con*: Amplifies the noise floor, causing massive false positives.
- **Architecture B (MIC -> VAD -> AGC)**: VAD analyzes the *raw* signal.
  - *Pro*: Background noise stays below threshold, preventing false positives.
  - *Con*: Very quiet speech might fall below the raw VAD threshold and be gated before the AGC can rescue it.

## 3. Architecture A
AGC -> VAD. Tested and shown to fail on low-level noise (100% false positive rate in our 300 RMS test).

## 4. Architecture B
VAD -> AGC. Tested and shown to safely reject low-level noise (0% false positive rate). However, struggles with low-level speech.

## 5. Test Methodology
Both architectures were subjected to identical, deterministic synthetic signals. The existing, validated AGC and VAD modules from Phase 2A/2B-1 were used as read-only components. No tuning was allowed.

## 6. Test Signals
1. True Silence
2. Low-Level Background Noise (300 RMS)
3. Clear Speech-Like Signal
4. Speech + Background Noise
5. Speech with Pauses
6. Low-Level Speech (Peak amplitude ~400, RMS below threshold)
7. Loud Non-Speech Noise

## 7. Quantitative Results
See `tests/results/audio-vad-agc-ordering.json`. 
- **Noise Test**: Arch A = 100% False Positive. Arch B = 0% False Positive.
- **Low-Level Speech Test**: Arch A = 100% Detected. Arch B = 0% Detected.

## 8. Comparison
- **Architecture A** continuously classifies amplified environmental noise as speech. This is disastrous for a voice-controlled wearable.
- **Architecture B** protects the system from background noise amplification but completely drops quiet speech.

## 9. Limitations
Neither architecture works flawlessly with fixed thresholds. True resolution likely requires a dynamic threshold where VAD estimates the noise floor independently of the AGC, or VAD drives the AGC state.

## 10. Recommended Ordering
**SELECTED:**
MIC -> VAD -> AGC

**REJECTED AS DEFAULT:**
MIC -> AGC -> VAD

**Reason:**
AGC -> VAD produced 100% false activation on the tested 300 RMS background-noise signal, while VAD -> AGC produced 0% false activation.

## 11. Known Limitations
- VAD -> AGC misses low-level speech below the VAD threshold.
- This is a known limitation.
- It must NOT be hidden.
- It must NOT be artificially tuned away in this checkpoint.
- Physical microphone validation has not yet occurred.

## 12. Conclusion
The VAD module itself operates correctly according to its energy-based design. The VAD logic can be frozen with MIC -> VAD -> AGC as the default architecture for safety against false positives.
