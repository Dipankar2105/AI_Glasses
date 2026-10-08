# AEC Double-Talk Detector Experiment (Phase 2B-4A)

## 1. Objective
Evaluate whether a deterministic Double-Talk Detector (DTD) is required to safely manage Acoustic Echo Cancellation (AEC) during simultaneous speaker playback and near-end speech.

## 2. Algorithm & Implementation
We implemented a **Geigel DTD** encapsulated purely in a subclass (`NLMSAECWithDTD`) without altering the existing frozen `NLMSAEC` logic.
- **Mechanism**: The Geigel algorithm compares the current microphone amplitude to the maximum amplitude of the reference signal (the echo tail history).
- **Threshold**: If `|mic| >= 0.5 * max(|ref_history|)`, near-end speech is declared present.
- **Hold Time**: The adaptation is frozen for 800 samples (50ms) to ensure stability across speech valleys.

## 3. Results Overview
*(See `audio-aec-dtd-phase2b4a.json` for precise metrics)*

1. **Does DTD materially improve double-talk safety?**
   **YES**. By halting the NLMS weight adaptation during double-talk, the DTD prevents the filter from mis-adapting and distorting the near-end speech. The Normalized Mean Square Error (NMSE) of the near-end signal during double talk was substantially lower (cleaner) than the baseline AEC.

2. **Does DTD reduce echo cancellation quality?**
   **NO**. During pure echo scenarios, the Geigel detector correctly evaluates to false, and the NLMS filter converges identically to the baseline AEC.

3. **Does DTD introduce unacceptable false detections?**
   **NO**. With a 0.5 threshold, it effectively discriminates between standard echo paths (which are mathematically attenuated below 0.5) and near-end speech additions.

4. **Is DTD computationally simple enough for future ESP32 implementation?**
   **YES**. The Geigel algorithm requires finding a maximum over an array and performing one multiplication per sample. This is extremely lightweight and heavily optimized in C++.

5. **What persistent state would be required on hardware?**
   - The same weight array and history buffer from the baseline AEC.
   - An integer `hold_counter`.
   - The DTD does not introduce any complex memory requirements.

## 4. Unresolved Limitations & Final Decision
**Result: CONDITIONAL ACCEPT**

The Geigel DTD fundamentally protects the adaptive filter during double-talk and represents the minimum required safety logic for a production AEC. 

**What remains unresolved before AEC can be frozen?**
- Computational complexity: Pure Python NLMS+DTD loop evaluating 16kHz audio sample-by-sample takes significant runtime on a standard CPU.
- Hardware Translation: The entire block must be transposed to ESP-DSP hardware instructions before it can actually be merged into the `MIC -> NS -> VAD -> AGC` pipeline.

*The DTD is accepted as a valid algorithm but remains experimental alongside AEC, pending hardware realization.*
