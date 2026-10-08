# Audio VAD Evaluation (Phase 2B-2)

## 1. Objective
To evaluate whether a deterministic energy-based Voice Activity Detection (VAD) stage is useful for the AI Glasses audio pipeline.

## 2. Why VAD is Needed
Transmitting silence or background noise over Wi-Fi drains battery and wastes AI processing cycles. A VAD gates the transmission, saving power and bandwidth.

## 3. VAD Architecture
- **Type**: Lightweight deterministic energy-based VAD.
- **Dependency**: Zero external ML libraries. Strictly math-based.
- **Output**: Generates `speech_active` boolean states per frame.

## 4. Signal Processing Logic
1. **Frame processing**: Chunks audio into 10ms frames.
2. **Energy calculation**: Computes RMS energy per frame.
3. **Threshold logic**: Compares RMS against `energy_threshold`.
4. **Attack/Release**: Requires consecutive frames to change state.
5. **Hangover**: Maintains `speech_active` across short pauses (e.g., stops fragmentation).

## 5. Synthetic Test Signals & Ground Truth
- Pure Silence -> Ground truth: 0 frames active.
- Low-level Noise -> Ground truth: 0 frames active.
- Speech-like Signal -> Ground truth: Bursts of activity matching envelope.
- Loud Noise -> Ground truth: Falsely triggers (limitation of energy VAD).

## 6. Baseline Comparison
When processing speech with short pauses, a VAD *without* hangover fragments the speech into many small chunks. The implemented VAD *with* hangover successfully bridged the pauses, demonstrating the absolute necessity of temporal smoothing.

## 7. AGC Interaction Experiment
- **Setup**: Passed low-level noise through AGC, then into VAD.
- **Result**: Because AGC amplifies low-level noise to the target RMS, the VAD *falsely triggered* constantly.
- **Insight**: If AGC and VAD are used together, VAD must either operate *before* AGC, or the VAD threshold must be dynamically linked to the AGC gain state.

## 8. Limitations
- **Loud Noise False Positives**: Energy-based VAD cannot distinguish between a loud voice and a loud handclap.
- **AGC Conflict**: Requires architectural routing decisions (VAD before AGC).

## 9. Decision
**CONDITIONAL ACCEPT**. 
The temporal energy-based VAD works excellently and bridges pauses well. However, its interaction with the previously accepted AGC component requires the DSP pipeline to explicitly order them: `Input -> VAD -> AGC`, rather than `AGC -> VAD`. It is accepted as a software checkpoint, but integration logic must handle the routing.
*Note: Validation is software/synthetic only; physical microphone validation has not yet been performed.*
