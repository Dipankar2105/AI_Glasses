# Audio Noise Suppression Evaluation (Phase 2B-3)

## 1. Objective
To evaluate a lightweight software noise-suppression component for the AI Glasses audio pipeline. The goal is to reduce background noise while preserving speech content, without requiring heavy ML libraries.

## 2. Selected Algorithm
**Spectral Subtraction**. A classical frequency-domain approach.
- Converts framed audio to frequency domain via FFT.
- Estimates the noise magnitude spectrum during an initial "noise-only" period.
- Subtracts the noise magnitude from subsequent frames.
- Converts back via IFFT and overlap-add.

## 3. Algorithm Architecture
- **Window**: Hann window.
- **Frame Size**: 256 samples (16ms at 16kHz).
- **Overlap**: 50% (Hop size 128).
- **Noise Estimation**: Averages the first 10 frames (160ms) assuming they contain only background noise.

## 4. Synthetic Signals & Ground Truth
Deterministic signals were used:
- Clean Speech-Like signal (multi-frequency, amplitude-modulated).
- Deterministic background noise added at various amplitudes (Low, Medium, Strong).
- Initial 0.2s of the signal contains only noise to allow the algorithm to build a noise profile.

## 5. SNR Results
*(See JSON for exact metrics)*
- The algorithm successfully improves SNR on Low and Medium noise tests.
- Pure noise is visibly attenuated after the profile is learned.

## 6. Speech Preservation Results
- The NMSE (Normalized Mean Square Error) for clean speech is extremely low, meaning the algorithm does not aggressively damage the signal when no noise is present.
- However, frequency-domain spectral subtraction introduces phase artifacts ("musical noise") which are mathematically present in the residual error.

## 7. Interaction Experiment (Conceptual)
Current Baseline: MIC -> VAD -> AGC.
If Noise Suppression (NS) is added: MIC -> NS -> VAD -> AGC.
Placing NS first allows the VAD to operate on a cleaner signal, improving VAD accuracy and reducing false positives even further.

## 8. Computational Cost & Latency
- **Cost**: A pure-Python FFT is slow, taking significant time per frame in this test harness. However, ESP32-S3 contains hardware DSP instructions (esp-dsp) that can execute a 256-point complex FFT in ~15 microseconds. Python timing does not constitute ESP32 performance validation.
- **Latency**: Algorithmic latency is one frame (16ms) plus overlap-add buffering, which is highly suitable for real-time streaming.

## 9. Limitations
- **Estimation Assumption**: Real-world usage rarely guarantees 160ms of pure background noise at startup. A dynamic noise tracker (e.g., Minima Controlled Recursive Averaging) is required for production.
- **Musical Noise**: Standard spectral subtraction creates musical artifacts. Over-subtraction parameters help, but distort speech.

## 10. Final Decision
**CONDITIONAL ACCEPT**. The mathematical foundation of Spectral Subtraction effectively reduces deterministic noise while preserving speech energy. However, it requires a C++ ESP-DSP implementation to be viable for real-time use, and it requires a dynamic noise tracker rather than a fixed 10-frame assumption. It is accepted as a proof-of-concept software checkpoint.

*Validation is software/synthetic only.*
