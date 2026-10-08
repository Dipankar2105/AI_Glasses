# AEC + DTD Hardware Readiness Integration Review (Phase 2B-4B)

## 1. Purpose
Evaluate the combined NLMS Acoustic Echo Canceller and Geigel Double-Talk Detector (AEC + DTD) for hardware deployment readiness. Establish the exact deterministic contract required to port this reference logic from Python to ESP32-S3 C/C++.

## 2. Combined AEC + DTD Architecture
- The pipeline processes inputs sequentially per sample (or vectorized per block in hardware).
- **DTD** evaluates the mic sample against the reference history to detect near-end energy.
- **NLMS** calculates the error and updates the filter weights *only* if the DTD has not flagged a hold state.

## 3. Input/Output Contract
- **Inputs**: 
  - `mic_buffer`: 16000 Hz, 16-bit signed PCM (represented as floats in Python).
  - `ref_buffer`: 16000 Hz, 16-bit signed PCM.
- **Output**: 
  - `out_buffer`: Echo-reduced microphone PCM frame of the same length as the shortest input buffer.
- **Frame Size**: The algorithm theoretically supports any frame size, but real-time hardware constraints typically dictate fixed sizes (e.g., 256 or 512 samples/frame).

## 4. Persistent State
The following state MUST persist between audio frames in the hardware implementation:
- **`weights`**: (STATIC ARRAY) Filter coefficients (length = 256).
- **`x_hist`**: (STATIC ARRAY) Reference signal delay line/history (length = 256).
- **`ref_history` (DTD)**: (STATIC ARRAY) Reference amplitude history for DTD max lookup (length = 256).
- **`hold_counter`**: (STATIC INT) Countdown timer for DTD hold state.
- **`dt_detected_frames`**: (STATIC INT) Debug/metrics counter.

*There are zero dynamically allocated variables required between frames.*

## 5. Memory Requirements
For a 256-tap filter using 32-bit floats:
- Weights: 256 * 4 = 1024 bytes
- Reference History: 256 * 4 = 1024 bytes
- DTD History: 256 * 4 = 1024 bytes
- **Algorithm State Total**: ~3072 bytes (3 KB) static RAM.
- **Frame Buffer Memory**: Dependent on DMA chunk size. Two input buffers and one output buffer required.

## 6. Computational Complexity
- **Per Sample**: 
  - DTD: Array push/pop, Max() calculation, 1 comparison.
  - NLMS: 256 multiplications (dot product), 256 additions, 1 division (normalization), 256 coefficient updates.
- **Per Frame (256 samples)**: ~131,072 multiply-accumulate (MAC) operations.
- **Hardware Requirement**: Must be compiled utilizing `esp-dsp` SIMD instructions (e.g., `dsps_dotprod_f32`, `dsps_add_f32`) to achieve real-time execution.

## 7. Timing Requirements
**CRITICAL**: The speaker reference PCM and the microphone capture PCM must maintain deterministic, rigid time-alignment. 
If the hardware routing introduces variable buffering latency between the speaker DMA interrupt and the microphone DMA interrupt, the echo path will shift constantly, causing catastrophic adaptation failure. Clock synchronization between the playback I2S and capture I2S domains is strictly required.

## 8. Reset Semantics
Currently, the Python reference implementation has **NO** explicit `reset()` method.
- **Hardware Adapter Requirement**: The C++ implementation must provide a method to zero the `weights`, `x_hist`, `ref_history`, and `hold_counter`. This is required to handle stream restarts, hardware glitches, or dynamic sample rate changes.

## 9. Hardware Adapter Requirements
The C++ ESP32 wrapper must:
1. Handle block-based framing matching the DMA interrupts.
2. Provide a circular buffer for `x_hist` and `ref_history` to replace Python's O(N) array shifts.
3. Manage the strict time synchronization between I2S TX and RX.
4. Convert 16-bit integers to float32, process, and convert back.

## 10. Python Reference vs Future ESP32 Implementation
The Python interface remains the absolute mathematical reference. The higher-level DSP orchestration logic must be able to switch between the Python simulation classes and Pybind/C++ hardware drivers transparently via duck-typing.

## 11. Known Limitations
- The Geigel DTD requires O(N) `max()` calculations per sample. A more optimized C++ approach will use a running-maximum algorithm to achieve O(1).
- Python floats are 64-bit; hardware floats are 32-bit. Minor numerical divergence is expected.

## 12. What Remains Before Physical Validation
The algorithm is entirely frozen mathematically. It requires a complete C++ ESP-DSP rewrite and physical acoustic calibration on the ESP32-S3 Sense hardware.
