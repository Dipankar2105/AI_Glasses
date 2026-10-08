# Acoustic Echo Cancellation (AEC) Hardware Readiness Review (Phase 2B-4R)

## 1. Objective
To evaluate the current software AEC implementation (`aec.py`) for hardware portability and document the future contract required when transferring the DSP logic to the physical XIAO ESP32-S3 Sense.

## 2. AEC Interface Evaluation
The current AEC is encapsulated within the `NLMSAEC` class and exposes a clean interface:
```python
def process_aec(self, mic_buffer: AudioBuffer, ref_buffer: AudioBuffer) -> AudioBuffer:
```
- **Hardware-Portable**: Yes. It accepts two identical PCM buffers (microphone input and speaker reference) and returns one processed output buffer.
- **Abstract**: It has zero coupling to file I/O, hardware APIs, or simulation-specific state. The higher-level pipeline can pass in hardware DMA buffers without modifying the AEC concept.

## 3. Hardware Contract
When ported to C++/ESP-DSP, the hardware layer must provide the following:
- **Sample Rate**: 16000 Hz.
- **Sample Format**: 16-bit signed PCM (int16).
- **Channel**: Mono.
- **Processing**: Frame-based block processing.
- **Time Alignment**: The ESP32 I2S output (speaker playback) MUST be captured simultaneously with the I2S input (microphone) to ensure the reference buffer correlates tightly with the acoustic echo. If hardware routing introduces unknown variable delays between speaker output and mic capture, the AEC will fail to converge.

## 4. Signal Path (Future)
1. **Speaker Playback** -> I2S Output -> Copied to **AEC Reference Buffer**
2. **Microphone Capture** -> I2S Input -> Copied to **AEC Mic Buffer**
3. **AEC Execution** -> Processes Mic and Reference -> Generates Echo-Reduced Mic Buffer
4. **Higher Pipeline** -> Passes through Noise Suppression, VAD, and AGC

## 5. Portability Risks & Memory Review
The current `aec.py` relies on Python conveniences that must be replaced in embedded C:

- **Dynamic Allocations & Memory Shifts**: 
  - *Python*: `self.x_hist.insert(0, x)` and `.pop()` performs O(N) dynamic shifting.
  - *C++ Fix*: Must be replaced with a statically allocated circular buffer (ring buffer) using a modulo index pointer to achieve O(1) time without heap allocations.
- **Floating Point Assumptions**: 
  - *Python*: Uses 64-bit floats (`float`) for filter weights and calculations.
  - *C++ Fix*: ESP32-S3 has hardware single-precision (32-bit) FPU. The algorithm must be profiled to ensure 32-bit floats provide sufficient numerical stability (especially regarding the NLMS denominator regularization parameter). Alternatively, Fixed-Point math can be used.
- **Computational Complexity**: 
  - *Python*: Standard loops `sum(w * x)`. 
  - *C++ Fix*: Must be replaced by `esp-dsp` vector/SIMD instructions (e.g., `dsps_dotprod_f32`) to achieve real-time performance within a frame budget.
- **State Memory Requirements**: 
  - The filter requires two fixed arrays of size `filter_length`: the weights array and the history buffer. For a 256-tap filter using 32-bit floats, this requires `256 * 4 * 2 = 2048 bytes` (2 KB) of static RAM, which is completely trivial for the ESP32-S3.

## 6. Conclusion
The current `aec.py` interface is highly portable and ready for hardware abstraction. No interface modifications were required. It can remain indefinitely as the Python reference implementation for validating the mathematical constraints of the pipeline.
