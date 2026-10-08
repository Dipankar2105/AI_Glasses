# Audio DSP Architecture

## 1. Audio Data Contract
- **Sample Rate**: 16000 Hz (PROVISIONAL)
- **Bit Depth**: 16-bit
- **Channel Count**: 1 (Mono)
- **Representation**: Signed integer (int16)
- **Format**: PCM AudioBuffer list array

## 2. DSP Pipeline
The pipeline uses a sequential block-processing design. Data enters an `AudioBuffer`, flows through sequential, toggleable `DSPStage` subclasses, and outputs an updated `AudioBuffer`.

## 3. Implemented Stages
- **DCBlocker**: First-order IIR high-pass filter that removes DC offset.
- **HighPassFilter**: Standard RC high-pass filter to attenuate sub-bass/wind noise.
- **Gain**: Linear normalization multiplier.
- **Limiter**: Hard clipping to enforce ceiling constraints.

## 4. Parameters
- **DC Blocker**: `r=0.995`
- **High Pass**: `cutoff_freq=80.0` Hz, `sample_rate=16000.0`
- **Gain**: `gain_factor` (linear scalar)
- **Limiter**: `threshold=32767.0` (max int16)

## 5. Test Signals
Synthetic PCM signals are deterministically generated for testing:
- Clean sine wave
- Sine wave with DC offset
- Multi-frequency signal
- Amplified sine exceeding threshold

## 6. Metrics
Available numerical metrics for evaluation:
- RMS (Root Mean Square)
- Peak amplitude
- Crest factor
- DC offset
- Clipping percentage

## 7. Unit Tests
- Individual tests verify the numerical correctness of DC removal, HPF attenuation, gain application, and clipping behavior.
- A full End-to-End Pipeline test runs a heavily distorted signal through all components to verify integration.
- *Status:* **SOFTWARE VERIFIED**

## 8. Known Limitations
- The current implementation is a reference software model in Python intended for architectural validation.
- End-to-end real-time latency is currently unmeasured as it operates in non-real-time Python.

## 9. Hardware-Dependent Items (HARDWARE NOT VERIFIED)
- I2S hardware integration
- Onboard microphone behavior and true noise floor
- True required sample rate

## 10. Items Intentionally Deferred
- Automatic Gain Control (AGC)
- Voice Activity Detection (VAD)
- Noise suppression
- Acoustic Echo Cancellation (AEC)
- C++ Embedded Port (Awaiting complete pipeline validation before cross-compilation)
