# Phase 3.2: ESP32-S3 Audio HAL / I2S DMA Hardware Adapter

## 1. Purpose
Define and isolate the exact ESP-IDF v6.1 hardware-specific implementation boundary that fulfills the frozen Phase 3.1 `AudioHAL` contract. This ensures the frozen Phase 2 DSP code runs unmodified on the physical ESP32-S3 hardware.

## 2. Architecture
```
    XIAO ESP32-S3
          │
          ├── I2S RX (Hardware DMA)
          │       │
          │       ▼
          │   int16_t PCM [-32768, 32767]
          │       │
          │       ▼
          │   esp32_audio_adapter_read_capture() (C implementation)
          │       │
          │       ▼
          │   AudioHAL contract
          │       │
          │       ▼
          │   float32_t DSP frames
          │       │
          │       ▼
          │   FROZEN PHASE 2 DSP
```

## 3. Generic HAL Boundary vs ESP32 Boundary
- **Generic HAL**: Python classes (`AudioCaptureFrame`, `AudioPlaybackFrame`) and Enums (`HALState`, `HALError`).
- **ESP32 Adapter**: C header `firmware/hal/esp32/i2s_adapter.h` containing hardware-specific initialization, ESP-IDF DMA configuration (deferred), and int16<->float conversion.

## 4. ESP-IDF Target Environment
- **Target**: esp32s3
- **Version**: v6.1 
- **APIs Verified via Header Inspection**:
    - `i2s_chan_handle_t`
    - `i2s_channel_init_std_mode`
    - `i2s_channel_enable`
    - `i2s_channel_disable`
    - `i2s_channel_read`
    - `i2s_channel_write`

## 5. I2S Configuration
- **Sample Rate**: 16000 Hz.
- **Bit Depth**: 16-bit.
- **Channel Format**: Mono.
- **Role**: Master.
- **DMA Strategy**: Handled by the standard I2S driver (`dma_desc_num = 6`, `dma_frame_num = 256`).
- **GPIO/PIN VALIDATION**: DEFERRED
  - Reason: Physical board-level pin mapping has not yet been validated in this phase.

## 6. int16 ↔ float Conversion
The ESP32 adapter *owns* the data conversion bridging the hardware and DSP domains:
- **RX**: Casts `int16_t` hardware samples to `float` directly.
- **TX**: Casts `float` to `int16_t` applying strict hard-clipping saturation boundaries (`> 32767` = 32767, `< -32768` = -32768).

## 7. Frame Metadata
- **Frame Size**: Dependent on DMA chunk size (e.g., 256 samples).
- **Sequence Numbers**: Monotonically increase per buffer read. 
- **Timestamps**: Uses `esp_timer_get_time() / 1000` (ms).

## 8. AEC Reference Handling
The adapter will push outgoing playback frames into an internal queue for AEC retrieval.
*Logical association*: Maintained perfectly via `seq_num`.
*Physical synchronization*: NOT VALIDATED.

## 9. Lifecycle & Error Handling
- Respects `HALState` strict transitions. Invalid operations explicitly return `HAL_INVALID_STATE_TRANSITION` or `HAL_NOT_INITIALIZED`.
- ESP-IDF specific errors (e.g., `ESP_ERR_TIMEOUT`) are explicitly mapped to generic HAL errors (e.g., `HAL_BUFFER_UNAVAILABLE`). No ESP-IDF types leak into the upper architecture.

## 10. Queue Ownership & Underflow/Overflow
- The ESP-IDF I2S driver owns the hardware DMA queues. 
- If `i2s_channel_read` returns timeout, the adapter yields `HAL_BUFFER_UNAVAILABLE`.
- If `i2s_channel_write` returns timeout, the adapter yields `HAL_OVERRUN`.

## 11. Validation Status
- **HOST/SOFTWARE VALIDATION**: PASS
- **PHASE 2 REGRESSION**: PASS
- **PHASE 3.1 REGRESSION**: PASS
- **ESP-IDF COMPILATION**: NOT VERIFIED
  - Reason: The current host environment does not have an installed ESP-IDF/idf.py toolchain.
- **PHYSICAL HARDWARE**: NOT VALIDATED
  - Reason: XIAO ESP32-S3 Sense is not currently connected.
