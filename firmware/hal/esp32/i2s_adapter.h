#ifndef I2S_ADAPTER_H
#define I2S_ADAPTER_H

#include <stdint.h>
#include <stdbool.h>

/**
 * Audio HAL Boundary matching Phase 3.1 Contract.
 * These types match the conceptual Python HAL.
 */
typedef enum {
    HAL_OK = 0,
    HAL_NOT_INITIALIZED = 1,
    HAL_DEVICE_UNAVAILABLE = 2,
    HAL_INVALID_CONFIGURATION = 3,
    HAL_BUFFER_UNAVAILABLE = 4,
    HAL_OVERRUN = 5,
    HAL_UNDERRUN = 6,
    HAL_TIMEOUT = 7,
    HAL_HARDWARE_ERROR = 8,
    HAL_INVALID_STATE_TRANSITION = 9
} hal_err_t;

typedef enum {
    HAL_STATE_UNINITIALIZED = 0,
    HAL_STATE_INITIALIZED,
    HAL_STATE_RUNNING,
    HAL_STATE_STOPPED,
    HAL_STATE_DEINITIALIZED
} hal_state_t;

typedef struct {
    float* pcm_data;     // float representation for DSP (converted from int16_t)
    uint32_t sample_count;
    uint32_t timestamp;  // ms
    uint32_t seq_num;
} audio_capture_frame_t;

typedef struct {
    float* pcm_data;     // float representation from DSP (to be converted to int16_t)
    uint32_t sample_count;
    uint32_t timestamp;  // ms
    uint32_t seq_num;
} audio_playback_frame_t;

/**
 * ESP32-S3 Audio Adapter API
 */
hal_err_t esp32_audio_adapter_init(void);
hal_err_t esp32_audio_adapter_start(void);
hal_err_t esp32_audio_adapter_stop(void);
hal_err_t esp32_audio_adapter_deinit(void);

hal_err_t esp32_audio_adapter_read_capture(audio_capture_frame_t* out_frame);
hal_err_t esp32_audio_adapter_write_playback(const audio_playback_frame_t* in_frame);
hal_err_t esp32_audio_adapter_read_reference(audio_playback_frame_t* out_frame);

#endif // I2S_ADAPTER_H
