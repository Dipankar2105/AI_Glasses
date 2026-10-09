# NextSight — Phase 10: Hardware Integration Readiness & Validation

**Status:** BLOCKED — HARDWARE UNAVAILABLE  
**Date:** 2026-10-09  
**Execution Note:** Physical integration cannot be completed because the physical glasses hardware (Seeed Studio XIAO ESP32-S3 Sense board, OV3660 camera module, MAX98357A amplifier, speaker, MPU-6050 IMU, touch sensor, and battery) is currently unavailable. Software interfaces, communication protocols, binary framing, and device simulators have been fully prepared and validated on the host.

---

## 1. Executive Summary & Verification Matrix

| Subsystem | Software Simulator / Contract Status | Physical Hardware Bench Status | Blocker |
| :--- | :--- | :--- | :--- |
| **ESP32-S3 Core Runtime** | Tested & Validated ([`runtime.py`](file:///c:/Users/Routewise/AI_Glasses/firmware/system/runtime.py)) | Pending Physical Flash | Hardware Unavailable |
| **Binary Protocol & CRC32** | Tested & Validated ([`framing.py`](file:///c:/Users/Routewise/AI_Glasses/backend/protocol/framing.py)) | Pending Network Link Test | Hardware Unavailable |
| **OV3660 Camera Capture** | Simulated JPEG Streaming Validated | Pending DVP Camera Test | Hardware Unavailable |
| **I2S Digital Microphone** | Simulated 16kHz PCM Stream Validated | Pending PDM Microphone Test | Hardware Unavailable |
| **MAX98357A I2S Speaker** | Modeled & Contract Validated | Pending I2S Audio Output Test | Hardware Unavailable |
| **MPU-6050 IMU** | Motion Processor & Gestures Validated | Pending I2C Bus Bringup | Hardware Unavailable |
| **Capacitive Touch Pad** | Debounce & State Machine Validated | Pending GPIO Touch ADC Test | Hardware Unavailable |
| **LiPo Battery & Thermals** | Hysteresis & Power Policy Validated | Pending ADC Fuel Gauge Test | Hardware Unavailable |

---

## 2. Host-Side Validation Delivered

1. **Protocol Framing Engine ([`backend/protocol/framing.py`](file:///c:/Users/Routewise/AI_Glasses/backend/protocol/framing.py)):**
   - 12-byte header + CRC32 verification.
   - Robust rejection of corrupted magic bytes, corrupted CRC32, and truncated payloads.
   - Sequence tracking with drop and duplicate detection.
2. **Device Simulator ([`backend/protocol/device_sim.py`](file:///c:/Users/Routewise/AI_Glasses/backend/protocol/device_sim.py)):**
   - Emulates ESP32-S3 boot, telemetry stream, audio PCM burst, and camera JPEG frame dispatch.
   - Validates bounded queue backpressure and drop-oldest buffer behavior.
3. **Firmware System Runtime ([`firmware/system/runtime.py`](file:///c:/Users/Routewise/AI_Glasses/firmware/system/runtime.py)):**
   - Supports typed system states (`BOOT`, `INITIALIZING`, `READY`, `STREAMING`, `DEGRADED`, `SHUTDOWN`).

---

## 3. Physical Hardware Bringup & Acceptance Checklist (For Future Hardware Arrival)

When physical hardware becomes available, execute the following step-by-step checklist:

### Step 1: ESP-IDF Toolchain & Bootloader Flash
- Toolchain: ESP-IDF v5.1+ / PlatformIO `espressif32`.
- Connect XIAO ESP32-S3 Sense via USB-C (COM port detected).
- Command: `idf.py build flash monitor -p COMx -b 921600`
- Acceptance: Device boots cleanly into `READY` state without watchdog resets or PSRAM allocation failures.

### Step 2: I2C Bus & MPU-6050 Bringup
- Probe I2C bus on GPIO 5 (SDA) / GPIO 6 (SCL) at 400kHz.
- Verify MPU-6050 WHO_AM_I register returns `0x68`.
- Record stationary noise offset for zero-rate gyro calibration.

### Step 3: OV3660 Camera Initialization
- Verify camera SCCB communication on GPIO 39/40 and 20MHz XCLK on GPIO 10.
- Capture test frame (QVGA 320x240 RGB565 / JPEG).
- Verify DMA buffer alignment in external 8MB PSRAM.

### Step 4: I2S Microphone Audio Ingest
- Stream 16kHz 16-bit PCM from PDM mic (GPIO 41/42).
- Record 5 seconds of audio and verify SNR without clipping.

### Step 5: MAX98357A I2S Audio Playback
- Output 16kHz WAV test tone to I2S DAC (GPIO 1/2/3).
- Measure current draw and audio clarity.

### Step 6: End-to-End WebSocket Streaming
- Connect ESP32-S3 Wi-Fi to local router.
- Open WebSocket binary stream to Backend endpoint `/ws/stream`.
- Transmit interleaved audio PCM chunks and camera frames.
- Verify end-to-end latency < 150ms and 0 CRC errors.
