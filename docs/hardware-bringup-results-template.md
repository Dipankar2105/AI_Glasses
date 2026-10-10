# NextSight Hardware Bring-Up — Execution Evidence Template

> **INSTRUCTIONS FOR TEST OPERATOR:**  
> Use this template to record actual, un-fabricated measurements and diagnostic logs during physical hardware bring-up of the NextSight AI Smart Glasses.  
> Do **NOT** pre-populate physical sensor readings, voltages, or current draw before bench testing.

---

## 1. Test Session Metadata

- **Date of Execution:** `YYYY-MM-DD`
- **Lead Test Operator:** `[Name / Engineering Role]`
- **Repository Commit / Checkpoint:** `[Git SHA-1]`
- **Target Microcontroller:** Seeed Studio XIAO ESP32-S3 Sense
- **Board Serial / Unit Identifier:** `[e.g. Unit #01 / Prototype Rev A]`
- **ESP-IDF Toolchain Version:** `[e.g. ESP-IDF v5.1.2 / Python 3.10]`
- **Host Operating System:** `[e.g. Windows 11 / Ubuntu 22.04]`

---

## 2. Hardware & Power Supply Identification

| Component / Subsystem | Part Number / Specification | Hardware ID / Address | Visual Inspection (PASS/FAIL) |
| :--- | :--- | :--- | :--- |
| **Main SoC Board** | Seeed Studio XIAO ESP32-S3 Sense | Octal PSRAM (8MB) | `[ PENDING BENCH ]` |
| **Camera Module** | OV3660 3MP DVP Sensor | SCCB Addr `0x3C`/`0x78` | `[ PENDING BENCH ]` |
| **Microphone** | Onboard PDM / External INMP441 | GPIO 41/42 or I2S | `[ PENDING BENCH ]` |
| **I2S Amplifier** | MAX98357A Mono Class D | Breakout Header (D0-D2) | `[ PENDING BENCH ]` |
| **IMU Sensor** | MPU-6050 6-Axis MotionTracking | I2C Addr `0x68` (AD0=0) | `[ PENDING BENCH ]` |
| **Capacitive Touch** | Custom Copper Temple Pad | GPIO 4 (Touch4) | `[ PENDING BENCH ]` |
| **Battery Unit** | 1S 3.7V 500mAh LiPo Cell | JST 1.25 / BAT+ Pads | `[ PENDING BENCH ]` |
| **Bench Power Supply** | `[Model & Calibration Date]` | Set: `3.70V`, Limit: `500mA` | `[ PENDING BENCH ]` |

---

## 3. Firmware Build & Flashing Records

```text
[Paste Build Output & Flashing Terminal Log Here]
Example Command: idf.py -p COM<X> flash monitor
Build Artifact SHA-256: [e.g. xiaozhi.bin hash]
Flash Exit Code: [0 = Success]
```

---

## 4. Stage-by-Stage Verification Evidence

### Stage 1: Board Enumeration & Serial Boot
- **Assigned COM Port:** `COM<X>`
- **USB Enumeration:** `[ PASS / FAIL ]`
- **Bootloader Banner Captured:**
  ```text
  [Paste Boot Log Here]
  ```
- **PSRAM Initialized Size:** `_____ MB`
- **Free Heap at Boot:** `_____ KB`

---

### Stage 2: MPU-6050 IMU & Capacitive Touch Evidence
- **I2C WHO_AM_I Register Value:** `0x____` (Expected: `0x68`)
- **Stationary Accelerometer Readings ($g$):**
  - $X$: `_____ g` (Expected: $\approx 0.00$)
  - $Y$: `_____ g` (Expected: $\approx 0.00$)
  - $Z$: `_____ g` (Expected: $\approx +1.00$)
- **Stationary Gyroscope Rates ($^\circ/\text{s}$):**
  - $\omega_x$: `_____ deg/s`
  - $\omega_y$: `_____ deg/s`
  - $\omega_z$: `_____ deg/s`
- **Head Gesture Detection Status:**
  - Head Nod: `[ PASS / FAIL / NOT RUN ]`
  - Head Shake: `[ PASS / FAIL / NOT RUN ]`
  - Head Tilt: `[ PASS / FAIL / NOT RUN ]`
- **Touch Pad Interaction Status:**
  - Single Tap: `[ PASS / FAIL / NOT RUN ]`
  - Double Tap: `[ PASS / FAIL / NOT RUN ]`
  - Long Press: `[ PASS / FAIL / NOT RUN ]`

---

### Stage 3: OV3660 Camera Capture Evidence
- **Camera Sensor Detection:** `[ PASS / FAIL ]` (Expected PID: `0x3660`)
- **Frame Resolution Tested:** `[ QVGA 320x240 / VGA 640x480 ]`
- **Frame Buffer Allocation Location:** `[ External PSRAM / Internal SRAM ]`
- **Capture Latency Measured:** `_____ ms`
- **Host Image Quality Assessment:**
  - Brightness Score: `_____`
  - Sharpness Score: `_____`
  - Visual Artifacts Observed: `[ NONE / TEARING / GREEN LINES / NOISE ]`

---

### Stage 4: Microphone PCM Capture & DSP Audio Evidence
- **Mic Configuration:** `[ Onboard PDM / External I2S ]`
- **Captured Sample Rate / Width:** `16,000 Hz / 16-bit Mono`
- **Ambient Noise Floor:** `_____ dBFS` (Target: $< -45\text{ dBFS}$)
- **Spoken Voice SNR Improvement:** `_____ dB` (Target: $> 10\text{ dB}$)
- **VAD Trigger Reliability:** `[ PASS / FAIL ]` (Accurate speech bounding without clipping)
- **Audio Glitches / Overruns:** `[ ZERO / DETECTED ]`

---

### Stage 5: Speaker Playback & MAX98357A Evidence
- **Test Sine Wave Output (440 Hz @ -12 dBFS):** `[ AUDIBLE & CLEAN / DISTORTED / SILENT ]`
- **TTS Sample Playback Clarity:** `[ HIGH INTELLIGIBILITY / MUFFLED / NOISY ]`
- **AEC Double-Talk Attenuation (ERLE):** `_____ dB`
- **Pop / Click Noise on Start/Stop:** `[ NONE / DETECTED ]`

---

### Stage 6: Device-to-Host Transport & Framing Evidence
- **Transport Medium:** `[ Wi-Fi WebSocket / USB-CDC Serial ]`
- **100-Packet Telemetry Stream CRC Failure Count:** `_____ / 100` (Target: 0)
- **Network Reconnection Recovery Time:** `_____ ms` (Target: $< 3000\text{ ms}$)
- **Bounded Queue Eviction Behavior during Stalls:** `[ PASS / FAIL ]`

---

### Stage 7: Power & Thermal Safety Bench Measurements

| Operating State | Target Modeled Current | Measured Quiescent Current | Measured Supply Voltage | Measured Die/Skin Temp | Status (PASS/FAIL) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IDLE (Light Sleep)** | $< 10\text{ mA}$ | `_____ mA` | `3.70 V` | `_____ °C` | `[ PENDING ]` |
| **Active Audio Streaming** | $< 60\text{ mA}$ | `_____ mA` | `3.70 V` | `_____ °C` | `[ PENDING ]` |
| **Camera Capture Burst** | $< 120\text{ mA}$ | `_____ mA` | `3.70 V` | `_____ °C` | `[ PENDING ]` |
| **Full Vision + Audio + Spk** | $< 250\text{ mA}$ | `_____ mA` | `3.70 V` | `_____ °C` | `[ PENDING ]` |
| **Low-Battery Gate (3.55V)** | State $\rightarrow$ `LOW_POWER` | `_____ mA` | `3.55 V` | `_____ °C` | `[ PENDING ]` |
| **Hot Throttle Gate (56°C)** | State $\rightarrow$ `HOT_THROTTLED`| `_____ mA` | `3.70 V` | `56.0 °C` | `[ PENDING ]` |

---

### Stage 8: Integrated User Interaction Acceptance
- **Double Tap $\rightarrow$ Capture $\rightarrow$ AI Inference $\rightarrow$ Voice Playback:** `[ PASS / FAIL / NOT RUN ]`
- **Total Measured End-to-End Latency:** `_____ ms`
- **User Experience Assessment:** `[ ACCEPTABLE / HIGH LATENCY / UNSTABLE ]`

---

## 5. Bring-Up Summary & Decision Matrix

| Stage | Name | Target Outcome | Actual Outcome | Decision |
| :---: | :--- | :--- | :--- | :---: |
| **Stage 0** | Preflight & Host Environment | 219 tests pass, valid binary | 219 passed, binary ready | **PASSED** |
| **Stage 1** | Board Enumeration & Boot | Clean boot, 8MB PSRAM | Pending hardware | `NOT RUN` |
| **Stage 2** | IMU & Capacitive Touch | Valid 100Hz IMU, debounced touch | Pending hardware | `NOT RUN` |
| **Stage 3** | OV3660 Camera Capture | Valid JPEG frame in PSRAM | Pending hardware | `NOT RUN` |
| **Stage 4** | Microphone & DSP Audio | Clean PCM, SNR $>10\text{dB}$ | Pending hardware | `NOT RUN` |
| **Stage 5** | Speaker & I2S Amplifier | Intelligible voice, ERLE $>12\text{dB}$ | Pending hardware | `NOT RUN` |
| **Stage 6** | Device-to-Host Transport | CRC error rate $<0.01\%$ | Pending hardware | `NOT RUN` |
| **Stage 7** | Power & Thermal Bench | Quiescent $<10\text{mA}$, safe gates | Pending hardware | `NOT RUN` |
| **Stage 8** | Integrated E2E Acceptance | Latency $<3500\text{ms}$, voice reply | Pending hardware | `NOT RUN` |

---

## 6. Discrepancies, Anomalies & Corrective Action Items

| Issue # | Symptom / Discrepancy | Root Cause Analysis | Action Item / Owner | Status |
| :--- | :--- | :--- | :--- | :--- |
| `BRINGUP-01` | Onboard PDM vs External I2S Mic | Sense daughterboard uses PDM (GPIO 41/42) | Set firmware config to PDM mode | Documented |
| `BRINGUP-02` | Physical Hardware Not Connected | Hardware currently in transit / unpopulated | Execute Stages 1-8 upon bench delivery | Pending Arrival |
