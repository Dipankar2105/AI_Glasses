# Power Budget & Modeling

> **IMPORTANT**: All values in this document are modeled engineering estimates based on component datasheets at 3.3V supply rail. They must be updated with actual measured values once physical hardware is assembled and tested.

## Modeled Estimated Power Consumption

| Component | State | Estimated Current (mA) | Estimated Power (mW) |
| :--- | :--- | :--- | :--- |
| **XIAO ESP32-S3 Core** | Light Sleep | 2.0 mA | 6.6 mW |
| **XIAO ESP32-S3 Core** | Active Dual-Core (240MHz) | 45.0 mA | 148.5 mW |
| **Wi-Fi Radio** | Active (Tx/Rx Avg) | 180.0 mA | 594.0 mW |
| **OV3660 Camera** | Standby / Gated | 0.1 mA | 0.33 mW |
| **OV3660 Camera** | Active (Streaming) | 70.0 mA | 231.0 mW |
| **Audio Mic (I2S MEMS)** | Active | 1.5 mA | 4.95 mW |
| **Audio Speaker (MAX98357A)** | Playback Avg | 120.0 mA | 396.0 mW |
| **MPU-6050 IMU** | Active 6-DOF (100Hz) | 3.8 mA | 12.54 mW |
| **MicroSD Card** | Write Burst | 45.0 mA | 148.5 mW |

## Modeled Battery Life Projections (500 mAh / 3.7V LiPo)

- **Nominal Battery Energy**: 1850 mWh (approx 1665 mWh usable at 90% efficiency)
- **Estimated Idle Standby Time**: ~82.4 hours (ESP32 light sleep + IMU active)
- **Estimated Listening Time**: ~2.19 hours continuous
- **Estimated Camera Capture Loop**: ~1.68 hours continuous
- **Estimated Speech Playback Time**: ~1.45 hours continuous
