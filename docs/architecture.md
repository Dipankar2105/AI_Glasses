# Architecture

## High-Level Data Flow

1. **XIAO ESP32-S3** manages on-device peripherals:
   - Audio (microphone/speaker)
   - Camera (OV3660)
   - Sensors (MPU6050, Capacitive touch)
   - Power (LiPo battery management)
2. **Wi-Fi / MCP / API** handles data transmission between the device and the backend.
3. **Python AI Backend** processes the data.
4. **AI Processing** encompasses Vision, OCR, Translation, and AI interaction.
5. **Response** is sent back to the ESP32 device for output.

```
XIAO ESP32-S3 → audio/camera/sensors/power → Wi-Fi/MCP/API → Python AI backend → vision/OCR/translation/AI → response back to device
```
