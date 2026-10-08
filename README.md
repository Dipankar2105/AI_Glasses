# AI Glasses

## Purpose
A wearable multimodal AI assistant.

## Hardware
- XIAO ESP32-S3 Sense
- OV3660 Camera
- Audio hardware
- MPU6050 (6-axis motion tracking)
- Capacitive touch sensor
- LiPo battery
- MicroSD card

## High-level Architecture
The glasses capture audio, video, and sensor data, transmitting it over Wi-Fi to a Python AI backend for processing (Vision, OCR, Translation, AI). The backend sends responses back to the device.
See `docs/architecture.md` for more details.

## Development Philosophy
- Phase-by-phase approach.
- test → verify → document → commit.
- Never commit secrets, Wi-Fi passwords, API keys, or personal credentials.
- Do not make speculative changes.
- Preserve working states.
- Use separate commits for meaningful milestones.
