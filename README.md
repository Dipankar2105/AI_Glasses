# NextSight Smart Glasses — AI Glasses Framework

[![CI / Regression Tests](https://img.shields.io/badge/Tests-198%20Passed-brightgreen)](https://github.com/Dipankar2105/AI_Glasses)
[![Platform](https://img.shields.io/badge/Platform-ESP32--S3%20%7C%20FastAPI-blue)](https://github.com/Dipankar2105/AI_Glasses)
[![Hardware Readiness](https://img.shields.io/badge/Hardware-Ready%20for%20Physical%20Bringup-orange)](docs/phase10-hardware-integration-readiness.md)

NextSight is an open, modular software and firmware architecture for AI-powered smart glasses. Designed around the **Seeed Studio XIAO ESP32-S3 Sense**, NextSight provides real-time head motion gesture recognition, silent capacitive touch interaction, power and thermal management, conversational orchestration, Model Context Protocol (MCP) tool execution, and local vision processing.

> **Hardware Notice**: The entire software stack is fully validated on the host system using deterministic synthetic sensor traces and device simulation. Physical sensor and hardware validation is prepared and ready for prototype bench testing upon hardware arrival.

---

## Key Features & Architecture

```
+-----------------------------------------------------------------------------------+
|                        NextSight Smart Glasses Architecture                       |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ Physical Glasses / ESP32-S3 Simulator ]                                        |
|  ├── OV3660 Camera ──────> DVP Frame Capture ──────> JPEG Encoder                |
|  ├── I2S MEMS Mic  ──────> Audio DSP (AEC / AGC / VAD) ──> 16kHz PCM Stream       |
|  ├── MPU-6050 IMU  ──────> Zero-Crossing Motion Filter ──> Gesture Events        |
|  └── Touch Pad     ──────> 30ms Debounced State Machine ─> Tap / Long Press       |
|                                     │                                             |
|                                     ▼ (12-byte Binary Framing + CRC32 Trailer)    |
|                                                                                   |
|  [ Python AI Backend (FastAPI / Asynchronous Orchestrator) ]                      |
|  ├── Protocol Engine    ──> CRC Validation & Packet Sequence Tracking             |
|  ├── Power & Thermals   ──> Discharge Hysteresis & Workload Gating                |
|  ├── Interaction Router ──> Safe Gesture-to-Intent Dispatcher                     |
|  ├── Conversation / MCP ──> Bounded Session History & Sandboxed Tool Execution    |
|  └── Vision Pipeline    ──> Object Detection & Scene Understanding                |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/Dipankar2105/AI_Glasses.git
cd AI_Glasses
cp .env.example .env
pip install -r requirements.txt
```

### 2. Execute Full Software Validation Suite
Run all unit tests, integration scenarios, and 1,000-iteration performance benchmarks:
```bash
python run_full_validation.py
```

### 3. Start Python AI Backend Server
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Interactive API documentation will be available at `http://localhost:8000/docs`.

---

## Roadmap & Status

| Phase | Subsystem | Status | Documentation |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Architecture Foundation | COMPLETED | [`docs/architecture.md`](docs/architecture.md) |
| **Phase 1** | Xiaozhi Firmware Baseline | COMPLETED | [`docs/phase1-xiaozhi-baseline.md`](docs/phase1-xiaozhi-baseline.md) |
| **Phase 2** | Audio DSP (AEC/DTD/AGC/VAD) | COMPLETED | [`docs/audio-dsp-architecture.md`](docs/audio-dsp-architecture.md) |
| **Phase 3** | Hardware Abstraction Layer | COMPLETED | [`docs/phase3-hal-architecture.md`](docs/phase3-hal-architecture.md) |
| **Phase 4** | Vision Foundation & Pipeline | COMPLETED | [`docs/phase4-vision-foundation.md`](docs/phase4-vision-foundation.md) |
| **Phase 4C**| OCR Research & Benchmarking | ON HOLD | [`docs/phase4c19-notebook-ocr-pipeline.md`](docs/phase4c19-notebook-ocr-pipeline.md) |
| **Phase 5** | Python AI Backend Foundation | COMPLETED | [`docs/phase5-python-ai-backend.md`](docs/phase5-python-ai-backend.md) |
| **Phase 7** | Conversational Intelligence + MCP | COMPLETED | [`docs/phase7-conversation-mcp.md`](docs/phase7-conversation-mcp.md) |
| **Phase 8** | Motion & Silent Interaction | COMPLETED | [`docs/phase8-motion-silent-interaction.md`](docs/phase8-motion-silent-interaction.md) |
| **Phase 9** | Power & Thermal Management | COMPLETED | [`docs/phase9-power-thermal-management.md`](docs/phase9-power-thermal-management.md) |
| **Phase 10**| Hardware Integration Readiness | BLOCKED — HW UNAVAILABLE | [`docs/phase10-hardware-integration-readiness.md`](docs/phase10-hardware-integration-readiness.md) |
| **Phase 11**| Performance & Reliability | COMPLETED | [`docs/phase11-performance-reliability.md`](docs/phase11-performance-reliability.md) |
| **Phase 12**| Productization & Release Readiness | COMPLETED | [`docs/phase12-productization.md`](docs/phase12-productization.md) |

---

## License
MIT License.
