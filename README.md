# NextSight Smart Glasses — AI Glasses Framework

[![CI / Full Software Validation](https://img.shields.io/badge/Tests-329%20Passed-brightgreen)](https://github.com/Dipankar2105/AI_Glasses)
[![Platform](https://img.shields.io/badge/Platform-ESP32--S3%20%7C%20FastAPI-blue)](https://github.com/Dipankar2105/AI_Glasses)
[![Hardware Readiness](https://img.shields.io/badge/Hardware-Ready%20for%20Physical%20Bringup-orange)](docs/phase10-hardware-integration-readiness.md)
[![OCR Status](https://img.shields.io/badge/OCR-ON%20HOLD-yellow)](docs/ocr-evidence-audit.md)

NextSight is an open, modular software and firmware architecture for AI-powered smart glasses. Designed around the **Seeed Studio XIAO ESP32-S3 Sense**, NextSight provides real-time head motion gesture recognition, silent capacitive touch interaction, power and thermal management, software audio DSP (AEC, DTD, Noise Suppression, VAD, AGC, Limiter), speech-to-text (STT), text-to-speech (TTS), AI provider adapters (Whisper, OpenAI, Gemini, Anthropic), conversational orchestration, Model Context Protocol (MCP) tool execution, and local vision processing.

> **Hardware & Scope Notice**: The entire software stack is fully validated on the host system using deterministic synthetic sensor traces, simulated audio streams, simulated ESP32 framing, and mock AI providers. Physical hardware validation is prepared and ready for bench bringup once physical glasses are assembled. OCR research is strictly **ON HOLD** to preserve frozen benchmarks.

---

## System Architecture

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
|  [ Python AI Backend (FastAPI / Asynchronous Runtime Engine) ]                    |
|  ├── Protocol Engine    ──> CRC32 Validation & Packet Sequence Tracking           |
|  ├── Power & Thermals   ──> Discharge Hysteresis & Workload Gating (Strict Mode)  |
|  ├── Audio Service      ──> Integrated DSP Pipeline + Speech-to-Text (STT)        |
|  ├── AI Providers Layer ──> Real Provider Adapters (Whisper / OpenAI / API / Mock)|
|  ├── Interaction Router ──> Safe Gesture-to-Intent Dispatcher                     |
|  ├── Conversation / MCP ──> Bounded Session History & Sandboxed Tool Execution    |
|  ├── Speech Output      ──> Text-to-Speech (TTS) PCM Stream Synthesis             |
|  └── Vision Pipeline    ──> Object Detection & Scene Understanding (Quality Eval) |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## Runtime API Overview

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service liveness probe, uptime, and PID | No |
| `GET` | `/ready` | Subsystem pipeline readiness check | No |
| `GET` | `/api/v1/capabilities` | Honest capability declaration | No |
| `GET` | `/api/v1/providers/status` | Configuration and readiness of STT, LLM, TTS | Optional |
| `POST` | `/api/v1/audio/transcribe` | Audio DSP preprocessing + Speech-to-Text | Optional |
| `POST` | `/api/v1/speech/synthesize`| Text-to-Speech audio PCM generation | Optional |
| `POST` | `/api/v1/conversation/message`| Multi-turn conversation reasoning + MCP tools | Optional |
| `GET` | `/api/v1/conversation/sessions/{id}`| Session message history & context | Optional |
| `DELETE`| `/api/v1/conversation/sessions/{id}`| Clear conversation session | Optional |
| `GET` | `/api/v1/mcp/tools` | List registered MCP tool definitions | Optional |
| `POST` | `/api/v1/mcp/tools/call` | Sandboxed execution of registered MCP tool | Optional |
| `POST` | `/api/v1/mcp/jsonrpc` | JSON-RPC 2.0 protocol endpoint for MCP clients | Optional |
| `POST` | `/api/v1/vision/process` | Vision pipeline (object detection & scene eval) | Optional |

---

## Quick Start & Developer Guide

### 1. Environment Setup
```bash
git clone https://github.com/Dipankar2105/AI_Glasses.git
cd AI_Glasses
cp .env.example .env
pip install -r requirements.txt
```

### 2. Execute Full Software Validation Suite
Runs 329 regression tests across all subsystems and 1,000-iteration performance benchmarks:
```bash
python run_full_validation.py
```

### 3. Run Offline Local System Demonstration
Executes all 7 integration stages (Protocol, Motion, Power Gating, MCP, Conversation, Fault Tolerance, Voice AI Loop):
```bash
python tests/scripts/run_local_demo.py
```

### 4. Start Python AI Backend Server
```bash
# Offline development mode (uses deterministic mocks, no API keys needed)
python scripts/start_backend.py --offline

# Or standard startup using configured environment
python scripts/start_backend.py --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

---

## Configuration & AI Providers

Configure providers via environment variables or `.env`:

```ini
# Provider Types: 'mock' (offline), 'whisper' (STT), 'openai'/'api' (LLM/TTS), 'null' (unavailable)
NEXTSIGHT_STT_PROVIDER=mock
NEXTSIGHT_STT_MODEL=whisper-1

NEXTSIGHT_LLM_PROVIDER=mock
NEXTSIGHT_LLM_MODEL=gpt-4o-mini

NEXTSIGHT_TTS_PROVIDER=mock
NEXTSIGHT_TTS_MODEL=tts-1
NEXTSIGHT_TTS_VOICE=alloy

# Optional Real Credentials (Never committed; offline mode requires no keys)
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=AIza...
# ANTHROPIC_API_KEY=sk-ant-...
# NEXTSIGHT_API_KEY=your-custom-client-key
```

---

## Implementation & Validation Status Matrix

| Subsystem | Implemented | Unit Tested | Offline E2E Tested | Host Benchmarked | Real Provider Tested | Hardware Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Audio DSP Pipeline** | Yes | Yes | Yes | Yes (0.02ms/frame) | N/A (Local) | Simulated |
| **STT Provider Adapter**| Yes | Yes | Yes | Yes | Credential-Gated | Simulated Mic |
| **LLM Provider Adapter**| Yes | Yes | Yes | Yes | Credential-Gated | N/A (Cloud) |
| **TTS Provider Adapter**| Yes | Yes | Yes | Yes | Credential-Gated | Simulated Spk |
| **Conversation & MCP** | Yes | Yes | Yes | Yes (0.004ms) | Yes (Mock/Real) | N/A (Host) |
| **Vision & Image Eval** | Yes | Yes | Yes | Yes (1.2ms) | N/A (Local) | Simulated Cam |
| **Motion & Gestures** | Yes | Yes | Yes | Yes (0.02ms) | N/A (Local) | Simulated IMU |
| **Power & Thermals** | Yes | Yes | Yes | Yes (0.004ms) | N/A (Local) | Simulated ADC |
| **Protocol & Framing** | Yes | Yes | Yes | Yes (0.003ms) | N/A (Local) | Simulated Link|
| **OCR Notebook Eval** | Preserved | Preserved | N/A | Preserved | N/A | **ON HOLD** |

---

## License
MIT License.
