# NextSight Runtime API & Integration Guide

## 1. Overview & Trust Boundaries

The NextSight Python AI Backend serves as the central orchestration runtime for the smart glasses. It exposes asynchronous HTTP endpoints for:
1. Audio DSP ingestion and speech-to-text (STT).
2. Contextual conversation reasoning and Large Language Model (LLM) orchestration.
3. Model Context Protocol (MCP) sandboxed tool execution.
4. Text-to-speech (TTS) synthesis with accurate PCM metadata.
5. Multimodal vision frame processing and image-quality analysis.
6. Power/thermal policy gating and subsystem health monitoring.

---

## 2. API Endpoints & Request / Response Specifications

### 2.1 Audio Transcription (`POST /api/v1/audio/transcribe`)
Processes raw audio through the host DSP chain (DC Blocker, High Pass @ 80Hz, AEC+DTD, Spectral Noise Suppression, VAD, AGC, Limiter) and transcribes with the configured STT provider.

- **Request Body (JSON):**
```json
{
  "audio_base64": "<base64-encoded 16-bit PCM or WAV bytes>",
  "sample_rate": 16000,
  "channels": 1,
  "reference_audio_base64": null,
  "run_dsp": true
}
```
- **Response (200 OK):**
```json
{
  "success": true,
  "transcript": "What is the battery level of my glasses?",
  "confidence": 0.96,
  "vad_speech_active": true,
  "dsp_metrics": {
    "input_rms": 450.2,
    "output_rms": 3950.8,
    "rms_gain_db": 18.86,
    "vad_speech_active": true,
    "aec_status": "BYPASS_NO_REFERENCE",
    "clipping_detected": false,
    "stages_executed": ["DCBlocker", "HighPassFilter", "SpectralNoiseSuppression", "VAD", "AGC", "Limiter"]
  },
  "latency_ms": 14.5,
  "error": null,
  "request_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"
}
```

### 2.2 Speech Synthesis (`POST /api/v1/speech/synthesize`)
Synthesizes assistant response text into spoken audio PCM bytes with explicit sample rate metadata.

- **Request Body (JSON):**
```json
{
  "text": "Battery level is currently eighty-five percent. All sensors are operational.",
  "voice": "alloy",
  "format": "pcm"
}
```
- **Response (200 OK):**
```json
{
  "success": true,
  "audio_base64": "<base64-encoded PCM bytes>",
  "sample_rate": 24000,
  "encoding": "pcm_s16le",
  "channels": 1,
  "latency_ms": 18.2,
  "error": null,
  "request_id": "c1a9f334-3b87-4bd5-8c2e-9a0998d6044f"
}
```

### 2.3 Conversation Submission (`POST /api/v1/conversation/message`)
Submits a user query into a conversation session, optionally executes registered MCP tools, bounds tool recursion (max 3 loops), and returns structured responses.

- **Request Body (JSON):**
```json
{
  "session_id": "user-session-001",
  "message": "Check system status and camera health",
  "tool_to_invoke": "get_system_status",
  "tool_arguments": {},
  "vision_context": {
    "scene_description": "Office desk with laptop and monitor",
    "objects": ["laptop", "monitor", "cup"],
    "confidence": 0.92
  }
}
```
- **Response (200 OK):**
```json
{
  "session_id": "user-session-001",
  "message_id": "8f3b2008-a581-4c67-9d4b-8429f593dae2",
  "response": "NextSight AI Backend v0.5.0 is healthy and operational in development mode.",
  "llm_status": "OPERATIONAL",
  "tool_executions": [
    {
      "tool_name": "get_system_status",
      "arguments": {},
      "success": true,
      "result": [{"type": "text", "data": {"app_name": "NextSight AI Backend", "version": "0.5.0"}}],
      "error": null,
      "latency_ms": 0.85
    }
  ],
  "session_message_count": 2,
  "latency_ms": 22.4,
  "timestamp": 1791606012.0
}
```

### 2.4 Provider Status (`GET /api/v1/providers/status`)
Returns real configuration and readiness status of STT, LLM, and TTS providers without exposing API keys.

- **Response (200 OK):**
```json
{
  "stt": {
    "provider_type": "mock",
    "model": "whisper-1",
    "configured": true,
    "available": true,
    "status": "OPERATIONAL (Mock/Deterministic)",
    "offline_mode": true,
    "api_key_configured": false
  },
  "llm": {
    "provider_type": "mock",
    "model": "gpt-4o-mini",
    "configured": true,
    "available": true,
    "status": "OPERATIONAL (Mock/Deterministic)",
    "offline_mode": true,
    "api_key_configured": false
  },
  "tts": {
    "provider_type": "mock",
    "model": "tts-1",
    "configured": true,
    "available": true,
    "status": "OPERATIONAL (Mock/Deterministic)",
    "offline_mode": true,
    "api_key_configured": false
  },
  "environment": "development",
  "timestamp": 1791606015.0
}
```

---

## 3. Security, Rate Limiting & Resource Guards

1. **Payload Size Guard**: Global Content-Length limit of 15MB (`max_request_body_bytes`) protects against heap exhaustion. Audio payloads exceeding 10MB return `413 Request Entity Too Large`.
2. **Client Authentication**: When `NEXTSIGHT_API_KEY` is configured in `.env`, protected routes reject unauthenticated requests with `401 Unauthorized`. Liveness probes (`/health`, `/ready`, `/api/v1/capabilities`) remain public.
3. **Sliding-Window Rate Limiting**: Client IP rate limiting (default: 300 requests/minute) returns `429 Too Many Requests` with a `Retry-After` header during request spikes.
4. **Session History Bounding**: In-memory `SessionManager` bounds conversation histories (default: 20 messages per session) while preserving initial system instructions.
5. **Sandboxed Tool Execution**: `MCPToolRegistry` isolates execution in separate asynchronous worker threads with a 5.0-second timeout. Tools cannot invoke shell or disk routines.
6. **Secret Masking**: Application logging automatically suppresses secrets, authorization headers, and API keys.

---

## 4. Troubleshooting & FAQ

- **Q: Why are live cloud responses not returned in offline mode?**  
  *A:* Offline mode uses deterministic mock providers (`MockSTTProvider`, `MockLLMProvider`, `MockTTSProvider`) to enable 100% reproducible testing without network access or paid API keys.
- **Q: How do I enable real OpenAI/Whisper providers?**  
  *A:* Set `NEXTSIGHT_STT_PROVIDER=whisper`, `NEXTSIGHT_LLM_PROVIDER=openai`, `NEXTSIGHT_TTS_PROVIDER=openai`, and export `OPENAI_API_KEY=sk-...` in `.env`.
- **Q: What is the status of OCR?**  
  *A:* OCR research is strictly **ON HOLD** to maintain frozen benchmark baselines and avoid unverified claims on handwritten note recognition.
