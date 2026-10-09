# NextSight Smart Glasses — Local System Demonstration & Integration Gate

**Status:** Verified & Reproducible  
**Date:** 2026-10-09  
**Execution Command:** `python tests/scripts/run_local_demo.py`  
**Automated Tests:** [`backend/tests/test_local_demo_integration.py`](file:///c:/Users/Routewise/AI_Glasses/backend/tests/test_local_demo_integration.py)

---

## 1. Overview & Demonstration Purpose

This document details the reproducible host-side demonstration of the NextSight smart glasses platform. The demonstration exercises the complete, unbroken local software path:
1. **Backend Liveness, Readiness & Capabilities:** FastAPI application lifecycle, health probes, and honest capability declarations.
2. **Device Simulation & Binary Framing:** 12-byte header + CRC32 verification, sequence tracking, and telemetry decoding.
3. **Interaction Dispatching:** Head gestures (Nod, Shake, Tilt) and capacitive touch interactions (Tap, Double Tap, Long Press) routed to appropriate application services.
4. **Power & Thermal Policy Gating:** Synthetic telemetry drops trigger power-saving modes and gate vision capture workloads.
5. **Conversational Intelligence & MCP:** JSON-RPC 2.0 tool execution and bounded session conversation management.
6. **Fault Tolerance & Circuit Breaking:** Malformed packet rejection and downstream failure isolation.

---

## 2. Step-by-Step Demonstration Flow

```
[Simulated ESP32-S3 Glasses]
    │
    ├── 1. Binary Packet (Header: Magic 0xAA55, Seq, Ts) + CRC32 Trailer
    │
    ▼
[ProtocolFraming Engine] ───> CRC32 & Sequence Number Integrity Check
    │
    ├── 2. Telemetry Packet ───> [Power & Thermal Policy Manager]
    │                            ├── Evaluates Battery & Die Temperature
    │                            └── Sets Operating State (e.g. IDLE, LOW_POWER)
    │
    ├── 3. Gesture Event ──────> [Interaction Dispatcher]
    │                            ├── Tap ──────────> CONFIRM Intent
    │                            ├── Double Tap ───> TRIGGER_SCENE_ANALYSIS (Gated by Power Mode)
    │                            └── Nod / Shake ──> Safe Action Routing
    │
    ▼
[Application Services]
    ├── ConversationService ───> Bounded Session Manager
    ├── MCPServer ─────────────> Sandboxed System / Vision Status Tools
    └── VisionService ─────────> Preprocessing & Object/Scene Understanding
```

---

## 3. How to Run the Demonstration

### Command:
```bash
python tests/scripts/run_local_demo.py
```

### Expected Output Summary:
```text
================================================================================
 NEXTSIGHT LOCAL SYSTEM DEMONSTRATION & INTEGRATION GATE
================================================================================
Environment: Python 3.14 (AMD64 / Host Software Runtime)
Mode: Synthetic Sensor Traces & Local Software Simulation (No Physical Hardware Claimed)

================================================================================
 STEP 1: BACKEND HEALTH, READINESS & CAPABILITIES
================================================================================
[/health] Status: ok | App: NextSight AI Backend v0.5.0 | Uptime: 0.04s
[/ready] System State: READY
  - Subsystem 'vision_pipeline': READY
  - Subsystem 'object_detector': READY
  - Subsystem 'scene_analyzer': READY
  - Subsystem 'ocr_engine': DEFERRED_PHASE_5
  - Subsystem 'llm_reasoning': DEFERRED_PHASE_5
  - Subsystem 'hardware_camera': UNAVAILABLE
  - Subsystem 'hardware_audio': UNAVAILABLE

================================================================================
 STEP 2: DEVICE SIMULATOR & BINARY PROTOCOL FRAMING
================================================================================
[Device Simulator] Connected device ID: nextsight-glasses-demo
[Framing] Heartbeat Packet: Valid=True, Magic=0xAA55, Seq=0, CRC32=0xBAED3950
[Telemetry Ingest] Processed -> Battery Status: NORMAL, Thermal: NORMAL, Power State: IDLE

================================================================================
 STEP 3: MOTION & TOUCH INTERACTION DISPATCHING
================================================================================
[Touch: TAP] -> Intent: CONFIRM | Action: CONFIRMATION_SIGNAL_EMITTED | Service: InteractionState | Success: True
[Touch: DOUBLE_TAP] -> Intent: TRIGGER_SCENE_ANALYSIS | Action: SCENE_ANALYSIS_DISPATCHED | Service: VisionService | Success: True
[Head: NOD] -> Intent: CONFIRM | Action: COOLDOWN_SUPPRESSED | Service: NONE

================================================================================
 STEP 4: POWER POLICY GATING & WORKLOAD PROTECTION
================================================================================
[Battery Telemetry Drop] Battery=12.0% -> Power Mode: LOW_POWER, Battery Status: LOW
[Vision Trigger under Low Battery] Status: BLOCKED_BY_POWER_POLICY | Reason: Vision capture disabled to conserve low battery
[Battery Recovery] Battery restored to 50.0% -> Normal workload execution resumed.

================================================================================
 STEP 5: CONVERSATION ORCHESTRATION & MCP TOOL INVOCATION
================================================================================
[MCP Tool: get_system_status] Success: True | Latency: 1.0ms
[Conversation API] Session ID: <uuid>
  Response: [LLM Unavailable] Conversation orchestration and tool execution are active, but no LLM reasoning provider is currently configured.
  LLM Provider Status: PROVIDER_UNAVAILABLE (Honest declaration: paid LLM keys deferred)
  Tool Executions: 1 executed

================================================================================
 STEP 6: FAULT TOLERANCE & BOUNDARY VALIDATION
================================================================================
[Corrupted Magic] Status: REJECTED | Error: Invalid magic bytes 0x0000 (expected 0xAA55)
[Corrupted CRC32] Status: REJECTED | Error: CRC32 mismatch: calculated 0x763131B5, received 0x99EA0E20
[Circuit Breaker] Tripped to State: OPEN | can_execute: False

================================================================================
 LOCAL SYSTEM DEMONSTRATION SUMMARY
================================================================================
All 6 integration stages executed cleanly without errors!
```

---

## 4. Integration Boundaries & Known Limitations

1. **Host-Side Protocol vs. Physical Network Socket:**
   - The protocol framing engine and device simulator encode/decode binary frames in memory. When the physical ESP32-S3 hardware is brought up, a WebSocket endpoint (`/ws/stream`) will transport these same framed binary byte arrays over TCP/IP.
2. **Deterministic Mocks vs. Live Cloud AI:**
   - Object detection and scene analysis run locally using deterministic software vision models.
   - LLM conversation generation and speech synthesis report honest `PROVIDER_UNAVAILABLE` status because paid external API keys are deferred.
3. **Synthetic Traces vs. Physical Sensors:**
   - Gestures are evaluated against calibrated mathematical traces. Real-world sensor calibration (zero-rate gyro drift, touch capacitance threshold) will occur during physical bringup.
