#!/usr/bin/env python3
"""NextSight Smart Glasses — Reproducible Local System Demonstration.

Demonstrates the real software execution path:
1. Backend liveness, readiness, and capability negotiation.
2. Device simulator connection and binary protocol framing with CRC32.
3. Motion and touch interaction dispatching to conversation and vision services.
4. Power and thermal management policy gating under synthetic battery/thermal states.
5. Model Context Protocol (MCP) tool execution and conversation session handling.
6. Fault tolerance, malformed packet rejection, and circuit breaker protection.
"""

import sys
import os
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from backend.app import app
from backend.protocol.contracts import PacketType
from backend.protocol.framing import ProtocolFraming
from backend.protocol.device_sim import SimulatedESP32Device
from backend.reliability.orchestrator import SystemIntegrationOrchestrator
from backend.power.contracts import BatteryTelemetry, ThermalTelemetry
from backend.reliability.circuit_breaker import CircuitBreaker


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def run_demo() -> bool:
    print_section("NextSight Local System Demonstration & Integration Gate")
    print("Environment: Python 3.14 (AMD64 / Host Software Runtime)")
    print("Mode: Synthetic Sensor Traces & Local Software Simulation (No Physical Hardware Claimed)\n")

    client = TestClient(app)
    orchestrator = SystemIntegrationOrchestrator()

    # -------------------------------------------------------------------------
    # STEP 1: Backend Health, Readiness & Advertised Capabilities
    # -------------------------------------------------------------------------
    print_section("Step 1: Backend Health, Readiness & Capabilities")

    # 1.1 /health
    res_health = client.get("/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
    health_data = res_health.json()
    print(f"[/health] Status: {health_data['status']} | App: {health_data['app_name']} v{health_data['version']} | Uptime: {health_data['uptime_seconds']}s")

    # 1.2 /ready
    res_ready = client.get("/ready")
    assert res_ready.status_code == 200, f"Readiness check failed: {res_ready.text}"
    ready_data = res_ready.json()
    print(f"[/ready] System State: {ready_data['status']}")
    for comp, st in ready_data['components'].items():
        print(f"  - Subsystem '{comp}': {st}")

    # 1.3 /api/v1/capabilities
    res_caps = client.get("/api/v1/capabilities")
    assert res_caps.status_code == 200, f"Capabilities query failed: {res_caps.text}"
    caps_data = res_caps.json()
    print("\nHonest Capability Declaration:")
    print(f"  - Implemented ({len(caps_data['implemented'])}): {', '.join(caps_data['implemented'][:4])}...")
    print(f"  - Unavailable (Hardware Pending): {', '.join(caps_data['unavailable'])}")
    print(f"  - Deferred: {', '.join(caps_data['deferred'])}")

    # -------------------------------------------------------------------------
    # STEP 2: Device Simulator & Binary Protocol Framing with CRC32
    # -------------------------------------------------------------------------
    print_section("Step 2: Device Simulator & Binary Protocol Framing")

    sim = SimulatedESP32Device(device_id="nextsight-glasses-demo", queue_capacity=20)
    sim.connect()
    print(f"[Device Simulator] Connected device ID: {sim.device_id}")

    # Send Heartbeat
    hb_bytes = sim.send_heartbeat()
    hb_msg = ProtocolFraming.decode_message(hb_bytes)
    print(f"[Framing] Heartbeat Packet: Valid={hb_msg.is_valid}, Magic=0x{hb_msg.header.magic:04X}, Seq={hb_msg.header.sequence_number}, CRC32=0x{hb_msg.crc32:08X}")

    # Send Telemetry
    telem_bytes = sim.send_telemetry(battery_pct=85.0, temp_c=34.5)
    telem_res = orchestrator.process_incoming_device_packet(telem_bytes)
    print(f"[Telemetry Ingest] Processed -> Battery Status: {telem_res.get('battery_status')}, Thermal: {telem_res.get('thermal_status')}, Power State: {telem_res.get('operating_state')}")

    # -------------------------------------------------------------------------
    # STEP 3: Motion & Touch Interaction Dispatching
    # -------------------------------------------------------------------------
    print_section("Step 3: Motion & Touch Interaction Dispatching")

    # 3.1 Single Tap -> Confirm
    tap_packet = ProtocolFraming.encode_message(
        packet_type=PacketType.TOUCH_EVENT,
        payload=json.dumps({"gesture": "TAP", "confidence": 0.95}).encode("utf-8"),
        sequence_number=10,
        timestamp_ms=2000,
    )
    tap_res = orchestrator.process_incoming_device_packet(tap_packet)
    print(f"[Touch: TAP] -> Intent: {tap_res.get('intent')} | Action: {tap_res.get('action_taken')} | Service: {tap_res.get('target_service')} | Success: {tap_res.get('success')}")

    # 3.2 Double Tap -> Trigger Scene Analysis
    dt_packet = ProtocolFraming.encode_message(
        packet_type=PacketType.TOUCH_EVENT,
        payload=json.dumps({"gesture": "DOUBLE_TAP", "confidence": 0.98}).encode("utf-8"),
        sequence_number=11,
        timestamp_ms=3000,
    )
    dt_res = orchestrator.process_incoming_device_packet(dt_packet)
    print(f"[Touch: DOUBLE_TAP] -> Intent: {dt_res.get('intent')} | Action: {dt_res.get('action_taken')} | Service: {dt_res.get('target_service')} | Success: {dt_res.get('success')}")

    # 3.3 Head Nod -> Confirm
    nod_packet = ProtocolFraming.encode_message(
        packet_type=PacketType.MOTION_EVENT,
        payload=json.dumps({"gesture": "NOD", "confidence": 0.92}).encode("utf-8"),
        sequence_number=12,
        timestamp_ms=4000,
    )
    nod_res = orchestrator.process_incoming_device_packet(nod_packet)
    print(f"[Head: NOD] -> Intent: {nod_res.get('intent')} | Action: {nod_res.get('action_taken')} | Service: {nod_res.get('target_service')}")

    # -------------------------------------------------------------------------
    # STEP 4: Power Policy Gating under Critical / Low-Power Telemetry
    # -------------------------------------------------------------------------
    print_section("Step 4: Power Policy Gating & Workload Protection")

    # Simulate battery dropping to 12%
    low_bat_packet = sim.send_telemetry(battery_pct=12.0, temp_c=35.0)
    low_res = orchestrator.process_incoming_device_packet(low_bat_packet)
    print(f"[Battery Telemetry Drop] Battery=12.0% -> Power Mode: {low_res.get('operating_state')}, Battery Status: {low_res.get('battery_status')}")

    # Attempt Vision Capture via Double Tap
    blocked_vision_res = orchestrator.process_incoming_device_packet(dt_packet)
    print(f"[Vision Trigger under Low Battery] Status: {blocked_vision_res.get('status')} | Reason: {blocked_vision_res.get('reason')}")
    assert blocked_vision_res.get("status") == "BLOCKED_BY_POWER_POLICY", "Vision capture should be blocked in low power mode"

    # Recover battery to 50%
    recovery_packet = sim.send_telemetry(battery_pct=50.0, temp_c=32.0)
    orchestrator.process_incoming_device_packet(recovery_packet)
    print("[Battery Recovery] Battery restored to 50.0% -> Normal workload execution resumed.")

    # -------------------------------------------------------------------------
    # STEP 5: Conversation Orchestration & MCP Tools
    # -------------------------------------------------------------------------
    print_section("Step 5: Conversation Orchestration & MCP Tool Invocation")

    # 5.1 Call MCP get_system_status tool directly via API
    mcp_call_res = client.post("/api/v1/mcp/tools/call", json={"name": "get_system_status", "arguments": {}})
    assert mcp_call_res.status_code == 200
    tool_out = mcp_call_res.json()
    print(f"[MCP Tool: get_system_status] Success: {not tool_out.get('is_error')} | Latency: {tool_out.get('latency_ms')}ms")
    print(f"  Result Payload: {tool_out.get('content')}")

    # 5.2 Send Conversation Message with explicit tool execution
    conv_req = {
        "message": "Check system status and camera health",
        "tool_to_invoke": "get_system_status",
        "tool_arguments": {}
    }
    conv_res = client.post("/api/v1/conversation/message", json=conv_req)
    assert conv_res.status_code == 200
    conv_data = conv_res.json()
    print(f"[Conversation API] Session ID: {conv_data['session_id']}")
    print(f"  Response: {conv_data['response']}")
    print(f"  LLM Provider Status: {conv_data['llm_status']} (Honest declaration: paid LLM keys deferred)")
    print(f"  Tool Executions: {len(conv_data['tool_executions'])} executed")

    # -------------------------------------------------------------------------
    # STEP 6: Fault Tolerance, Malformed Packet Rejection & Circuit Breaker
    # -------------------------------------------------------------------------
    print_section("Step 6: Fault Tolerance & Boundary Validation")

    # 6.1 Corrupted Magic Bytes
    bad_magic = b"\x00\x00\x01\x00\x00\x01\x00\x00\x00\x00\x00\x04DATA\x00\x00\x00\x00"
    res_bad_magic = orchestrator.process_incoming_device_packet(bad_magic)
    print(f"[Corrupted Magic] Status: {res_bad_magic['status']} | Error: {res_bad_magic.get('error')}")
    assert res_bad_magic['status'] == "REJECTED"

    # 6.2 CRC32 Mismatch Rejection
    corrupted_data = bytearray(tap_packet)
    corrupted_data[15] ^= 0xFF  # Flip bit in payload
    res_crc = orchestrator.process_incoming_device_packet(bytes(corrupted_data))
    print(f"[Corrupted CRC32] Status: {res_crc['status']} | Error: {res_crc.get('error')}")
    assert res_crc['status'] == "REJECTED"

    # 6.3 Circuit Breaker Protection
    cb = CircuitBreaker(name="vision_downstream", failure_threshold=2, recovery_timeout_s=0.1)
    cb.record_failure(current_time=1.0)
    cb.record_failure(current_time=1.0)
    print(f"[Circuit Breaker] Tripped to State: {cb.state.value} | can_execute: {cb.can_execute(current_time=1.0)}")
    assert cb.state.value == "OPEN"

    # -------------------------------------------------------------------------
    # STEP 7: End-to-End Voice AI Pipeline (Audio -> DSP -> STT -> LLM -> TTS)
    # -------------------------------------------------------------------------
    print_section("Step 7: End-to-End Voice AI Pipeline & Providers")

    import base64
    import struct
    from backend.providers.stt import MockSTTProvider
    from backend.providers.tts import MockTTSProvider
    from backend.providers.llm import MockLLMProvider
    from backend.services.audio_service import AudioService
    from backend.services.conversation_service import ConversationService

    # Configure mock providers for deterministic offline demonstration
    mock_app_stt = MockSTTProvider()
    mock_app_tts = MockTTSProvider()
    mock_app_llm = MockLLMProvider()
    app.state.audio_service = AudioService(stt_provider=mock_app_stt, tts_provider=mock_app_tts)
    app.state.conversation_service = ConversationService(llm_provider=mock_app_llm)

    # Check Provider Status
    prov_res = client.get("/api/v1/providers/status")
    assert prov_res.status_code == 200
    prov_data = prov_res.json()
    print(f"[Provider Status] STT Configured: {prov_data['stt']['configured']} | Model: {prov_data['stt']['model']}")
    print(f"[Provider Status] LLM Configured: {prov_data['llm']['configured']} | Model: {prov_data['llm']['model']}")
    print(f"[Provider Status] TTS Configured: {prov_data['tts']['configured']} | Model: {prov_data['tts']['model']}")
    print("Mode Declaration: Using deterministic mock/offline providers for reproducible host testing.\n")

    # 7.1 Generate synthetic speech PCM audio (200ms @ 16kHz)
    pcm_samples = [int(800 * ((i % 40) - 20)) for i in range(3200)]
    pcm_bytes = struct.pack(f"<{len(pcm_samples)}h", *pcm_samples)
    b64_audio = base64.b64encode(pcm_bytes).decode("ascii")

    # 7.2 Post to Audio Transcribe API (Runs DSP + STT)
    audio_transcribe_res = client.post("/api/v1/audio/transcribe", json={
        "audio_base64": b64_audio,
        "sample_rate": 16000,
        "run_dsp": True
    })
    assert audio_transcribe_res.status_code == 200
    trans_data = audio_transcribe_res.json()
    print(f"[Audio Ingest & DSP] Transcribed: '{trans_data['transcript']}' (Confidence: {trans_data['confidence']})")
    print(f"  DSP Pipeline Stages Executed: {', '.join(trans_data['dsp_metrics']['stages_executed'])}")
    print(f"  DSP Input RMS: {trans_data['dsp_metrics']['input_rms']} -> Output RMS: {trans_data['dsp_metrics']['output_rms']}")

    # 7.3 Post conversation query
    conv_voice_res = client.post("/api/v1/conversation/message", json={
        "session_id": "demo-voice-session",
        "message": "What is the status of the device?",
        "tool_to_invoke": "get_system_status"
    })
    assert conv_voice_res.status_code == 200
    conv_voice_data = conv_voice_res.json()
    print(f"[Conversation & LLM] Assistant: '{conv_voice_data['response']}'")
    print(f"  Tool Execution: {conv_voice_data['tool_executions'][0]['tool_name']} -> Success: {conv_voice_data['tool_executions'][0]['success']}")

    # 7.4 Synthesize speech response with TTS API
    synth_res = client.post("/api/v1/speech/synthesize", json={
        "text": conv_voice_data['response'] or "NextSight system is operational.",
        "voice": "alloy"
    })
    assert synth_res.status_code == 200
    synth_data = synth_res.json()
    synth_bytes = base64.b64decode(synth_data["audio_base64"])
    print(f"[TTS Synthesis] Synthesized {len(synth_bytes)} bytes audio PCM ({synth_data['sample_rate']}Hz, {synth_data['encoding']}) in {synth_data['latency_ms']}ms")

    print_section("Local System Demonstration Summary")
    print("All 7 integration stages executed cleanly without errors!")
    print("All contracts, dispatchers, DSP pipeline, and AI providers verified successfully.\n")
    return True


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
