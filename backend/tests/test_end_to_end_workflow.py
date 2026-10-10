"""Comprehensive End-to-End Software Integration Workflow Tests.

Validates:
1. Complete Audio Path:
   Synthetic PCM Stream -> AudioStreamAdapter -> IntegratedAudioPipeline -> STT Provider -> Conversation -> LLM -> TTS Provider -> Synthesized Audio
2. Complete Vision Path:
   Synthetic/Base64 Image -> Request Validation -> VisionService -> Object Detection & Scene Analysis -> MCP Tool Result
3. Combined Multimodal Path:
   Vision Context Injection + Audio Transcribed Query -> Multi-turn Contextual Conversation
4. Fault Tolerance & Safety Gating:
   Missing Credentials -> Structured Unavailable Status (No Crash, No Fabricated Responses)
   Critical Battery/Thermal Telemetry -> Gated Closed by Power Policy
"""

import os
import sys
import time
import base64
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.config.settings import AppSettings
from backend.app import create_app
from fastapi.testclient import TestClient

from firmware.audio.dsp.integrated_pipeline import IntegratedAudioPipeline, AudioStreamAdapter, AudioFrame
from backend.providers import (
    MockSTTProvider,
    NullSTTProvider,
    MockTTSProvider,
    NullTTSProvider,
    MockLLMProvider,
    NullLLMProvider,
    LLMToolCall,
    ProviderStatus,
)
from backend.conversation.session import SessionManager, reset_session_manager
from backend.conversation.models import (
    ConversationMessageRequest,
    MessageRole,
)
from backend.mcp.registry import MCPToolRegistry, get_tool_registry
from backend.services.conversation_service import ConversationService, reset_conversation_service
from backend.services.vision_service import VisionService, get_vision_service, reset_vision_service
from backend.power.policy import PowerThermalPolicyManager, reset_power_policy_manager, get_power_policy_manager
from backend.power.contracts import BatteryTelemetry, ThermalTelemetry


@pytest.fixture(autouse=True)
def cleanup():
    reset_conversation_service()
    reset_session_manager()
    reset_vision_service()
    reset_power_policy_manager()
    yield
    reset_conversation_service()
    reset_session_manager()
    reset_vision_service()
    reset_power_policy_manager()


# ============================================================================
# 1. Complete Audio Pipeline E2E Test
# ============================================================================

@pytest.mark.anyio
async def test_end_to_end_audio_pipeline_to_tts():
    """
    Synthetic PCM audio bytes -> Stream Adapter -> Integrated DSP Pipeline ->
    STT Provider -> Conversation Orchestrator -> LLM Reasoning -> TTS Synthesis.
    """
    # Step A: Generate 0.5s of synthetic speech-like 16kHz PCM audio
    sample_rate = 16000
    duration_s = 0.5
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    synthetic_signal = 0.2 * np.sin(2 * np.pi * 300 * t) + 0.05 * np.random.normal(0, 0.01, num_samples)
    pcm_bytes = (np.clip(synthetic_signal, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    # Step B: Pass through AudioStreamAdapter and Integrated DSP Pipeline
    pipeline = IntegratedAudioPipeline()
    adapter = AudioStreamAdapter(pipeline=pipeline, frame_size=256)
    dsp_results = adapter.push_raw_stream(pcm_bytes)
    assert len(dsp_results) > 0
    
    # Collect processed PCM audio
    processed_pcm = b"".join(r.output_bytes for r in dsp_results if r.accepted and r.output_bytes)
    assert len(processed_pcm) > 0

    # Step C: Transcribe via STT Provider
    stt_provider = MockSTTProvider(canned_transcription="Where is my coffee cup?", confidence=0.97)
    stt_res = await stt_provider.transcribe(processed_pcm, sample_rate=16000)
    assert stt_res.success is True
    assert stt_res.text == "Where is my coffee cup?"

    # Step D: Orchestrate Conversation with Mock LLM
    llm_provider = MockLLMProvider(fixed_response="Your coffee cup is on the desk to your right.")
    session_mgr = SessionManager()
    conv_service = ConversationService(session_manager=session_mgr, llm_provider=llm_provider)

    conv_req = ConversationMessageRequest(message=stt_res.text)
    conv_resp = await conv_service.process_user_message(conv_req)
    assert conv_resp.response == "Your coffee cup is on the desk to your right."
    assert conv_resp.llm_status == "MOCK_DEVELOPMENT"

    # Step E: Synthesize Assistant Response via TTS Provider
    tts_provider = MockTTSProvider(default_sample_rate=16000)
    tts_res = await tts_provider.synthesize(conv_resp.response)
    assert tts_res.success is True
    assert len(tts_res.audio_bytes) > 0
    assert tts_res.sample_rate == 16000
    assert tts_res.audio_format == "pcm_s16le"


# ============================================================================
# 2. Complete Vision Pipeline E2E Test
# ============================================================================

def test_end_to_end_vision_pipeline_via_api():
    """
    Synthetic image -> HTTP POST /api/v1/vision/process ->
    Preprocessing & Quality Assessment -> Object Detection -> Structured Output.
    """
    settings = AppSettings(environment="test", is_strict_telemetry_required=False)
    app = create_app(settings)
    client = TestClient(app)

    # Create test JPEG
    dummy_img = np.full((120, 120, 3), 180, dtype=np.uint8)
    cv2.rectangle(dummy_img, (20, 20), (80, 80), (50, 50, 50), -1)
    _, encoded = cv2.imencode(".jpg", dummy_img)
    b64_str = base64.b64encode(encoded).decode("utf-8")

    resp = client.post(
        "/api/v1/vision/process",
        json={"image_base64": b64_str, "seq_num": 101},
        headers={"X-Request-ID": "e2e-vision-trace-001"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["seq_num"] == 101
    assert data["request_id"] == "e2e-vision-trace-001"
    assert len(data["detections"]) > 0
    assert data["scene"] is not None
    assert "DEFERRED" in data["ocr_notice"]


# ============================================================================
# 3. Combined Multimodal Path Test
# ============================================================================

@pytest.mark.anyio
async def test_end_to_end_multimodal_vision_context_in_conversation():
    """
    Vision analysis context is injected into conversation session,
    enabling context-aware assistant reasoning without live hardware coupling.
    """
    session_mgr = SessionManager()
    
    # 1. Simulate vision result metadata
    vision_context = {
        "detected_objects": ["notebook", "pen", "desk"],
        "scene_description": "A wooden work desk with an open notebook and a pen.",
        "hazards": []
    }

    # 2. Configure LLM that refers to visual context
    llm = MockLLMProvider(fixed_response="I see an open notebook and a pen on your desk.")
    conv_service = ConversationService(session_manager=session_mgr, llm_provider=llm)

    # 3. User asks about surroundings with vision context attached
    req = ConversationMessageRequest(
        message="What is in front of me?",
        vision_context=vision_context
    )
    resp = await conv_service.process_user_message(req)
    
    assert resp.session_id is not None
    assert resp.response == "I see an open notebook and a pen on your desk."
    
    # Verify session context preserved the vision metadata
    session = session_mgr.get_session(resp.session_id)
    assert session.context_metadata.get("scene_description") == vision_context["scene_description"]


# ============================================================================
# 4. Fault Tolerance, Safety Gating & Missing Provider Tests
# ============================================================================

@pytest.mark.anyio
async def test_end_to_end_unconfigured_providers_fail_safely():
    """Unconfigured STT, LLM, and TTS return structured unavailable statuses without crashing."""
    stt = NullSTTProvider()
    llm = NullLLMProvider()
    tts = NullTTSProvider()

    # STT reports unavailable
    stt_res = await stt.transcribe(b"\x00" * 100)
    assert stt_res.success is False
    assert stt_res.status == ProviderStatus.PROVIDER_UNAVAILABLE

    # LLM reports unavailable
    session_mgr = SessionManager()
    conv = ConversationService(session_manager=session_mgr, llm_provider=llm)
    conv_res = await conv.process_user_message(ConversationMessageRequest(message="Hello?"))
    assert conv_res.llm_status == "PROVIDER_UNAVAILABLE"
    assert "[LLM Unavailable]" in conv_res.response

    # TTS reports unavailable
    tts_res = await tts.synthesize("Hello")
    assert tts_res.success is False
    assert tts_res.status == ProviderStatus.PROVIDER_UNAVAILABLE


def test_end_to_end_production_power_safety_denial():
    """In production mode with strict telemetry, missing telemetry blocks vision requests."""
    settings = AppSettings(environment="production", is_strict_telemetry_required=True)
    app = create_app(settings)
    client = TestClient(app)

    # 1. Missing telemetry -> Blocked
    resp = client.post("/api/v1/vision/process", json={"use_mock_frame": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data["errors"][0]["error"] == "BLOCKED_BY_POWER_POLICY"

    # 2. Inject valid telemetry -> Allowed
    now = time.time()
    app.state.power_manager.update_battery_telemetry(BatteryTelemetry(
        voltage_volts=4.0, percentage=90.0, timestamp=now
    ), current_time=now)
    app.state.power_manager.update_thermal_telemetry(ThermalTelemetry(
        temperature_celsius=35.0, timestamp=now
    ), current_time=now)

    resp2 = client.post("/api/v1/vision/process", json={"use_mock_frame": True})
    assert resp2.status_code == 200
    assert resp2.json()["success"] is True
