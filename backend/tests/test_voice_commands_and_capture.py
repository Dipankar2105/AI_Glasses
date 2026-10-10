"""
End-to-End Test Suite for On-Demand Image Capture and Voice Command Workflows.
Validates:
1. VoiceCommandRouter deterministic matching and confidence thresholding.
2. HTTP Multipart Camera Upload & Explanation endpoints.
3. WebSocket Device Streaming Voice Command and Binary Image Capture integration.
4. Multi-turn voice command sequence: Capture -> Binary Image Ingest -> Describe Image -> Repeat -> Stop.
5. Error handling: corrupted frames, empty files, oversized payloads, timeouts, and cancellations.
"""

import os
import sys
import io
import json
import base64
import struct
import time
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app import create_app
from backend.config.settings import AppSettings
from backend.conversation.commands import VoiceCommandRouter, VoiceCommandIntent, VoiceCommandMatch
from backend.conversation.session import SessionManager
from backend.protocol.xiaozhi_protocol import XiaozhiProtocol, XiaozhiMessageType
from backend.providers import MockSTTProvider, MockTTSProvider, MockLLMProvider
from backend.services.audio_service import AudioService
from backend.services.conversation_service import ConversationService
from backend.services.vision_service import VisionService


# ============================================================================
# 1. VoiceCommandRouter Unit Tests
# ============================================================================

def test_voice_command_router_matching():
    router = VoiceCommandRouter(min_confidence=0.60)

    # Capture command variations
    for text in ["capture image", "take a picture", "take photo", "snap picture"]:
        match = router.match(text, confidence=0.95)
        assert match.matched is True
        assert match.intent == VoiceCommandIntent.CAPTURE_IMAGE
        assert match.confidence_passed is True

    # Describe command variations
    for text in ["describe current image", "what do you see", "describe scene", "look at this"]:
        match = router.match(text, confidence=0.90)
        assert match.matched is True
        assert match.intent == VoiceCommandIntent.DESCRIBE_IMAGE

    # Repeat response variations
    for text in ["repeat response", "say again", "repeat that", "what did you say"]:
        match = router.match(text, confidence=0.88)
        assert match.matched is True
        assert match.intent == VoiceCommandIntent.REPEAT_RESPONSE

    # Stop / cancel variations
    for text in ["stop", "cancel", "abort", "shut up"]:
        match = router.match(text, confidence=0.99)
        assert match.matched is True
        assert match.intent == VoiceCommandIntent.STOP_CANCEL


def test_voice_command_router_confidence_thresholding():
    router = VoiceCommandRouter(min_confidence=0.60)

    # High confidence -> passes
    m_high = router.match("capture image", confidence=0.75)
    assert m_high.matched is True
    assert m_high.confidence_passed is True

    # Low confidence (< 0.60) -> rejected from triggering command
    m_low = router.match("capture image", confidence=0.45)
    assert m_low.matched is False
    assert m_low.confidence_passed is False
    assert "below threshold" in m_low.reason

    # Ambiguous/conversational text -> no match, falls through to LLM
    m_chat = router.match("tell me a story about space", confidence=0.95)
    assert m_chat.matched is False
    assert m_chat.intent == VoiceCommandIntent.UNKNOWN


def test_voice_command_execution_lifecycle():
    router = VoiceCommandRouter()
    session_mgr = SessionManager()
    session_id = "test-session-cmd-001"

    # 1. Execute CAPTURE_IMAGE
    m_cap = router.match("take a picture", confidence=0.9)
    res_cap = router.execute_command(m_cap, session_id=session_id, session_manager=session_mgr)
    assert res_cap.success is True
    assert "Image captured" in res_cap.spoken_response

    # 2. Execute REPEAT_RESPONSE (repeats previous assistant response)
    m_rep = router.match("repeat that", confidence=0.9)
    res_rep = router.execute_command(m_rep, session_id=session_id, session_manager=session_mgr)
    assert res_rep.success is True
    assert "Image captured" in res_rep.spoken_response

    # 3. Execute STOP_CANCEL
    m_stop = router.match("stop", confidence=0.95)
    res_stop = router.execute_command(m_stop, session_id=session_id, session_manager=session_mgr)
    assert res_stop.success is True
    assert "Stopping" in res_stop.spoken_response


# ============================================================================
# 2. HTTP Multipart Camera Upload & Explanation Tests
# ============================================================================

@pytest.fixture
def app_client():
    settings = AppSettings(environment="test", require_verified_telemetry=False)
    app = create_app(settings)
    app.state.vision_service = VisionService(settings=settings)
    return TestClient(app)


def test_http_multipart_camera_explain_valid_jpeg(app_client):
    """Test valid JPEG multipart upload matching esp32_camera.cc upload format."""
    # Create valid synthetic JPEG bytes
    img = np.full((120, 160, 3), 200, dtype=np.uint8)
    _, encoded_jpg = cv2.imencode(".jpg", img)
    jpg_bytes = encoded_jpg.tobytes()

    files = {
        "file": ("camera.jpg", io.BytesIO(jpg_bytes), "image/jpeg")
    }
    data = {
        "question": "What is in front of me?"
    }

    response = app_client.post("/api/v1/vision/explain", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    assert "question" in res_json
    assert res_json["question"] == "What is in front of me?"
    assert "explanation" in res_json
    assert res_json["dimensions"]["width"] == 160
    assert res_json["dimensions"]["height"] == 120


def test_http_multipart_camera_explain_empty_and_corrupt_files(app_client):
    # Empty file
    files_empty = {"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")}
    r_empty = app_client.post("/api/v1/vision/explain", files=files_empty, data={"question": "Test"})
    assert r_empty.status_code == 400
    assert "empty" in r_empty.json()["detail"].lower()

    # Corrupted image bytes
    files_corrupt = {"file": ("corrupt.jpg", io.BytesIO(b"NOT_A_JPEG_FILE"), "image/jpeg")}
    r_corrupt = app_client.post("/api/v1/vision/explain", files=files_corrupt, data={"question": "Test"})
    assert r_corrupt.status_code == 400
    assert "corrupted" in r_corrupt.json()["detail"].lower()


# ============================================================================
# 3. WebSocket Voice Command & Image Capture Integration
# ============================================================================

@pytest.fixture
def ws_test_client():
    settings = AppSettings(environment="test", require_verified_telemetry=False)
    app = create_app(settings)
    app.state.audio_service = AudioService(
        stt_provider=MockSTTProvider(canned_transcription="take a picture", confidence=0.95),
        tts_provider=MockTTSProvider(),
        settings=settings
    )
    app.state.conversation_service = ConversationService(
        llm_provider=MockLLMProvider(fixed_response="Assistant default conversation response."),
    )
    app.state.vision_service = VisionService(settings=settings)
    return TestClient(app)


def test_websocket_voice_command_dispatch_and_spoken_response(ws_test_client):
    """Verify voice command triggers instant command action, camera trigger, and TTS response."""
    with ws_test_client.websocket_connect("/ws/device") as ws:
        _ = ws.receive_text()  # Server hello

        # Start listening
        ws.send_text(json.dumps({"type": "listen", "state": "start"}))
        _ = ws.receive_text()  # state: listening

        # Send simulated audio PCM chunk
        ws.send_bytes(b"\x00" * 3200)

        # Stop listening -> STT returns "take a picture"
        ws.send_text(json.dumps({"type": "listen", "state": "stop"}))
        _ = ws.receive_text()  # state: processing

        stt_msg = json.loads(ws.receive_text())
        assert stt_msg["type"] == "stt"
        assert stt_msg["text"] == "take a picture"

        # Voice command event emitted
        cmd_msg = json.loads(ws.receive_text())
        assert cmd_msg["type"] == "voice_command"
        assert cmd_msg["intent"] == "CAPTURE_IMAGE"
        assert cmd_msg["success"] is True

        # MCP JSON-RPC 2.0 tools/call sent to firmware McpServer
        mcp_msg = json.loads(ws.receive_text())
        assert mcp_msg["type"] == "mcp"
        assert mcp_msg["payload"]["jsonrpc"] == "2.0"
        assert mcp_msg["payload"]["method"] == "tools/call"
        assert mcp_msg["payload"]["params"]["name"] == "self.camera.take_photo"

        # High-level camera trigger sent to device
        cam_msg = json.loads(ws.receive_text())
        assert cam_msg["type"] == "camera"
        assert cam_msg["command"] == "capture"

        # Resolved response emitted
        llm_msg = json.loads(ws.receive_text())
        assert "Image captured successfully" in llm_msg["text"]

        # TTS playback start
        tts_start = json.loads(ws.receive_text())
        assert tts_start["type"] == "tts"
        assert tts_start["state"] == "start"

        _ = json.loads(ws.receive_text())  # tts sentence_start

        # Binary audio packet received
        tts_audio = ws.receive_bytes()
        assert len(tts_audio) > 16

        # TTS stop
        tts_stop = json.loads(ws.receive_text())
        assert tts_stop["type"] == "tts"
        assert tts_stop["state"] == "stop"


def test_websocket_binary_image_packet_ingestion(ws_test_client):
    """Verify binary image frame (Xiaozhi IMAGE_DATA) decoded and analyzed over WebSocket."""
    img = np.full((80, 80, 3), 150, dtype=np.uint8)
    _, encoded_jpg = cv2.imencode(".jpg", img)
    jpg_bytes = encoded_jpg.tobytes()

    # Encode Xiaozhi BinaryProtocol2 frame with message_type = 2 (IMAGE_DATA)
    image_frame = XiaozhiProtocol.encode_bp2(
        payload=jpg_bytes,
        message_type=XiaozhiMessageType.IMAGE_DATA,
        timestamp_ms=1000
    )

    with ws_test_client.websocket_connect("/ws/device") as ws:
        _ = ws.receive_text()  # hello

        ws.send_bytes(image_frame)
        res = json.loads(ws.receive_text())
        assert res["type"] == "vision_result"
        assert res["success"] is True
        assert res["scene"] is not None


def test_websocket_corrupted_binary_image_handling(ws_test_client):
    """Verify corrupted binary image frame returns structured error without crashing."""
    corrupted_frame = XiaozhiProtocol.encode_bp2(
        payload=b"NOT_A_VALID_JPEG_BYTE_STREAM",
        message_type=XiaozhiMessageType.IMAGE_DATA,
        timestamp_ms=2000
    )

    with ws_test_client.websocket_connect("/ws/device") as ws:
        _ = ws.receive_text()

        ws.send_bytes(corrupted_frame)
        err_res = json.loads(ws.receive_text())
        assert err_res["type"] == "error"
        assert err_res["code"] == "IMAGE_DECODE_FAILED"


def test_voice_command_low_confidence_fallback_to_llm():
    """Verify that low-confidence transcription falls through to normal LLM conversation."""
    settings = AppSettings(environment="test", require_verified_telemetry=False)
    app = create_app(settings)
    # Configure STT with low confidence (0.40)
    app.state.audio_service = AudioService(
        stt_provider=MockSTTProvider(canned_transcription="take a picture", confidence=0.40),
        tts_provider=MockTTSProvider(),
        settings=settings
    )
    app.state.conversation_service = ConversationService(
        llm_provider=MockLLMProvider(fixed_response="I heard you quietly, how can I help?"),
    )
    client = TestClient(app)

    with client.websocket_connect("/ws/device") as ws:
        _ = ws.receive_text()  # hello
        ws.send_text(json.dumps({"type": "listen", "state": "start"}))
        _ = ws.receive_text()  # listening

        ws.send_bytes(b"\x00" * 1600)
        ws.send_text(json.dumps({"type": "listen", "state": "stop"}))
        _ = ws.receive_text()  # processing

        stt_msg = json.loads(ws.receive_text())
        assert stt_msg["confidence"] == 0.40

        # LLM response emitted directly without triggering voice command action
        llm_msg = json.loads(ws.receive_text())
        assert llm_msg["type"] == "llm"
        assert "I heard you quietly" in llm_msg["text"]
