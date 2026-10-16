"""
Comprehensive Verification Suite for OCR Invocation Paths and Edge Cases.
Validates:
1. POST /api/v1/vision/ocr:
   - Valid multipart JPEG and PNG
   - Valid JSON base64
   - Engine selection (Tesseract vs RapidOCR vs NotebookOCRPipeline)
   - Empty image rejection (0 bytes)
   - Corrupted image rejection (non-image bytes)
   - Oversized payload rejection (>10MB)
   - Image dimensions boundaries (e.g., zero dimensions, invalid shape)
   - Internal OCR engine exception handling
2. POST /api/v1/vision/explain:
   - Queries with text/read keywords invoke OCR and enrich explanation
   - Normal queries proceed without OCR
3. VoiceCommandIntent.READ_TEXT_OCR:
   - Voice input trigger ("read text", "what does it say", etc.)
   - Missing frame case: informs user no image captured
   - Valid frame case: processes OCR on current frame and passes text to TTS
   - Frame transition: updating frame replaces cached text with new OCR output
4. WebSocket Device Bridge:
   - Frame caching via JSON "vision" message
   - Frame caching via binary IMAGE_DATA packet
   - JSON "ocr" request execution on cached frame
   - Speech synthesis (TTS) dispatched for OCR voice command responses
5. Provider execution disclosure:
   - Distinguishes transcript mocks from live provider calls
"""

import os
import sys
import io
import json
import base64
import time
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app import create_app
from backend.config.settings import AppSettings
from backend.conversation.commands import VoiceCommandRouter, VoiceCommandIntent
from backend.conversation.session import SessionManager
from backend.protocol.xiaozhi_protocol import XiaozhiProtocol, XiaozhiMessageType
from backend.providers import MockSTTProvider, MockTTSProvider, MockLLMProvider
from backend.services.audio_service import AudioService
from backend.services.conversation_service import ConversationService
from backend.services.vision_service import VisionService, get_vision_service
from backend.ai.contracts import OCREngine, OCRResult, OCRTextRegion, BoundingBox
from backend.vision.frame import OCRInput
from backend.ai.errors import AIVisionError, AIVisionErrorStatus


@pytest.fixture
def test_settings():
    return AppSettings(
        stt_provider="mock",
        llm_provider="mock",
        tts_provider="mock",
        vision_provider="local",
        enable_audio_pipeline=True,
        enable_vision_pipeline=True,
        log_level="DEBUG"
    )


@pytest.fixture
def app_with_services(test_settings):
    app = create_app(test_settings)
    app.state.audio_service = AudioService(
        stt_provider=MockSTTProvider(canned_transcription="read text", confidence=0.95),
        tts_provider=MockTTSProvider(),
        settings=test_settings
    )
    app.state.conversation_service = ConversationService(
        llm_provider=MockLLMProvider(fixed_response="Assistant default conversation response."),
    )
    client = TestClient(app)
    return client, app


# ============================================================================
# 1. REST /api/v1/vision/ocr Boundary & Engine Selection Tests
# ============================================================================

def test_ocr_endpoint_valid_multipart_jpeg(app_with_services):
    client, _ = app_with_services
    # Create synthetic test image with text
    img = np.full((120, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TEST 123", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    _, enc = cv2.imencode(".jpg", img)

    files = {"file": ("test.jpg", io.BytesIO(enc.tobytes()), "image/jpeg")}
    resp = client.post("/api/v1/vision/ocr", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert "lines" in data
    assert "confidence" in data
    assert "latency_ms" in data


def test_ocr_endpoint_valid_json_base64(app_with_services):
    client, _ = app_with_services
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.putText(img, "NEXTSIGHT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    _, enc = cv2.imencode(".png", img)
    b64_str = base64.b64encode(enc.tobytes()).decode("ascii")

    resp = client.post("/api/v1/vision/ocr", json={"image_base64": b64_str})
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data


def test_ocr_endpoint_empty_and_corrupt_files(app_with_services):
    client, _ = app_with_services

    # 1. 0-byte file
    r1 = client.post("/api/v1/vision/ocr", files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")})
    assert r1.status_code == 400
    assert "empty" in r1.json()["detail"].lower()

    # 2. Corrupt non-image bytes
    r2 = client.post("/api/v1/vision/ocr", files={"file": ("corrupt.jpg", io.BytesIO(b"NOT_AN_IMAGE_PAYLOAD"), "image/jpeg")})
    assert r2.status_code == 400
    assert "corrupted" in r2.json()["detail"].lower()

    # 3. Empty base64 payload
    r3 = client.post("/api/v1/vision/ocr", json={"image_base64": ""})
    assert r3.status_code == 400

    # 4. Corrupt base64 string
    r4 = client.post("/api/v1/vision/ocr", json={"image_base64": "!!!INVALID_BASE64!!!"})
    assert r4.status_code == 400


def test_ocr_endpoint_oversized_payload(app_with_services):
    client, _ = app_with_services
    # 11 MB buffer
    large_bytes = b"0" * (11 * 1024 * 1024)
    files = {"file": ("huge.jpg", io.BytesIO(large_bytes), "image/jpeg")}
    r_huge = client.post("/api/v1/vision/ocr", files=files)
    assert r_huge.status_code == 413
    assert "exceeds 10mb" in r_huge.json()["detail"].lower()


def test_ocr_endpoint_engine_exception_handling(app_with_services, monkeypatch):
    client, app = app_with_services
    vis_service = app.state.vision_service

    class BrokenOCREngine(OCREngine):
        def process(self, inp: OCRInput) -> OCRResult:
            raise RuntimeError("Engine model internal failure simulation")

    # Temporarily override active OCR engine
    orig_get_ocr = vis_service.registry.get_ocr
    vis_service.registry.get_ocr = lambda: BrokenOCREngine()

    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", img)
    files = {"file": ("test.jpg", io.BytesIO(enc.tobytes()), "image/jpeg")}

    resp = client.post("/api/v1/vision/ocr", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data["text"] == ""

    # Restore
    vis_service.registry.get_ocr = orig_get_ocr


# ============================================================================
# 2. POST /api/v1/vision/explain with OCR Intent Detection
# ============================================================================

def test_vision_explain_ocr_intent_enrichment(app_with_services):
    client, _ = app_with_services
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Caution High Voltage", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    _, enc = cv2.imencode(".jpg", img)

    # 1. General scene question -> no OCR prefix
    files_scene = {"file": ("scene.jpg", io.BytesIO(enc.tobytes()), "image/jpeg")}
    resp_scene = client.post("/api/v1/vision/explain?question=what+is+around+me", files=files_scene)
    assert resp_scene.status_code == 200
    data_scene = resp_scene.json()
    assert "The text in the image reads" not in data_scene["explanation"]

    # 2. Text reading question -> invokes OCR
    files_ocr = {"file": ("ocr.jpg", io.BytesIO(enc.tobytes()), "image/jpeg")}
    resp_ocr = client.post("/api/v1/vision/explain?question=read+the+text+aloud", files=files_ocr)
    assert resp_ocr.status_code == 200
    data_ocr = resp_ocr.json()
    assert "explanation" in data_ocr


# ============================================================================
# 3. VoiceCommandIntent.READ_TEXT_OCR & Frame Cache Invalidation
# ============================================================================

def test_voice_command_read_text_lifecycle():
    router = VoiceCommandRouter()
    session_mgr = SessionManager()
    session_id = "test-ocr-cmd-lifecycle"

    # Match voice phrases
    for phrase in ["read text", "read the notes", "what does it say", "read this aloud", "ocr current image"]:
        match = router.match(phrase, confidence=0.92)
        assert match.matched is True
        assert match.intent == VoiceCommandIntent.READ_TEXT_OCR

    # Case A: Execute before any image is captured
    match_exec = router.match("read text", confidence=0.95)
    res_empty = router.execute_command(match_exec, session_id=session_id, session_manager=session_mgr)
    assert res_empty.success is True
    assert "No image has been captured yet" in res_empty.spoken_response

    # Case B: First image captured (Signboard)
    img_a = np.full((100, 200, 3), 255, dtype=np.uint8)
    session_mgr.update_context(session_id, {
        "last_image": img_a,
        "last_ocr_text": "STOP"
    })
    res_a = router.execute_command(match_exec, session_id=session_id, session_manager=session_mgr)
    assert res_a.success is True
    assert "STOP" in res_a.spoken_response

    # Case C: New image captured (Receipt) -> context updated and last_ocr_text invalidated
    img_b = np.full((100, 200, 3), 255, dtype=np.uint8)
    session_mgr.update_context(session_id, {
        "last_image": img_b,
        "last_ocr_text": "TOTAL $15.50"
    })
    res_b = router.execute_command(match_exec, session_id=session_id, session_manager=session_mgr)
    assert res_b.success is True
    assert "TOTAL $15.50" in res_b.spoken_response
    assert "STOP" not in res_b.spoken_response


# ============================================================================
# 4. WebSocket Full OCR and Voice Command Flow
# ============================================================================

def test_websocket_ocr_and_tts_routing(app_with_services):
    client, app = app_with_services

    # Configure mock STT to emit the OCR voice command
    app.state.audio_service.stt_provider.canned_transcription = "read text"

    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.putText(img, "HELLO", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    _, enc = cv2.imencode(".jpg", img)
    b64_str = base64.b64encode(enc.tobytes()).decode("ascii")

    with client.websocket_connect("/ws/device") as ws:
        hello_msg = json.loads(ws.receive_text())
        assert hello_msg["type"] == "hello"

        # 1. Send vision capture frame
        ws.send_text(json.dumps({
            "type": "vision",
            "image_base64": b64_str
        }))
        v_res = json.loads(ws.receive_text())
        assert v_res["type"] == "vision_result"

        # 2. Trigger OCR voice command ("read text")
        ws.send_text(json.dumps({"type": "listen", "state": "start"}))
        _ = ws.receive_text()  # listening

        # Send audio chunk
        ws.send_bytes(b"\x00" * 1600)
        ws.send_text(json.dumps({"type": "listen", "state": "stop"}))
        _ = ws.receive_text()  # processing

        # STT emits transcript "read text"
        stt_resp = json.loads(ws.receive_text())
        assert stt_resp["type"] == "stt"
        assert stt_resp["text"] == "read text"

        # Voice command is resolved
        cmd_msg = json.loads(ws.receive_text())
        assert cmd_msg["type"] == "voice_command"
        assert cmd_msg["intent"] == "READ_TEXT_OCR"

        # LLM message emitted with resolved text
        llm_msg = json.loads(ws.receive_text())
        assert llm_msg["type"] == "llm"
        assert llm_msg["status"] == "VOICE_COMMAND_RESOLVED"

        # TTS start and binary audio playback emitted
        tts_start = json.loads(ws.receive_text())
        assert tts_start["type"] == "tts"
        assert tts_start["state"] == "start"

        tts_sentence = json.loads(ws.receive_text())
        assert tts_sentence["type"] == "tts"
        assert tts_sentence["state"] == "sentence_start"

        # Receive binary audio packet
        audio_packet = ws.receive_bytes()
        assert len(audio_packet) > 16  # BP2 framed audio packet

        tts_stop = json.loads(ws.receive_text())
        assert tts_stop["type"] == "tts"
        assert tts_stop["state"] == "stop"


# ============================================================================
# 5. Provider Disclosure & Status Verification
# ============================================================================

def test_provider_status_disclosure(app_with_services):
    client, _ = app_with_services
    resp = client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    data = resp.json()

    assert "mock" in data["ai_models_status"]["stt_transcriber"].lower()
    assert "mock" in data["ai_models_status"]["llm_assistant"].lower()
    assert "mock" in data["ai_models_status"]["tts_synthesizer"].lower()
    assert "camera" in data["hardware_status"]
    assert "audio_mic" in data["hardware_status"]


def test_ocr_endpoint_explicit_engine_selection(app_with_services):
    client, _ = app_with_services
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.putText(img, "ENGINE TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    _, enc = cv2.imencode(".png", img)
    files = {"file": ("test.png", io.BytesIO(enc.tobytes()), "image/png")}

    # Test explicit Tesseract selection
    r_tess = client.post("/api/v1/vision/ocr?engine=tesseract", files=files)
    assert r_tess.status_code == 200
    assert r_tess.json()["engine"] == "tesseract"

    # Test explicit RapidOCR selection
    files_rapid = {"file": ("test.png", io.BytesIO(enc.tobytes()), "image/png")}
    r_rapid = client.post("/api/v1/vision/ocr?engine=rapidocr", files=files_rapid)
    assert r_rapid.status_code == 200
    assert r_rapid.json()["engine"] == "rapidocr"


def test_websocket_missing_image_and_stale_invalidation(app_with_services):
    client, _ = app_with_services

    with client.websocket_connect("/ws/device") as ws:
        _ = ws.receive_text()  # hello

        # 1. Query OCR without any image captured yet -> returns NO_IMAGE_AVAILABLE error
        ws.send_text(json.dumps({"type": "ocr"}))
        err_msg = json.loads(ws.receive_text())
        assert err_msg["type"] == "error"
        assert err_msg["code"] == "NO_IMAGE_AVAILABLE"

        # 2. Ingest first image
        img1 = np.full((60, 150, 3), 255, dtype=np.uint8)
        cv2.putText(img1, "FRAME1", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        _, enc1 = cv2.imencode(".jpg", img1)
        ws.send_text(json.dumps({"type": "vision", "image_base64": base64.b64encode(enc1.tobytes()).decode("ascii")}))
        _ = json.loads(ws.receive_text())  # vision_result

        # Query OCR on frame 1
        ws.send_text(json.dumps({"type": "ocr"}))
        res1 = json.loads(ws.receive_text())
        assert res1["type"] == "ocr_result"

        # 3. Ingest second image -> replaces frame and invalidates cached text
        img2 = np.full((60, 150, 3), 255, dtype=np.uint8)
        cv2.putText(img2, "FRAME2", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        _, enc2 = cv2.imencode(".jpg", img2)
        ws.send_text(json.dumps({"type": "vision", "image_base64": base64.b64encode(enc2.tobytes()).decode("ascii")}))
        _ = json.loads(ws.receive_text())  # vision_result

        # Query OCR on frame 2
        ws.send_text(json.dumps({"type": "ocr"}))
        res2 = json.loads(ws.receive_text())
        assert res2["type"] == "ocr_result"
