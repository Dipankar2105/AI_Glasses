"""Integration tests for Runtime Audio, Speech, Providers, and Conversation APIs."""

import base64
import struct
import io
import wave
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config.settings import AppSettings
from backend.providers.stt import MockSTTProvider
from backend.providers.tts import MockTTSProvider
from backend.providers.llm import MockLLMProvider
from backend.services.audio_service import AudioService, reset_audio_service
from backend.services.conversation_service import ConversationService, reset_conversation_service
from backend.conversation.session import SessionManager, reset_session_manager


def create_synthetic_pcm_bytes(sample_count: int = 1600, sample_rate: int = 16000) -> bytes:
    """Creates synthetic 16-bit PCM bytes (100ms at 16kHz)."""
    samples = [int(1000 * ((i % 50) - 25)) for i in range(sample_count)]
    return struct.pack(f"<{len(samples)}h", *samples)


def create_synthetic_wav_bytes(sample_count: int = 1600, sample_rate: int = 16000) -> bytes:
    """Creates synthetic WAV container bytes."""
    pcm = create_synthetic_pcm_bytes(sample_count, sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def cleanup_singletons():
    yield
    reset_audio_service()
    reset_conversation_service()
    reset_session_manager()


@pytest.fixture
def mock_app():
    settings = AppSettings(
        environment="test",
        stt_provider="mock",
        llm_provider="mock",
        tts_provider="mock",
        is_strict_telemetry_required=False
    )
    app = create_app(settings)
    stt = MockSTTProvider()
    tts = MockTTSProvider()
    llm = MockLLMProvider()
    app.state.audio_service = AudioService(stt_provider=stt, tts_provider=tts, settings=settings)
    app.state.conversation_service = ConversationService(llm_provider=llm)
    return app


@pytest.fixture
def client(mock_app):
    return TestClient(mock_app)


def test_capabilities_endpoint(client):
    res = client.get("/api/v1/capabilities")
    assert res.status_code == 200
    data = res.json()
    assert "integrated_audio_dsp_pipeline" in data["implemented"]
    assert "stt_speech_transcription" in data["implemented"]
    assert "tts_speech_audio_output" in data["implemented"]
    assert "llm_reasoning_and_conversation" in data["implemented"]
    assert "ocr_production_routing (ON_HOLD in Phase 5)" in data["deferred"]


def test_providers_status_endpoint(client):
    res = client.get("/api/v1/providers/status")
    assert res.status_code == 200
    data = res.json()
    assert data["stt"]["provider_type"] == "mock"
    assert data["stt"]["offline_mode"] is True
    assert data["stt"]["available"] is True
    assert data["llm"]["provider_type"] == "mock"
    assert data["tts"]["provider_type"] == "mock"


def test_audio_transcribe_pcm_success(client):
    pcm = create_synthetic_pcm_bytes(1600)
    b64_audio = base64.b64encode(pcm).decode("ascii")

    res = client.post("/api/v1/audio/transcribe", json={
        "audio_base64": b64_audio,
        "sample_rate": 16000,
        "channels": 1,
        "run_dsp": True
    })

    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert len(data["transcript"]) > 0
    assert data["confidence"] > 0.0
    assert data["dsp_metrics"] is not None
    assert "stages_executed" in data["dsp_metrics"]


def test_audio_transcribe_wav_success(client):
    wav_bytes = create_synthetic_wav_bytes(1600)
    b64_audio = base64.b64encode(wav_bytes).decode("ascii")

    res = client.post("/api/v1/audio/transcribe", json={
        "audio_base64": b64_audio,
        "sample_rate": 16000,
        "run_dsp": True
    })

    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["transcript"] != ""


def test_audio_transcribe_empty_payload(client):
    res = client.post("/api/v1/audio/transcribe", json={
        "audio_base64": "",
        "sample_rate": 16000
    })
    assert res.status_code == 400


def test_audio_transcribe_invalid_base64(client):
    res = client.post("/api/v1/audio/transcribe", json={
        "audio_base64": "!!!not-valid-base64@@@",
        "sample_rate": 16000
    })
    assert res.status_code == 400


def test_audio_transcribe_oversized_payload(client):
    # Simulate payload larger than limit
    huge_data = b"0" * (11 * 1024 * 1024)
    b64_huge = base64.b64encode(huge_data).decode("ascii")

    res = client.post("/api/v1/audio/transcribe", json={
        "audio_base64": b64_huge,
        "sample_rate": 16000
    })
    assert res.status_code == 413


def test_speech_synthesize_success(client):
    res = client.post("/api/v1/speech/synthesize", json={
        "text": "Hello, smart glasses user!",
        "voice": "alloy"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["sample_rate"] in (16000, 24000)
    assert data["encoding"] == "pcm_s16le"
    assert len(data["audio_base64"]) > 0

    # Verify decoded bytes
    decoded = base64.b64decode(data["audio_base64"])
    assert len(decoded) > 0


def test_speech_synthesize_empty_text(client):
    res = client.post("/api/v1/speech/synthesize", json={
        "text": "   "
    })
    assert res.status_code in (400, 422)


def test_speech_synthesize_null_provider():
    settings = AppSettings(
        environment="test",
        tts_provider="null"
    )
    app = create_app(settings)
    c = TestClient(app)

    res = c.post("/api/v1/speech/synthesize", json={
        "text": "Test speech"
    })
    assert res.status_code == 503
    assert "not configured" in res.json()["detail"].lower() or "deferred" in res.json()["detail"].lower()


def test_conversation_message_flow(client):
    # 1. Send conversation message
    res = client.post("/api/v1/conversation/message", json={
        "session_id": "test-session-123",
        "message": "What is the status of NextSight?"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "test-session-123"
    assert data["response"] != ""
    assert data["session_message_count"] == 2  # user + assistant

    # 2. Retrieve session context
    res_session = client.get("/api/v1/conversation/sessions/test-session-123")
    assert res_session.status_code == 200
    sess_data = res_session.json()
    assert sess_data["session_id"] == "test-session-123"
    assert sess_data["message_count"] == 2

    # 3. Delete session
    res_del = client.delete("/api/v1/conversation/sessions/test-session-123")
    assert res_del.status_code == 200

    # 4. Confirm deleted
    res_after = client.get("/api/v1/conversation/sessions/test-session-123")
    assert res_after.status_code == 404
