import os
import sys
import json
import struct
import time
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app import create_app
from backend.config.settings import AppSettings
from backend.protocol.contracts import PacketType
from backend.protocol.framing import ProtocolFraming, HEADER_SIZE
from backend.protocol.xiaozhi_protocol import (
    XiaozhiProtocol,
    XiaozhiMessageType,
    DecodedXiaozhiPacket,
    BP2_HEADER_SIZE,
    BP3_HEADER_SIZE,
)
from backend.providers import MockSTTProvider, MockTTSProvider, MockLLMProvider
from backend.services.audio_service import AudioService
from backend.services.conversation_service import ConversationService


# ============================================================================
# 1. Xiaozhi Binary Protocol 2 & 3 Golden Vector Tests
# ============================================================================

def test_binary_protocol_2_encode_decode_roundtrip():
    payload = b"test_audio_pcm_stream_data_12345"
    timestamp = 12345678
    encoded = XiaozhiProtocol.encode_bp2(
        payload=payload,
        message_type=XiaozhiMessageType.AUDIO_STREAM,
        timestamp_ms=timestamp,
        version=1
    )

    assert len(encoded) == BP2_HEADER_SIZE + len(payload)
    decoded = XiaozhiProtocol.decode_packet(encoded)

    assert decoded.is_valid is True
    assert decoded.protocol_version == 2
    assert decoded.message_type == XiaozhiMessageType.AUDIO_STREAM
    assert decoded.timestamp_ms == timestamp
    assert decoded.payload == payload
    assert decoded.json_data is None


def test_binary_protocol_2_json_control_packet():
    json_dict = {"type": "listen", "state": "start", "mode": "realtime"}
    json_bytes = json.dumps(json_dict).encode("utf-8")
    encoded = XiaozhiProtocol.encode_bp2(
        payload=json_bytes,
        message_type=XiaozhiMessageType.JSON_CONTROL,
        timestamp_ms=999
    )

    decoded = XiaozhiProtocol.decode_packet(encoded)
    assert decoded.is_valid is True
    assert decoded.protocol_version == 2
    assert decoded.message_type == XiaozhiMessageType.JSON_CONTROL
    assert decoded.json_data == json_dict


def test_binary_protocol_3_encode_decode_roundtrip():
    payload = b"pcm_samples_s16le"
    encoded = XiaozhiProtocol.encode_bp3(
        payload=payload,
        message_type=XiaozhiMessageType.AUDIO_STREAM
    )

    assert len(encoded) == BP3_HEADER_SIZE + len(payload)
    decoded = XiaozhiProtocol.decode_packet(encoded)

    assert decoded.is_valid is True
    assert decoded.protocol_version == 3
    assert decoded.message_type == XiaozhiMessageType.AUDIO_STREAM
    assert decoded.payload == payload


def test_xiaozhi_protocol_malformed_and_truncated_packets():
    # Truncated packet
    truncated = b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x20\x00\x00\x00"  # expects 32 bytes payload, got 0
    decoded = XiaozhiProtocol.decode_packet(truncated)
    assert decoded.is_valid is False
    assert "mismatch" in decoded.error_detail.lower()

    # Empty bytes
    decoded_empty = XiaozhiProtocol.decode_packet(b"")
    assert decoded_empty.is_valid is False


# ============================================================================
# 2. NextSight 0xAA55 Framing vs Xiaozhi Wire Protocol Coexistence
# ============================================================================

def test_protocol_cross_identification():
    ns_payload = b"nextsight_pcm_data"
    ns_framed = ProtocolFraming.encode_message(
        packet_type=PacketType.AUDIO_PCM,
        payload=ns_payload,
        sequence_number=42,
        timestamp_ms=5000
    )

    # NextSight packet has 0xAA55 magic
    assert ns_framed[:2] == b"\xAA\x55"
    decoded_ns = ProtocolFraming.decode_message(ns_framed)
    assert decoded_ns.is_valid is True
    assert decoded_ns.header.packet_type == PacketType.AUDIO_PCM
    assert decoded_ns.payload == ns_payload

    # Xiaozhi packet has version 1 (0x0001) in little-endian (<H)
    xz_encoded = XiaozhiProtocol.encode_bp2(payload=b"xz_pcm", message_type=XiaozhiMessageType.AUDIO_STREAM)
    assert xz_encoded[:2] != b"\xAA\x55"
    decoded_xz = XiaozhiProtocol.decode_packet(xz_encoded)
    assert decoded_xz.is_valid is True


# ============================================================================
# 3. WebSocket Device Streaming Endpoint Integration Tests
# ============================================================================

@pytest.fixture
def mock_ws_client():
    settings = AppSettings(
        environment="test",
        stt_provider="mock",
        tts_provider="mock",
        llm_provider="mock",
        require_verified_telemetry=False
    )
    app = create_app(settings)
    
    # Configure deterministic mock services
    app.state.audio_service = AudioService(
        stt_provider=MockSTTProvider(canned_transcription="Hello NextSight from ESP32", confidence=0.99),
        tts_provider=MockTTSProvider(),
        settings=settings
    )
    app.state.conversation_service = ConversationService(
        llm_provider=MockLLMProvider(fixed_response="NextSight Assistant is active."),
    )

    client = TestClient(app)
    return client


def test_websocket_device_handshake_and_ping(mock_ws_client):
    with mock_ws_client.websocket_connect("/ws/device") as ws:
        # 1. Receive server hello
        raw_hello = ws.receive_text()
        hello_data = json.loads(raw_hello)
        assert hello_data["type"] == "hello"
        assert hello_data["version"] == 3
        assert "session_id" in hello_data
        assert hello_data["audio_params"]["sample_rate"] == 16000

        # 2. Send ping
        ws.send_text(json.dumps({"type": "ping"}))
        pong_raw = ws.receive_text()
        pong_data = json.loads(pong_raw)
        assert pong_data["type"] == "pong"
        assert "timestamp" in pong_data


def test_websocket_device_full_voice_ai_turn(mock_ws_client):
    with mock_ws_client.websocket_connect("/api/v1/ws/device") as ws:
        # Server hello
        _ = ws.receive_text()

        # Start listening
        ws.send_text(json.dumps({"type": "listen", "state": "start"}))
        state_msg = json.loads(ws.receive_text())
        assert state_msg["type"] == "state"
        assert state_msg["state"] == "listening"

        # Stream audio chunks using Xiaozhi BinaryProtocol2
        pcm_chunk = b"\x00" * 3200  # 100ms of 16kHz audio
        bp2_frame = XiaozhiProtocol.encode_bp2(
            payload=pcm_chunk,
            message_type=XiaozhiMessageType.AUDIO_STREAM,
            timestamp_ms=100
        )
        ws.send_bytes(bp2_frame)

        # Stop listening to trigger transcription & conversation turn
        ws.send_text(json.dumps({"type": "listen", "state": "stop"}))

        # Expect processing state
        proc_msg = json.loads(ws.receive_text())
        assert proc_msg["type"] == "state"
        assert proc_msg["state"] == "processing"

        # Expect STT transcript
        stt_msg = json.loads(ws.receive_text())
        assert stt_msg["type"] == "stt"
        assert "Hello NextSight from ESP32" in stt_msg["text"]

        # Expect LLM assistant response
        llm_msg = json.loads(ws.receive_text())
        assert llm_msg["type"] == "llm"
        assert "NextSight Assistant is active." in llm_msg["text"]

        # Expect binary audio frame from TTS
        audio_frame_bytes = ws.receive_bytes()
        assert len(audio_frame_bytes) > BP2_HEADER_SIZE
        decoded_tts_audio = XiaozhiProtocol.decode_packet(audio_frame_bytes)
        assert decoded_tts_audio.is_valid is True
        assert decoded_tts_audio.message_type == XiaozhiMessageType.AUDIO_STREAM
        assert len(decoded_tts_audio.payload) > 0

        # Expect TTS completion state
        tts_state = json.loads(ws.receive_text())
        assert tts_state["type"] == "tts"
        assert tts_state["state"] == "stop"


def test_websocket_device_abort_speaking(mock_ws_client):
    with mock_ws_client.websocket_connect("/ws/device") as ws:
        _ = ws.receive_text()

        # Send abort command
        ws.send_text(json.dumps({"type": "abort"}))
        abort_res = json.loads(ws.receive_text())
        assert abort_res["type"] == "state"
        assert abort_res["state"] == "idle"
        assert abort_res["reason"] == "aborted"


def test_websocket_device_vision_request(mock_ws_client):
    with mock_ws_client.websocket_connect("/ws/device") as ws:
        _ = ws.receive_text()

        # Request vision processing
        ws.send_text(json.dumps({"type": "vision", "use_mock_frame": True}))
        vision_msg = json.loads(ws.receive_text())
        assert vision_msg["type"] == "vision_result"
        assert vision_msg["success"] is True
