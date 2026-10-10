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
from backend.services.vision_service import VisionService


# ============================================================================
# 1. Independent C-Struct Wire Format Vectors (Non-Circular Tests)
# ============================================================================

def test_independent_c_struct_bp2_wire_format():
    """
    Directly construct binary frames using raw struct.pack matching
    the ESP32-S3 C++ firmware BinaryHeader struct (websocket_protocol.cc):
    struct __attribute__((packed)) BinaryHeader {
        uint16_t version;     // 1 or 2
        uint16_t type;        // 0=Audio, 1=JSON
        uint32_t reserved1;   // 0
        uint32_t timestamp;   // ms
        uint32_t payload_size;
    };
    All in Network Byte Order (Big-Endian '>HHIII').
    """
    raw_pcm = b"\x10\x20\x30\x40" * 8  # 32 bytes
    version = 2
    msg_type = 0  # 0 = Audio in C++ firmware
    reserved = 0
    timestamp = 987654
    payload_size = len(raw_pcm)

    # Pack manually without using XiaozhiProtocol serializer
    raw_packet = struct.pack(">HHIII", version, msg_type, reserved, timestamp, payload_size) + raw_pcm

    assert len(raw_packet) == 16 + 32
    # Verify that backend XiaozhiProtocol correctly deserializes the exact C++ firmware frame
    decoded = XiaozhiProtocol.decode_packet(raw_packet)
    assert decoded.is_valid is True
    assert decoded.protocol_version == 2
    assert decoded.message_type == XiaozhiMessageType.AUDIO_STREAM
    assert decoded.timestamp_ms == timestamp
    assert decoded.payload == raw_pcm


def test_independent_c_struct_bp3_wire_format():
    """
    Directly construct binary frames matching ESP32-S3 C++ BinaryHeader3:
    struct __attribute__((packed)) BinaryHeader3 {
        uint8_t type;         // 0=Audio
        uint8_t reserved;
        uint16_t payload_size;
    };
    All in Network Byte Order (Big-Endian '>BBH').
    """
    raw_pcm = b"\x01\x02\x03\x04" * 4  # 16 bytes
    msg_type = 0  # 0 = Audio in C++ firmware
    reserved = 0
    payload_size = len(raw_pcm)

    raw_packet = struct.pack(">BBH", msg_type, reserved, payload_size) + raw_pcm
    assert len(raw_packet) == 4 + 16

    decoded = XiaozhiProtocol.decode_packet(raw_packet)
    assert decoded.is_valid is True
    assert decoded.protocol_version == 3
    assert decoded.message_type == XiaozhiMessageType.AUDIO_STREAM
    assert decoded.payload == raw_pcm


def test_backend_response_decodable_by_independent_c_unpacker():
    """
    Verify that backend-generated binary frames can be unpacked using raw
    firmware C-struct unpack logic without helper classes.
    """
    test_audio = b"\xAA\xBB\xCC\xDD" * 10  # 40 bytes
    ts = 55443322

    encoded = XiaozhiProtocol.encode_bp2(
        payload=test_audio,
        message_type=XiaozhiMessageType.AUDIO_STREAM,
        timestamp_ms=ts
    )

    # Independent unpack matching C++ ntohs / ntohl logic
    header_format = ">HHIII"
    version, msg_type, reserved, timestamp, payload_size = struct.unpack(header_format, encoded[:16])
    payload = encoded[16:16 + payload_size]

    assert version in (1, 2)
    assert msg_type == 0  # 0 = Audio stream
    assert timestamp == ts
    assert payload_size == len(test_audio)
    assert payload == test_audio


# ============================================================================
# 2. Malformed, Truncated, Incorrect Lengths, and Invalid Message Types
# ============================================================================

def test_xiaozhi_protocol_partial_and_malformed_packets():
    # Incomplete header (< 16 bytes for BP2, < 4 bytes for BP3)
    assert XiaozhiProtocol.decode_packet(b"\x00\x01").is_valid is False

    # Header claims 64 bytes, but only 10 bytes present
    incomplete_payload = struct.pack(">HHIII", 1, 0, 0, 100, 64) + (b"\x00" * 10)
    decoded = XiaozhiProtocol.decode_packet(incomplete_payload)
    assert decoded.is_valid is False
    assert "mismatch" in decoded.error_detail.lower()

    # Invalid message type (e.g. 99) rejected as invalid wire format
    bad_type_pkt = struct.pack(">HHIII", 1, 99, 0, 100, 8) + (b"\x00" * 8)
    decoded_bad = XiaozhiProtocol.decode_packet(bad_type_pkt)
    assert decoded_bad.is_valid is False
    assert decoded_bad.message_type == -1

    # Invalid JSON in JSON_CONTROL packet (type 1 is JSON)
    bad_json_pkt = struct.pack(">HHIII", 1, 1, 0, 100, 12) + b"not a json!!"
    decoded_json = XiaozhiProtocol.decode_packet(bad_json_pkt)
    assert decoded_json.is_valid is True
    assert decoded_json.message_type == XiaozhiMessageType.JSON_CONTROL
    assert decoded_json.json_data is None  # Handled safely


# ============================================================================
# 3. NextSight 0xAA55 Framing vs Xiaozhi Protocol Coexistence
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

    # Xiaozhi packet has version 1 (0x0001) in big-endian (>H)
    xz_encoded = XiaozhiProtocol.encode_bp2(payload=b"xz_pcm", message_type=XiaozhiMessageType.AUDIO_STREAM)
    assert xz_encoded[:2] != b"\xAA\x55"
    decoded_xz = XiaozhiProtocol.decode_packet(xz_encoded)
    assert decoded_xz.is_valid is True


# ============================================================================
# 4. WebSocket Device Bridge Integration & Lifecycle Tests
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
    
    app.state.audio_service = AudioService(
        stt_provider=MockSTTProvider(canned_transcription="Hello NextSight from ESP32", confidence=0.99),
        tts_provider=MockTTSProvider(),
        settings=settings
    )
    app.state.conversation_service = ConversationService(
        llm_provider=MockLLMProvider(fixed_response="NextSight Assistant is active."),
    )
    app.state.vision_service = VisionService(settings=settings)

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

        # Stream multiple consecutive audio chunks using Xiaozhi BinaryProtocol2
        pcm_chunk = b"\x00" * 640  # 20ms of 16kHz audio
        for i in range(5):
            bp2_frame = XiaozhiProtocol.encode_bp2(
                payload=pcm_chunk,
                message_type=XiaozhiMessageType.AUDIO_STREAM,
                timestamp_ms=i * 20
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

        # Expect TTS start event
        tts_start = json.loads(ws.receive_text())
        assert tts_start["type"] == "tts"
        assert tts_start["state"] == "start"

        # Expect TTS sentence_start event
        tts_sent = json.loads(ws.receive_text())
        assert tts_sent["type"] == "tts"
        assert tts_sent["state"] == "sentence_start"

        # Expect binary audio frame from TTS
        audio_frame_bytes = ws.receive_bytes()
        assert len(audio_frame_bytes) > BP2_HEADER_SIZE
        decoded_tts_audio = XiaozhiProtocol.decode_packet(audio_frame_bytes)
        assert decoded_tts_audio.is_valid is True
        assert decoded_tts_audio.message_type == XiaozhiMessageType.AUDIO_STREAM
        assert len(decoded_tts_audio.payload) > 0

        # Expect TTS completion state
        tts_stop = json.loads(ws.receive_text())
        assert tts_stop["type"] == "tts"
        assert tts_stop["state"] == "stop"


def test_websocket_device_empty_audio_handling(mock_ws_client):
    """Verify listening stop with 0 audio frames handles gracefully."""
    with mock_ws_client.websocket_connect("/ws/device") as ws:
        _ = ws.receive_text()

        ws.send_text(json.dumps({"type": "listen", "state": "start"}))
        _ = json.loads(ws.receive_text())

        # Stop immediately with no audio bytes sent
        ws.send_text(json.dumps({"type": "listen", "state": "stop"}))
        proc_msg = json.loads(ws.receive_text())
        assert proc_msg["type"] == "state"
        assert proc_msg["state"] == "processing"

        stt_msg = json.loads(ws.receive_text())
        assert stt_msg["type"] == "stt"
        assert stt_msg["error"] == "No audio received"

        idle_msg = json.loads(ws.receive_text())
        assert idle_msg["type"] == "state"
        assert idle_msg["state"] == "idle"


def test_websocket_device_abort_speaking(mock_ws_client):
    with mock_ws_client.websocket_connect("/ws/device") as ws:
        _ = ws.receive_text()

        # Send abort command
        ws.send_text(json.dumps({"type": "abort"}))
        abort_res = json.loads(ws.receive_text())
        assert abort_res["type"] == "state"
        assert abort_res["state"] == "idle"
        assert abort_res["reason"] == "aborted"


def test_websocket_device_disconnect_and_reconnect(mock_ws_client):
    """Verify session isolation between disconnect and reconnect."""
    # Session 1: Stream some audio and disconnect
    with mock_ws_client.websocket_connect("/ws/device") as ws1:
        hello1 = json.loads(ws1.receive_text())
        session1_id = hello1["session_id"]
        ws1.send_text(json.dumps({"type": "listen", "state": "start"}))
        _ = ws1.receive_text()
        ws1.send_bytes(b"\x00" * 1000)

    # Session 2: Connect fresh session
    with mock_ws_client.websocket_connect("/ws/device") as ws2:
        hello2 = json.loads(ws2.receive_text())
        session2_id = hello2["session_id"]
        assert session1_id != session2_id

        # Empty stop -> verify buffer was not leaked from session 1
        ws2.send_text(json.dumps({"type": "listen", "state": "stop"}))
        _ = json.loads(ws2.receive_text())  # processing
        stt_res = json.loads(ws2.receive_text())
        assert stt_res["error"] == "No audio received"


def test_websocket_device_concurrent_sessions(mock_ws_client):
    """Verify two concurrent WebSocket sessions maintain isolated state."""
    with mock_ws_client.websocket_connect("/ws/device") as ws1:
        with mock_ws_client.websocket_connect("/ws/device") as ws2:
            h1 = json.loads(ws1.receive_text())
            h2 = json.loads(ws2.receive_text())
            assert h1["session_id"] != h2["session_id"]

            ws1.send_text(json.dumps({"type": "ping"}))
            ws2.send_text(json.dumps({"type": "ping"}))

            p1 = json.loads(ws1.receive_text())
            p2 = json.loads(ws2.receive_text())
            assert p1["type"] == "pong"
            assert p2["type"] == "pong"


def test_websocket_device_provider_failure_resilience():
    """Verify that a provider throwing an unhandled exception is caught gracefully."""
    class FailingSTT:
        async def transcribe(self, *args, **kwargs):
            raise RuntimeError("Live Provider Connection Refused (503)")

    settings = AppSettings(environment="test", require_verified_telemetry=False)
    app = create_app(settings)
    app.state.audio_service = AudioService(stt_provider=FailingSTT(), settings=settings)
    app.state.conversation_service = ConversationService()
    client = TestClient(app)

    with client.websocket_connect("/ws/device") as ws:
        _ = ws.receive_text()  # hello
        ws.send_text(json.dumps({"type": "listen", "state": "start"}))
        _ = ws.receive_text()

        ws.send_bytes(b"\x00" * 800)
        ws.send_text(json.dumps({"type": "listen", "state": "stop"}))
        _ = ws.receive_text()  # processing

        # Server sends error message rather than crashing or terminating silently
        err_msg = json.loads(ws.receive_text())
        assert err_msg["type"] == "error"
        assert err_msg["code"] == "PROCESSING_ERROR"

        # State returns to idle
        idle_msg = json.loads(ws.receive_text())
        assert idle_msg["type"] == "state"
        assert idle_msg["state"] == "idle"


def test_websocket_device_vision_request(mock_ws_client):
    with mock_ws_client.websocket_connect("/ws/device") as ws:
        _ = ws.receive_text()

        # Request vision processing
        ws.send_text(json.dumps({"type": "vision", "use_mock_frame": True}))
        vision_msg = json.loads(ws.receive_text())
        assert vision_msg["type"] == "vision_result"
        assert vision_msg["success"] is True
