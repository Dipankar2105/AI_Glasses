"""Unit and integration tests for Phase 10: Hardware Integration Readiness & Device Protocol."""

import json
import pytest
from backend.protocol.contracts import PacketType, ProtocolHeader, ProtocolMessage
from backend.protocol.framing import ProtocolFraming, SequenceTracker, HEADER_SIZE
from backend.protocol.device_sim import SimulatedESP32Device
from firmware.system.runtime import SystemRuntime, SystemState


def test_packet_encode_decode_roundtrip():
    """Verify binary message encoding, CRC32 generation, and valid decoding."""
    payload = b"Hello NextSight Glasses Protocol"
    encoded = ProtocolFraming.encode_message(
        packet_type=PacketType.TELEMETRY,
        payload=payload,
        sequence_number=42,
        timestamp_ms=123456,
        flags=0x01,
    )

    decoded = ProtocolFraming.decode_message(encoded)
    assert decoded.is_valid is True
    assert decoded.error_detail is None
    assert decoded.header.packet_type == PacketType.TELEMETRY
    assert decoded.header.sequence_number == 42
    assert decoded.header.timestamp_ms == 123456
    assert decoded.header.flags == 0x01
    assert decoded.payload == payload


def test_packet_crc_corruption_detection():
    """Verify corrupted payload or header causes CRC32 validation failure."""
    payload = b"Sensor data payload"
    encoded = bytearray(
        ProtocolFraming.encode_message(
            packet_type=PacketType.AUDIO_PCM,
            payload=payload,
            sequence_number=1,
            timestamp_ms=1000,
        )
    )

    # Corrupt 1 byte in payload
    encoded[HEADER_SIZE + 3] ^= 0xFF

    decoded = ProtocolFraming.decode_message(bytes(encoded))
    assert decoded.is_valid is False
    assert "CRC32 mismatch" in decoded.error_detail


def test_packet_invalid_magic_rejection():
    """Verify invalid magic bytes are rejected."""
    payload = b"data"
    encoded = bytearray(
        ProtocolFraming.encode_message(packet_type=PacketType.HEARTBEAT, payload=payload)
    )
    # Corrupt magic byte
    encoded[0] = 0x00

    decoded = ProtocolFraming.decode_message(bytes(encoded))
    assert decoded.is_valid is False
    assert "Invalid magic bytes" in decoded.error_detail


def test_packet_truncated_rejection():
    """Verify truncated bytes shorter than header + trailer are safely rejected."""
    decoded = ProtocolFraming.decode_message(b"\xAA\x55\x01")
    assert decoded.is_valid is False
    assert "Packet too short" in decoded.error_detail


def test_sequence_tracker_drop_and_duplicate_detection():
    """Verify sequence tracker accurately detects dropped packets and duplicates."""
    tracker = SequenceTracker(max_gap=50)

    # First packet: seq=10
    is_dup, dropped = tracker.process_sequence(10)
    assert is_dup is False
    assert dropped == 0

    # In-order packet: seq=11
    is_dup, dropped = tracker.process_sequence(11)
    assert is_dup is False
    assert dropped == 0

    # Duplicate packet: seq=11
    is_dup, dropped = tracker.process_sequence(11)
    assert is_dup is True
    assert dropped == 0
    assert tracker.duplicate_packets_count == 1

    # Jump packet: seq=15 (dropped 12, 13, 14 -> 3 dropped)
    is_dup, dropped = tracker.process_sequence(15)
    assert is_dup is False
    assert dropped == 3
    assert tracker.dropped_packets_count == 3


def test_simulated_esp32_device():
    """Verify SimulatedESP32Device telemetry, audio, and queue backpressure."""
    sim = SimulatedESP32Device(device_id="glasses-01", queue_capacity=5)

    assert sim.is_connected is False
    sim.connect()
    assert sim.is_connected is True

    # Send heartbeat and telemetry
    hb_bytes = sim.send_heartbeat()
    telem_bytes = sim.send_telemetry(battery_pct=90.0, temp_c=31.2)

    decoded_hb = ProtocolFraming.decode_message(hb_bytes)
    assert decoded_hb.is_valid is True
    assert decoded_hb.header.packet_type == PacketType.HEARTBEAT

    decoded_telem = ProtocolFraming.decode_message(telem_bytes)
    assert decoded_telem.is_valid is True
    assert decoded_telem.header.packet_type == PacketType.TELEMETRY
    payload_obj = json.loads(decoded_telem.payload.decode("utf-8"))
    assert payload_obj["battery_pct"] == 90.0
    assert payload_obj["temp_c"] == 31.2

    # Fill queue to test backpressure drop
    for i in range(10):
        sim.send_audio_chunk(b"PCM_CHUNK_" + str(i).encode("utf-8"))

    assert sim.dropped_tx_packets > 0
    assert len(sim.tx_queue) == 5

    flushed = sim.flush_tx_queue()
    assert len(flushed) == 5
    assert len(sim.tx_queue) == 0


def test_system_runtime_lifecycle():
    """Verify system runtime boot, transitions, streaming, degradation, and shutdown."""
    runtime = SystemRuntime()
    assert runtime.state == SystemState.BOOT.value

    runtime.boot()
    assert runtime.state == SystemState.READY.value
    assert runtime.peripherals["camera"] is True

    runtime.process_event("START_STREAM")
    assert runtime.state == SystemState.STREAMING.value

    runtime.process_event("STOP_STREAM")
    assert runtime.state == SystemState.READY.value

    runtime.process_event("PERIPHERAL_ERROR")
    assert runtime.state == SystemState.DEGRADED.value

    runtime.process_event("SHUTDOWN")
    assert runtime.state == SystemState.SHUTDOWN.value
    assert runtime.peripherals["camera"] is False
