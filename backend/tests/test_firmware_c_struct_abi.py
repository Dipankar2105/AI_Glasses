"""
Independent C-ABI and Wire Protocol Byte-Level Verification Suite.
Validates exact C struct packing, field offsets, field widths, endianness,
and wire-level compatibility between ESP32-S3 C++ firmware structs and Python backend.
"""

import os
import sys
import struct
import ctypes
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.protocol.xiaozhi_protocol import (
    XiaozhiProtocol,
    XiaozhiMessageType,
    DecodedXiaozhiPacket,
    BP2_HEADER_SIZE,
    BP3_HEADER_SIZE,
)


# ============================================================================
# 1. Exact C-ABI Representation of Firmware Packed Structs
# ============================================================================

class FirmwareBinaryProtocol2CStruct(ctypes.BigEndianStructure):
    """
    Direct C-ABI representation of ESP32-S3 firmware BinaryProtocol2:
    struct __attribute__((packed)) BinaryProtocol2 {
        uint16_t version;
        uint16_t type;
        uint32_t reserved;
        uint32_t timestamp;
        uint32_t payload_size;
    };
    """
    _pack_ = 1
    _fields_ = [
        ("version", ctypes.c_uint16),
        ("type", ctypes.c_uint16),
        ("reserved", ctypes.c_uint32),
        ("timestamp", ctypes.c_uint32),
        ("payload_size", ctypes.c_uint32),
    ]


class FirmwareBinaryProtocol3CStruct(ctypes.BigEndianStructure):
    """
    Direct C-ABI representation of ESP32-S3 firmware BinaryProtocol3:
    struct __attribute__((packed)) BinaryProtocol3 {
        uint8_t type;
        uint8_t reserved;
        uint16_t payload_size;
    };
    """
    _pack_ = 1
    _fields_ = [
        ("type", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
        ("payload_size", ctypes.c_uint16),
    ]


# ============================================================================
# 2. Byte-Level Field Offsets, Sizes, and Alignment Tests
# ============================================================================

def test_c_struct_bp2_abi_alignment_and_offsets():
    """Verify exact byte offsets and sizes of BinaryProtocol2 fields."""
    assert ctypes.sizeof(FirmwareBinaryProtocol2CStruct) == 16
    assert ctypes.sizeof(FirmwareBinaryProtocol2CStruct) == BP2_HEADER_SIZE

    # Field offsets
    assert FirmwareBinaryProtocol2CStruct.version.offset == 0
    assert FirmwareBinaryProtocol2CStruct.version.size == 2

    assert FirmwareBinaryProtocol2CStruct.type.offset == 2
    assert FirmwareBinaryProtocol2CStruct.type.size == 2

    assert FirmwareBinaryProtocol2CStruct.reserved.offset == 4
    assert FirmwareBinaryProtocol2CStruct.reserved.size == 4

    assert FirmwareBinaryProtocol2CStruct.timestamp.offset == 8
    assert FirmwareBinaryProtocol2CStruct.timestamp.size == 4

    assert FirmwareBinaryProtocol2CStruct.payload_size.offset == 12
    assert FirmwareBinaryProtocol2CStruct.payload_size.size == 4


def test_c_struct_bp3_abi_alignment_and_offsets():
    """Verify exact byte offsets and sizes of BinaryProtocol3 fields."""
    assert ctypes.sizeof(FirmwareBinaryProtocol3CStruct) == 4
    assert ctypes.sizeof(FirmwareBinaryProtocol3CStruct) == BP3_HEADER_SIZE

    assert FirmwareBinaryProtocol3CStruct.type.offset == 0
    assert FirmwareBinaryProtocol3CStruct.type.size == 1

    assert FirmwareBinaryProtocol3CStruct.reserved.offset == 1
    assert FirmwareBinaryProtocol3CStruct.reserved.size == 1

    assert FirmwareBinaryProtocol3CStruct.payload_size.offset == 2
    assert FirmwareBinaryProtocol3CStruct.payload_size.size == 2


def test_c_abi_bp2_binary_serialization_cross_verification():
    """
    Serialize with ctypes C-ABI struct, deserialize with XiaozhiProtocol,
    and verify exact byte match with network big-endian format.
    """
    pcm_payload = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"
    
    c_header = FirmwareBinaryProtocol2CStruct()
    c_header.version = 2
    c_header.type = 0  # 0: Audio in firmware
    c_header.reserved = 0
    c_header.timestamp = 0x12345678
    c_header.payload_size = len(pcm_payload)

    raw_c_bytes = bytes(c_header) + pcm_payload
    assert len(raw_c_bytes) == 16 + 10

    # Verify byte values at exact indices
    # Version: 0x0002 (big-endian) -> bytes [0x00, 0x02]
    assert raw_c_bytes[0:2] == b"\x00\x02"
    # Type: 0x0000 -> bytes [0x00, 0x00]
    assert raw_c_bytes[2:4] == b"\x00\x00"
    # Reserved: 0x00000000 -> bytes [0x00, 0x00, 0x00, 0x00]
    assert raw_c_bytes[4:8] == b"\x00\x00\x00\x00"
    # Timestamp: 0x12345678 -> bytes [0x12, 0x34, 0x56, 0x78]
    assert raw_c_bytes[8:12] == b"\x12\x34\x56\x78"
    # Payload size: 10 (0x0000000A) -> bytes [0x00, 0x00, 0x00, 0x0A]
    assert raw_c_bytes[12:16] == b"\x00\x00\x00\x0A"

    # Verify decode
    decoded = XiaozhiProtocol.decode_packet(raw_c_bytes)
    assert decoded.is_valid is True
    assert decoded.protocol_version == 2
    assert decoded.message_type == XiaozhiMessageType.AUDIO_STREAM
    assert decoded.timestamp_ms == 0x12345678
    assert decoded.payload == pcm_payload


def test_c_abi_bp3_binary_serialization_cross_verification():
    """
    Serialize with ctypes C-ABI struct, deserialize with XiaozhiProtocol,
    and verify exact byte match with network big-endian format.
    """
    pcm_payload = b"\xFE\xDC\xBA\x98"
    
    c_header = FirmwareBinaryProtocol3CStruct()
    c_header.type = 0  # 0: Audio in firmware
    c_header.reserved = 0
    c_header.payload_size = len(pcm_payload)

    raw_c_bytes = bytes(c_header) + pcm_payload
    assert len(raw_c_bytes) == 4 + 4

    # Type: 0x00
    assert raw_c_bytes[0] == 0x00
    # Reserved: 0x00
    assert raw_c_bytes[1] == 0x00
    # Payload size: 4 (0x0004)
    assert raw_c_bytes[2:4] == b"\x00\x04"

    decoded = XiaozhiProtocol.decode_packet(raw_c_bytes)
    assert decoded.is_valid is True
    assert decoded.protocol_version == 3
    assert decoded.message_type == XiaozhiMessageType.AUDIO_STREAM
    assert decoded.payload == pcm_payload


# ============================================================================
# 3. Audio Streaming Edge Cases: Fragmented, Odd-Sized, and Multi-Chunk Streams
# ============================================================================

def test_odd_sized_and_fragmented_audio_chunks():
    """Verify protocol handling of odd-sized or fragmented binary frames."""
    odd_payload = b"\x12\x34\x56"  # 3 bytes (odd length)
    encoded = XiaozhiProtocol.encode_bp2(
        payload=odd_payload,
        message_type=XiaozhiMessageType.AUDIO_STREAM,
        timestamp_ms=100
    )
    decoded = XiaozhiProtocol.decode_packet(encoded)
    assert decoded.is_valid is True
    assert decoded.payload == odd_payload

    # Empty payload
    empty_encoded = XiaozhiProtocol.encode_bp2(
        payload=b"",
        message_type=XiaozhiMessageType.AUDIO_STREAM,
        timestamp_ms=200
    )
    decoded_empty = XiaozhiProtocol.decode_packet(empty_encoded)
    assert decoded_empty.is_valid is True
    assert decoded_empty.payload == b""


def test_audio_format_contracts():
    """
    Verify audio formatting invariants:
    16 kHz, 16-bit signed PCM mono:
    - 2 bytes per sample
    - 16000 samples/sec = 32000 bytes/sec
    - 60ms frame = 960 samples = 1920 bytes
    - 20ms frame = 320 samples = 640 bytes
    """
    sample_rate = 16000
    bytes_per_sample = 2  # int16
    channels = 1

    frame_duration_60ms = 0.060
    expected_bytes_60ms = int(sample_rate * frame_duration_60ms * bytes_per_sample * channels)
    assert expected_bytes_60ms == 1920

    frame_duration_20ms = 0.020
    expected_bytes_20ms = int(sample_rate * frame_duration_20ms * bytes_per_sample * channels)
    assert expected_bytes_20ms == 640
