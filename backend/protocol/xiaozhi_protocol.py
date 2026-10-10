"""Xiaozhi ESP32-S3 Binary and JSON Protocol Parser & Formatter.

Implements binary protocol specifications used by Xiaozhi ESP-IDF firmware:
1. BinaryProtocol2:
   - version: uint16 (e.g. 1)
   - type: uint16 (0: OPUS/Audio, 1: JSON)
   - reserved: uint32 (0)
   - timestamp: uint32 (ms timestamp for server-side AEC)
   - payload_size: uint32 (bytes count)
   - payload: bytes
   (Total header size: 16 bytes, little-endian packed)

2. BinaryProtocol3:
   - type: uint8 (0: Audio, 1: JSON, 2: Control)
   - reserved: uint8 (0)
   - payload_size: uint16 (bytes count)
   - payload: bytes
   (Total header size: 4 bytes, little-endian packed)
"""

import struct
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any
import json


# BinaryProtocol2 format: <H H I I I (16 bytes)
# uint16 version, uint16 type, uint32 reserved, uint32 timestamp, uint32 payload_size
BP2_HEADER_FORMAT = "<HHIII"
BP2_HEADER_SIZE = struct.calcsize(BP2_HEADER_FORMAT)  # 16 bytes

# BinaryProtocol3 format: <B B H (4 bytes)
# uint8 type, uint8 reserved, uint16 payload_size
BP3_HEADER_FORMAT = "<BBH"
BP3_HEADER_SIZE = struct.calcsize(BP3_HEADER_FORMAT)  # 4 bytes


class XiaozhiMessageType:
    AUDIO_STREAM = 0
    JSON_CONTROL = 1
    IMAGE_DATA = 2
    HEARTBEAT = 3


@dataclass
class DecodedXiaozhiPacket:
    """Decoded packet from Xiaozhi ESP32 firmware."""
    protocol_version: int  # 2 or 3
    message_type: int      # 0: Audio, 1: JSON, 2: Image
    timestamp_ms: int
    payload: bytes
    json_data: Optional[Dict[str, Any]] = None
    is_valid: bool = True
    error_detail: Optional[str] = None


class XiaozhiProtocol:
    """Encoder and Decoder for Xiaozhi ESP32-S3 binary wire protocols."""

    @staticmethod
    def encode_bp2(
        payload: bytes,
        message_type: int = XiaozhiMessageType.AUDIO_STREAM,
        timestamp_ms: int = 0,
        version: int = 1
    ) -> bytes:
        """Encodes payload into BinaryProtocol2 wire frame."""
        payload_len = len(payload)
        header = struct.pack(
            BP2_HEADER_FORMAT,
            version,
            message_type,
            0,  # reserved
            timestamp_ms & 0xFFFFFFFF,
            payload_len
        )
        return header + payload

    @staticmethod
    def encode_bp3(
        payload: bytes,
        message_type: int = XiaozhiMessageType.AUDIO_STREAM
    ) -> bytes:
        """Encodes payload into BinaryProtocol3 wire frame."""
        payload_len = len(payload)
        if payload_len > 65535:
            raise ValueError(f"Payload length {payload_len} exceeds max 65535 for BinaryProtocol3")
        header = struct.pack(
            BP3_HEADER_FORMAT,
            message_type,
            0,  # reserved
            payload_len
        )
        return header + payload

    @staticmethod
    def decode_packet(raw_bytes: bytes) -> DecodedXiaozhiPacket:
        """
        Auto-detects and decodes BinaryProtocol2, BinaryProtocol3, or JSON text from bytes.
        """
        if not raw_bytes:
            return DecodedXiaozhiPacket(
                protocol_version=0,
                message_type=-1,
                timestamp_ms=0,
                payload=b"",
                is_valid=False,
                error_detail="Empty byte packet"
            )

        total_len = len(raw_bytes)

        # Check if text JSON
        if raw_bytes.startswith(b"{") and raw_bytes.endswith(b"}"):
            try:
                parsed_json = json.loads(raw_bytes.decode("utf-8"))
                return DecodedXiaozhiPacket(
                    protocol_version=1,
                    message_type=XiaozhiMessageType.JSON_CONTROL,
                    timestamp_ms=0,
                    payload=raw_bytes,
                    json_data=parsed_json,
                    is_valid=True
                )
            except Exception as e:
                pass

        # Try BinaryProtocol2 first (16-byte header)
        if total_len >= BP2_HEADER_SIZE:
            ver, mtype, reserved, ts_ms, p_len = struct.unpack(BP2_HEADER_FORMAT, raw_bytes[:BP2_HEADER_SIZE])
            if ver in (1, 2) and mtype in (0, 1, 2, 3) and total_len == (BP2_HEADER_SIZE + p_len):
                payload = raw_bytes[BP2_HEADER_SIZE:]
                json_obj = None
                if mtype == XiaozhiMessageType.JSON_CONTROL:
                    try:
                        json_obj = json.loads(payload.decode("utf-8"))
                    except Exception:
                        pass
                return DecodedXiaozhiPacket(
                    protocol_version=2,
                    message_type=mtype,
                    timestamp_ms=ts_ms,
                    payload=payload,
                    json_data=json_obj,
                    is_valid=True
                )

        # Try BinaryProtocol3 (4-byte header)
        if total_len >= BP3_HEADER_SIZE:
            mtype, reserved, p_len = struct.unpack(BP3_HEADER_FORMAT, raw_bytes[:BP3_HEADER_SIZE])
            if mtype in (0, 1, 2, 3) and total_len == (BP3_HEADER_SIZE + p_len):
                payload = raw_bytes[BP3_HEADER_SIZE:]
                json_obj = None
                if mtype == XiaozhiMessageType.JSON_CONTROL:
                    try:
                        json_obj = json.loads(payload.decode("utf-8"))
                    except Exception:
                        pass
                return DecodedXiaozhiPacket(
                    protocol_version=3,
                    message_type=mtype,
                    timestamp_ms=0,
                    payload=payload,
                    json_data=json_obj,
                    is_valid=True
                )

        # Unknown or malformed packet
        return DecodedXiaozhiPacket(
            protocol_version=0,
            message_type=-1,
            timestamp_ms=0,
            payload=raw_bytes,
            is_valid=False,
            error_detail=f"Unrecognized wire format or payload length mismatch (received {total_len} bytes)"
        )
