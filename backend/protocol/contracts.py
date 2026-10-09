"""Protocol data contracts and packet definitions for ESP32 <-> Backend communication."""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class PacketType(IntEnum):
    """Binary packet type identifiers."""
    HEARTBEAT = 0x01
    TELEMETRY = 0x02
    AUDIO_PCM = 0x03
    CAMERA_FRAME = 0x04
    MOTION_EVENT = 0x05
    TOUCH_EVENT = 0x06
    COMMAND = 0x10
    ACK = 0x11
    NACK = 0x12
    ERROR = 0xFF


@dataclass(frozen=True)
class ProtocolHeader:
    """Standard 12-byte binary protocol packet header.

    Header Structure:
    - Magic bytes: 2 bytes (0xAA, 0x55)
    - Packet type: 1 byte (PacketType)
    - Flags: 1 byte (Bit 0: compressed, Bit 1: encrypted, Bit 2: urgent)
    - Sequence number: 2 bytes (0 - 65535 unsigned short)
    - Timestamp: 4 bytes (uint32 milliseconds since boot)
    - Payload length: 2 bytes (uint16 length in bytes, max 65535)
    """
    magic: int = 0xAA55
    packet_type: PacketType = PacketType.HEARTBEAT
    flags: int = 0
    sequence_number: int = 0
    timestamp_ms: int = 0
    payload_length: int = 0


@dataclass
class ProtocolMessage:
    """Full decoded message containing header, payload, and CRC32 verification status."""
    header: ProtocolHeader
    payload: bytes
    crc32: int
    is_valid: bool = True
    error_detail: Optional[str] = None
