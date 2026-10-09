"""Binary framing, serialization, deserialization, and CRC validation for device communication."""

import struct
import zlib
from typing import List, Tuple, Optional
from backend.protocol.contracts import (
    PacketType,
    ProtocolHeader,
    ProtocolMessage,
)

# Header format: >H (magic 2B), B (type 1B), B (flags 1B), H (seq 2B), I (ts_ms 4B), H (length 2B) -> 12 bytes
HEADER_FORMAT = ">HBBHIH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC_BYTES = 0xAA55
TRAILER_FORMAT = ">I"  # CRC32 4 bytes
TRAILER_SIZE = struct.calcsize(TRAILER_FORMAT)


class ProtocolFraming:
    """Encodes and decodes framed binary messages with CRC32 verification."""

    @staticmethod
    def encode_message(
        packet_type: PacketType,
        payload: bytes,
        sequence_number: int = 0,
        timestamp_ms: int = 0,
        flags: int = 0,
    ) -> bytes:
        """Encode a message into a framed binary byte sequence with CRC32."""
        payload_len = len(payload)
        if payload_len > 65535:
            raise ValueError(f"Payload length {payload_len} exceeds max protocol length 65535")

        seq = sequence_number % 65536

        header_bytes = struct.pack(
            HEADER_FORMAT,
            MAGIC_BYTES,
            int(packet_type),
            flags,
            seq,
            timestamp_ms & 0xFFFFFFFF,
            payload_len,
        )

        data_to_crc = header_bytes + payload
        crc = zlib.crc32(data_to_crc) & 0xFFFFFFFF
        trailer_bytes = struct.pack(TRAILER_FORMAT, crc)

        return data_to_crc + trailer_bytes

    @staticmethod
    def decode_message(raw_bytes: bytes) -> ProtocolMessage:
        """Decode a single framed binary message and verify CRC32."""
        total_len = len(raw_bytes)
        min_len = HEADER_SIZE + TRAILER_SIZE

        if total_len < min_len:
            return ProtocolMessage(
                header=ProtocolHeader(),
                payload=b"",
                crc32=0,
                is_valid=False,
                error_detail=f"Packet too short ({total_len} bytes, minimum {min_len})",
            )

        magic, p_type, flags, seq, ts_ms, payload_len = struct.unpack(
            HEADER_FORMAT, raw_bytes[:HEADER_SIZE]
        )

        if magic != MAGIC_BYTES:
            return ProtocolMessage(
                header=ProtocolHeader(),
                payload=b"",
                crc32=0,
                is_valid=False,
                error_detail=f"Invalid magic bytes 0x{magic:04X} (expected 0x{MAGIC_BYTES:04X})",
            )

        expected_total_len = HEADER_SIZE + payload_len + TRAILER_SIZE
        if total_len != expected_total_len:
            return ProtocolMessage(
                header=ProtocolHeader(),
                payload=b"",
                crc32=0,
                is_valid=False,
                error_detail=f"Length mismatch: got {total_len} bytes, expected {expected_total_len}",
            )

        payload = raw_bytes[HEADER_SIZE : HEADER_SIZE + payload_len]
        (received_crc,) = struct.unpack(TRAILER_FORMAT, raw_bytes[HEADER_SIZE + payload_len :])

        data_to_crc = raw_bytes[: HEADER_SIZE + payload_len]
        calculated_crc = zlib.crc32(data_to_crc) & 0xFFFFFFFF

        if received_crc != calculated_crc:
            return ProtocolMessage(
                header=ProtocolHeader(
                    packet_type=PacketType(p_type) if p_type in PacketType._value2member_map_ else PacketType.ERROR,
                    flags=flags,
                    sequence_number=seq,
                    timestamp_ms=ts_ms,
                    payload_length=payload_len,
                ),
                payload=payload,
                crc32=received_crc,
                is_valid=False,
                error_detail=f"CRC32 mismatch: calculated 0x{calculated_crc:08X}, received 0x{received_crc:08X}",
            )

        try:
            valid_type = PacketType(p_type)
        except ValueError:
            valid_type = PacketType.ERROR

        header = ProtocolHeader(
            magic=magic,
            packet_type=valid_type,
            flags=flags,
            sequence_number=seq,
            timestamp_ms=ts_ms,
            payload_length=payload_len,
        )

        return ProtocolMessage(
            header=header,
            payload=payload,
            crc32=received_crc,
            is_valid=True,
        )


class SequenceTracker:
    """Tracks sequence numbers to detect dropped packets and duplicates."""

    def __init__(self, max_gap: int = 100) -> None:
        self.last_sequence: Optional[int] = None
        self.dropped_packets_count: int = 0
        self.duplicate_packets_count: int = 0
        self.total_received_count: int = 0
        self.max_gap = max_gap

    def process_sequence(self, seq: int) -> Tuple[bool, int]:
        """Process an incoming sequence number.

        Returns: (is_duplicate, dropped_count_in_this_step)
        """
        self.total_received_count += 1

        if self.last_sequence is None:
            self.last_sequence = seq
            return False, 0

        expected = (self.last_sequence + 1) % 65536

        if seq == self.last_sequence:
            self.duplicate_packets_count += 1
            return True, 0

        if seq == expected:
            self.last_sequence = seq
            return False, 0

        # Check gap
        gap = (seq - self.last_sequence) % 65536
        if 0 < gap <= self.max_gap:
            dropped = gap - 1
            self.dropped_packets_count += dropped
            self.last_sequence = seq
            return False, dropped

        # Out of order or huge jump
        self.last_sequence = seq
        return False, 0
