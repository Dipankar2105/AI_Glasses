"""Deterministic software simulator for Seeed Studio XIAO ESP32-S3 Sense smart glasses."""

import time
import json
from typing import List, Optional, Tuple
from collections import deque
from backend.protocol.contracts import PacketType
from backend.protocol.framing import ProtocolFraming, SequenceTracker


class SimulatedESP32Device:
    """Simulates the hardware device side of NextSight smart glasses.

    Supports:
    - Connect/Disconnect/Reconnect state transitions
    - Bounded transmission buffer with drop-oldest backpressure policy
    - Telemetry packets, Audio PCM bursts, Camera JPEG mock frames
    - Heartbeat generation
    """

    def __init__(
        self,
        device_id: str = "nextsight-proto-01",
        queue_capacity: int = 50,
        firmware_version: str = "0.8.0-dev",
    ) -> None:
        self.device_id = device_id
        self.queue_capacity = queue_capacity
        self.firmware_version = firmware_version

        self.is_connected = False
        self.sequence_number = 0
        self.boot_time = time.time()
        self.tx_queue: deque[bytes] = deque(maxlen=queue_capacity)
        self.dropped_tx_packets = 0

    def connect(self) -> None:
        """Establish simulated network connection."""
        self.is_connected = True

    def disconnect(self) -> None:
        """Simulate network disconnection."""
        self.is_connected = False

    def get_timestamp_ms(self) -> int:
        """Return milliseconds elapsed since simulated device boot."""
        return int((time.time() - self.boot_time) * 1000)

    def _enqueue_packet(self, packet_type: PacketType, payload: bytes) -> bytes:
        """Create and queue framed binary packet with backpressure tracking."""
        seq = self.sequence_number
        self.sequence_number = (self.sequence_number + 1) % 65536
        ts = self.get_timestamp_ms()

        packet = ProtocolFraming.encode_message(
            packet_type=packet_type,
            payload=payload,
            sequence_number=seq,
            timestamp_ms=ts,
        )

        if len(self.tx_queue) == self.tx_queue.maxlen:
            self.dropped_tx_packets += 1

        self.tx_queue.append(packet)
        return packet

    def send_heartbeat(self) -> bytes:
        """Generate framed heartbeat packet."""
        payload = json.dumps({"device_id": self.device_id, "fw": self.firmware_version}).encode("utf-8")
        return self._enqueue_packet(PacketType.HEARTBEAT, payload)

    def send_telemetry(self, battery_pct: float = 85.0, temp_c: float = 34.5, wifi_rssi: int = -55) -> bytes:
        """Generate framed JSON telemetry packet."""
        payload_dict = {
            "device_id": self.device_id,
            "battery_pct": battery_pct,
            "voltage_v": round(3.3 + (battery_pct / 100.0) * 0.9, 2),
            "temp_c": temp_c,
            "wifi_rssi": wifi_rssi,
        }
        payload = json.dumps(payload_dict).encode("utf-8")
        return self._enqueue_packet(PacketType.TELEMETRY, payload)

    def send_audio_chunk(self, pcm_bytes: bytes) -> bytes:
        """Generate framed 16kHz 16-bit mono audio PCM packet."""
        return self._enqueue_packet(PacketType.AUDIO_PCM, pcm_bytes)

    def send_camera_frame(self, jpeg_bytes: bytes) -> bytes:
        """Generate framed camera image packet."""
        return self._enqueue_packet(PacketType.CAMERA_FRAME, jpeg_bytes)

    def flush_tx_queue(self) -> List[bytes]:
        """Drain and return all queued transmission packets."""
        packets = list(self.tx_queue)
        self.tx_queue.clear()
        return packets
