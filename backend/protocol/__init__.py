"""NextSight Device Protocol and Simulation Package."""

from backend.protocol.contracts import (
    PacketType,
    ProtocolHeader,
    ProtocolMessage,
)
from backend.protocol.framing import (
    ProtocolFraming,
    SequenceTracker,
    HEADER_SIZE,
    MAGIC_BYTES,
)
from backend.protocol.device_sim import SimulatedESP32Device

__all__ = [
    "PacketType",
    "ProtocolHeader",
    "ProtocolMessage",
    "ProtocolFraming",
    "SequenceTracker",
    "HEADER_SIZE",
    "MAGIC_BYTES",
    "SimulatedESP32Device",
]
