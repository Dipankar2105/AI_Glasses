# Communication Protocol Specification

**Status:** Software Implementation Validated (Device Simulator & Protocol Framing)  
**Hardware Status:** Pending physical ESP32-S3 bench test

---

## 1. Protocol Architecture & Transports

NextSight smart glasses use a dual-channel transport:
1. **HTTP/REST Control API (`FastAPI` Backend):** For session initialization, health checks, capabilities negotiation, and RESTful tool invocations.
2. **Binary Framed Transport (WebSocket / TCP Stream):** For low-latency streaming of audio PCM chunks, OV3660 camera JPEG frames, IMU/Touch events, and device telemetry.

---

## 2. Binary Framing Specification

Every streaming packet consists of a **12-byte fixed header**, a variable-length payload (up to 65,535 bytes), and a **4-byte CRC32 trailer**.

```
+----------------------------------------------------------------------------------+
| Magic (2B) | Type (1B) | Flags (1B) | Seq (2B) | Timestamp (4B) | Length (2B)    |
|   0xAA55   |  uint8    |   uint8    |  uint16  | uint32 (ms)    | uint16 (bytes) |
+----------------------------------------------------------------------------------+
| Payload (0 to 65,535 bytes) ...                                                  |
+----------------------------------------------------------------------------------+
| CRC32 (4B) - IEEE 802.3 over Header + Payload                                    |
+----------------------------------------------------------------------------------+
```

### Packet Types

| Type Code | Identifier | Description | Typical Rate |
| :--- | :--- | :--- | :--- |
| `0x01` | `HEARTBEAT` | Ping / keepalive containing device ID & firmware version | 1 Hz |
| `0x02` | `TELEMETRY` | Battery voltage/%, die temp, Wi-Fi RSSI in JSON format | 0.5 – 1 Hz |
| `0x03` | `AUDIO_PCM` | Raw 16kHz 16-bit Mono PCM audio chunks (640 bytes = 20ms) | 50 Hz |
| `0x04` | `CAMERA_FRAME` | Compressed JPEG image frame with capture timestamp | On-demand / 1-5 FPS |
| `0x05` | `MOTION_EVENT` | Detected head gesture (Nod, Shake, Tilt) event with confidence | On-event |
| `0x06` | `TOUCH_EVENT` | Capacitive touch gesture (Tap, Double Tap, Long Press) | On-event |
| `0x10` | `COMMAND` | Backend-to-device command (e.g., enter sleep, start TTS) | Asynchronous |
| `0x11` | `ACK` | Acknowledgment packet with matching sequence number | Asynchronous |
| `0x12` | `NACK` | Negative acknowledgment / retransmission request | Asynchronous |
| `0xFF` | `ERROR` | Malformed packet or error notification | Asynchronous |

---

## 3. Audio & Camera Payloads

- **Audio Payload Format:** 16,000 Hz, 16-bit signed integer, 1-channel (Mono) Little-Endian PCM.
- **Camera Frame Format:** JPEG compressed byte stream (RGB/Grayscale, QVGA 320x240 or VGA 640x480).

---

## 4. Backpressure & Bounded Queue Policy

- Embedded ESP32-S3 uses a circular ring buffer (bounded queue capacity = 50 packets).
- If network congestion occurs and the transmission queue fills, the oldest non-essential packet (e.g. dropped audio or telemetry) is evicted to protect real-time synchronization and prevent out-of-memory crashes.
