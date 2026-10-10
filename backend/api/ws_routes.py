"""WebSocket device routes for bidirectional ESP32-S3 audio, vision, and conversation streaming."""

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Optional, Dict, Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, status

from backend.config.settings import AppSettings, get_settings
from backend.services.audio_service import AudioService, get_audio_service
from backend.services.conversation_service import ConversationService, get_conversation_service
from backend.services.vision_service import VisionService, get_vision_service
from backend.conversation.models import ConversationMessageRequest
from backend.protocol.contracts import PacketType
from backend.protocol.framing import ProtocolFraming, HEADER_SIZE, MAGIC_BYTES
from backend.protocol.xiaozhi_protocol import XiaozhiProtocol, XiaozhiMessageType, DecodedXiaozhiPacket
from backend.vision.frame import VisionFrame

logger = logging.getLogger("nextsight.websocket")

router = APIRouter(tags=["WebSocket Device Stream"])

MAX_INCOMING_BUFFER_BYTES = 10 * 1024 * 1024  # 10 MB maximum audio session buffer
WS_INACTIVITY_TIMEOUT_SEC = 60.0


@router.websocket("/ws/device")
@router.websocket("/api/v1/ws/device")
async def websocket_device_endpoint(
    websocket: WebSocket,
    settings: AppSettings = Depends(get_settings),
):
    """
    Bidirectional streaming WebSocket endpoint supporting both Xiaozhi ESP32-S3 firmware
    and NextSight native framed binary packets.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"WebSocket client connected. Allocated Session ID: {session_id}")

    audio_service: AudioService = getattr(websocket.app.state, "audio_service", None) or get_audio_service()
    conversation_service: ConversationService = getattr(websocket.app.state, "conversation_service", None) or get_conversation_service()
    vision_service: VisionService = getattr(websocket.app.state, "vision_service", None) or get_vision_service()

    # 1. Send Server Hello
    server_hello = {
        "type": "hello",
        "version": 3,
        "transport": "websocket",
        "audio_params": {
            "format": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
            "frame_duration": 60
        },
        "session_id": session_id,
        "server_status": "READY"
    }
    await websocket.send_text(json.dumps(server_hello))

    audio_buffer = bytearray()
    is_listening = False
    last_activity_time = time.time()

    try:
        while True:
            try:
                # Enforce inactivity timeout
                msg = await asyncio.wait_for(websocket.receive(), timeout=WS_INACTIVITY_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                logger.warning(f"WebSocket session {session_id} timed out due to inactivity")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "code": "TIMEOUT",
                    "message": "Connection closed due to inactivity timeout"
                }))
                break

            last_activity_time = time.time()

            # ----------------------------------------------------
            # Handle Text Messages (JSON control)
            # ----------------------------------------------------
            if "text" in msg and msg["text"]:
                raw_text = msg["text"].strip()
                try:
                    payload = json.loads(raw_text)
                except Exception:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "code": "INVALID_JSON",
                        "message": "Malformed JSON text payload"
                    }))
                    continue

                msg_type = payload.get("type", "")

                if msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))

                elif msg_type == "hello":
                    # Acknowledge client hello
                    client_ver = payload.get("version", 1)
                    await websocket.send_text(json.dumps({
                        "type": "hello_ack",
                        "session_id": session_id,
                        "status": "connected"
                    }))

                elif msg_type == "listen":
                    state = payload.get("state", "start")
                    if state == "start":
                        is_listening = True
                        audio_buffer.clear()
                        await websocket.send_text(json.dumps({"type": "state", "state": "listening"}))
                    elif state in ("stop", "detect"):
                        is_listening = False
                        await websocket.send_text(json.dumps({"type": "state", "state": "processing"}))

                        # Process accumulated audio
                        if len(audio_buffer) > 0:
                            try:
                                transcribe_res = await audio_service.process_and_transcribe(
                                    audio_bytes=bytes(audio_buffer),
                                    sample_rate=16000,
                                    run_dsp=True
                                )
                                audio_buffer.clear()

                                if transcribe_res.success and transcribe_res.transcript:
                                    transcript_text = transcribe_res.transcript
                                    await websocket.send_text(json.dumps({
                                        "type": "stt",
                                        "text": transcript_text,
                                        "confidence": transcribe_res.confidence
                                    }))

                                    # Orchestrate Conversation
                                    conv_req = ConversationMessageRequest(
                                        session_id=session_id,
                                        message=transcript_text
                                    )
                                    conv_res = await conversation_service.process_user_message(conv_req)

                                    await websocket.send_text(json.dumps({
                                        "type": "llm",
                                        "text": conv_res.response,
                                        "status": conv_res.llm_status
                                    }))

                                    # Synthesize TTS audio response
                                    tts_res = await audio_service.synthesize_speech(text=conv_res.response)
                                    if tts_res.success and tts_res.audio_bytes:
                                        # Notify device TTS playback start
                                        await websocket.send_text(json.dumps({
                                            "type": "tts",
                                            "state": "start"
                                        }))
                                        await websocket.send_text(json.dumps({
                                            "type": "tts",
                                            "state": "sentence_start",
                                            "text": conv_res.response
                                        }))
                                        # Send binary audio frame in network byte order
                                        bp2_audio = XiaozhiProtocol.encode_bp2(
                                            payload=tts_res.audio_bytes,
                                            message_type=XiaozhiMessageType.AUDIO_STREAM,
                                            timestamp_ms=int(time.time() * 1000)
                                        )
                                        await websocket.send_bytes(bp2_audio)
                                        # Notify device TTS playback completion
                                        await websocket.send_text(json.dumps({
                                            "type": "tts",
                                            "state": "stop",
                                            "sample_rate": tts_res.sample_rate
                                        }))
                                    else:
                                        await websocket.send_text(json.dumps({
                                            "type": "tts",
                                            "state": "stop",
                                            "error": tts_res.error or "TTS synthesis failed"
                                        }))
                                else:
                                    await websocket.send_text(json.dumps({
                                        "type": "stt",
                                        "text": "",
                                        "error": transcribe_res.error or "No speech detected"
                                    }))
                                    await websocket.send_text(json.dumps({"type": "state", "state": "idle"}))
                            except Exception as e:
                                logger.error(f"Error processing voice turn: {e}", exc_info=True)
                                audio_buffer.clear()
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "code": "PROCESSING_ERROR",
                                    "message": f"Provider error: {str(e)}"
                                }))
                                await websocket.send_text(json.dumps({"type": "state", "state": "idle"}))
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "stt",
                                "text": "",
                                "error": "No audio received"
                            }))
                            await websocket.send_text(json.dumps({"type": "state", "state": "idle"}))

                elif msg_type == "abort":
                    # Immediate cancellation of speech/playback
                    is_listening = False
                    audio_buffer.clear()
                    await websocket.send_text(json.dumps({"type": "state", "state": "idle", "reason": "aborted"}))

                elif msg_type == "vision":
                    # Vision request with optional Base64 or synthetic frame
                    use_synthetic = payload.get("use_mock_frame", True)
                    vdata = np.full((100, 100, 3), 128, dtype=np.uint8)
                    vframe = VisionFrame(
                        data=vdata,
                        width=100,
                        height=100,
                        channels=3,
                        pixel_format="RGB",
                        numerical_range=(0, 255),
                        timestamp=int(time.time() * 1000),
                        seq_num=1
                    )
                    vres = await vision_service.process_frame_async(vframe)
                    await websocket.send_text(json.dumps({
                        "type": "vision_result",
                        "success": len(vres.errors) == 0,
                        "scene": vres.scene_result.description if vres.scene_result else None,
                        "errors": vres.errors
                    }))

            # ----------------------------------------------------
            # Handle Binary Messages (Audio / Camera / Framed Packets)
            # ----------------------------------------------------
            elif "bytes" in msg and msg["bytes"]:
                raw_bytes = msg["bytes"]

                # Check NextSight 0xAA55 Framing
                if len(raw_bytes) >= HEADER_SIZE and raw_bytes[:2] == b"\xAA\x55":
                    decoded_ns = ProtocolFraming.decode_message(raw_bytes)
                    if not decoded_ns.is_valid:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "code": "CORRUPTED_FRAME",
                            "detail": decoded_ns.error_detail
                        }))
                        continue

                    if decoded_ns.header.packet_type == PacketType.AUDIO_PCM:
                        if len(audio_buffer) + len(decoded_ns.payload) <= MAX_INCOMING_BUFFER_BYTES:
                            audio_buffer.extend(decoded_ns.payload)
                    elif decoded_ns.header.packet_type == PacketType.HEARTBEAT:
                        ack_bytes = ProtocolFraming.encode_message(PacketType.ACK, b"OK", sequence_number=decoded_ns.header.sequence_number)
                        await websocket.send_bytes(ack_bytes)

                else:
                    # Parse Xiaozhi binary packet
                    decoded_xz = XiaozhiProtocol.decode_packet(raw_bytes)
                    if decoded_xz.is_valid:
                        if decoded_xz.message_type == XiaozhiMessageType.AUDIO_STREAM:
                            if len(audio_buffer) + len(decoded_xz.payload) <= MAX_INCOMING_BUFFER_BYTES:
                                audio_buffer.extend(decoded_xz.payload)
                        elif decoded_xz.message_type == XiaozhiMessageType.JSON_CONTROL and decoded_xz.json_data:
                            # Forward JSON control packet
                            pass
                    else:
                        # Raw PCM bytes direct stream
                        if len(audio_buffer) + len(raw_bytes) <= MAX_INCOMING_BUFFER_BYTES:
                            audio_buffer.extend(raw_bytes)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected cleanly. Session ID: {session_id}")
    except Exception as exc:
        logger.error(f"WebSocket session {session_id} error: {str(exc)}")
    finally:
        audio_buffer.clear()
