"""FastAPI routes for runtime Audio transcription, Speech synthesis, and Provider status."""

import base64
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse, Response

from backend.config.settings import AppSettings, get_settings
from backend.services.audio_service import AudioService, get_audio_service
from backend.providers.contracts import ProviderStatus
from backend.api.models import (
    AudioTranscribeRequest,
    AudioTranscribeResponse,
    SpeechSynthesizeRequest,
    SpeechSynthesizeResponse,
    ProviderStatusInfo,
    ProvidersStatusResponse
)

router = APIRouter(prefix="/api/v1", tags=["Audio & Providers"])

MAX_AUDIO_BYTES_LIMIT = 10 * 1024 * 1024  # 10 MB maximum request payload size


def get_app_settings(request: Request) -> AppSettings:
    if hasattr(request.app.state, "settings") and request.app.state.settings is not None:
        return request.app.state.settings
    return get_settings()


def get_app_audio_service(request: Request) -> AudioService:
    if hasattr(request.app.state, "audio_service") and request.app.state.audio_service is not None:
        return request.app.state.audio_service
    return get_audio_service()


@router.post("/audio/transcribe", response_model=AudioTranscribeResponse)
async def transcribe_audio(
    payload: AudioTranscribeRequest,
    raw_request: Request,
    service: AudioService = Depends(get_app_audio_service)
):
    """
    Transcribes incoming audio. Runs host DSP preprocessing (noise suppression,
    VAD, AGC, Limiter, AEC) and invokes the configured STT provider.
    Accepts Base64 PCM or WAV audio bytes.
    """
    if not payload.audio_base64 or not payload.audio_base64.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed audio payload: audio_base64 is required and cannot be empty"
        )

    try:
        audio_bytes = base64.b64decode(payload.audio_base64.strip(), validate=True)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed audio payload: Invalid base64 encoding ({str(e)})"
        )

    if len(audio_bytes) > MAX_AUDIO_BYTES_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio payload size ({len(audio_bytes)} bytes) exceeds maximum limit of {MAX_AUDIO_BYTES_LIMIT} bytes"
        )

    ref_bytes = None
    if payload.reference_audio_base64 and payload.reference_audio_base64.strip():
        try:
            ref_bytes = base64.b64decode(payload.reference_audio_base64.strip(), validate=True)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed reference audio: Invalid base64 encoding ({str(e)})"
            )

    result = await service.process_and_transcribe(
        audio_bytes=audio_bytes,
        sample_rate=payload.sample_rate,
        channels=payload.channels,
        reference_audio_bytes=ref_bytes,
        run_dsp=payload.run_dsp
    )

    req_id = getattr(raw_request.state, "request_id", None) if hasattr(raw_request, "state") else None

    if not result.success:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        if result.raw_status == "AUTH_FAILED":
            status_code = status.HTTP_502_BAD_GATEWAY
        elif result.raw_status == "INVALID_INPUT":
            status_code = status.HTTP_400_BAD_REQUEST
        raise HTTPException(
            status_code=status_code,
            detail=f"STT Provider error: {result.error}"
        )

    return AudioTranscribeResponse(
        success=result.success,
        transcript=result.transcript,
        confidence=result.confidence,
        vad_speech_active=result.vad_speech_active,
        dsp_metrics=result.dsp_metrics,
        latency_ms=result.latency_ms,
        error=result.error,
        request_id=req_id
    )


@router.post("/speech/synthesize", response_model=SpeechSynthesizeResponse)
async def synthesize_speech(
    payload: SpeechSynthesizeRequest,
    raw_request: Request,
    service: AudioService = Depends(get_app_audio_service)
):
    """
    Synthesizes speech audio from text using the configured TTS provider.
    Returns Base64 encoded audio with accurate sample rate and encoding metadata.
    """
    cleaned_text = payload.text.strip()
    if not cleaned_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text to synthesize cannot be empty"
        )

    result = await service.synthesize_speech(
        text=cleaned_text,
        voice=payload.voice
    )

    req_id = getattr(raw_request.state, "request_id", None) if hasattr(raw_request, "state") else None

    if not result.success:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        if result.raw_status == "AUTH_FAILED":
            status_code = status.HTTP_502_BAD_GATEWAY
        elif result.raw_status == "INVALID_INPUT":
            status_code = status.HTTP_400_BAD_REQUEST

        raise HTTPException(
            status_code=status_code,
            detail=f"TTS Provider error: {result.error}"
        )

    audio_b64 = base64.b64encode(result.audio_bytes).decode("ascii") if result.audio_bytes else ""

    return SpeechSynthesizeResponse(
        success=result.success,
        audio_base64=audio_b64,
        sample_rate=result.sample_rate,
        encoding=result.encoding,
        channels=result.channels,
        latency_ms=result.latency_ms,
        error=result.error,
        request_id=req_id
    )


@router.get("/providers/status", response_model=ProvidersStatusResponse)
async def get_providers_status(settings: AppSettings = Depends(get_app_settings)):
    """
    Returns the real configuration and availability status of STT, LLM, and TTS providers.
    Never exposes raw API keys or secrets.
    """
    def check_info(provider_type: str, model: Optional[str], api_key: Optional[str]) -> ProviderStatusInfo:
        ptype = (provider_type or "null").lower()
        is_mock = ptype in ("mock", "mock_stt", "mock_llm", "mock_tts")
        is_null = ptype in ("null", "none", "disabled")
        has_key = bool(api_key and api_key.strip() and not api_key.startswith("your_"))
        
        is_configured = not is_null
        is_available = is_mock or (is_configured and has_key)
        
        if is_null:
            status_str = "UNAVAILABLE"
        elif is_mock:
            status_str = "OPERATIONAL (Mock/Deterministic)"
        elif has_key:
            status_str = "OPERATIONAL (Configured)"
        else:
            status_str = "MISSING_CREDENTIALS"

        return ProviderStatusInfo(
            provider_type=provider_type,
            model=model,
            configured=is_configured,
            available=is_available,
            status=status_str,
            offline_mode=is_mock or is_null,
            api_key_configured=has_key
        )

    stt_key = settings.openai_api_key
    llm_key = settings.openai_api_key or settings.gemini_api_key or settings.anthropic_api_key
    tts_key = settings.tts_api_key or settings.openai_api_key

    stt_info = check_info(settings.stt_provider, settings.stt_model, stt_key)
    llm_info = check_info(settings.llm_provider, settings.llm_model, llm_key)
    tts_info = check_info(settings.tts_provider, settings.tts_model, tts_key)

    return ProvidersStatusResponse(
        stt=stt_info,
        llm=llm_info,
        tts=tts_info,
        environment=settings.environment,
        timestamp=time.time()
    )
