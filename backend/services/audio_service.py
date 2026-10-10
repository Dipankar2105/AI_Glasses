"""Audio processing and AI voice service integrating DSP and Speech Providers."""

import io
import time
import wave
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple

from firmware.audio.dsp.integrated_pipeline import (
    IntegratedAudioPipeline,
    AudioPipelineConfig,
    AudioFrame,
    AudioProcessingResult,
    AudioProcessingMetrics
)
from backend.providers.contracts import STTResult, TTSResult, ProviderStatus
from backend.providers.stt import BaseSTTProvider
from backend.providers.tts import BaseTTSProvider
from backend.providers.factory import get_stt_provider, get_tts_provider
from backend.config.settings import AppSettings, get_settings


@dataclass
class AudioTranscribeResult:
    """Result of audio DSP processing and speech transcription."""
    success: bool
    transcript: str
    confidence: float
    dsp_metrics: Optional[Dict[str, Any]] = None
    vad_speech_active: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None
    raw_status: Optional[str] = None


@dataclass
class SpeechSynthesizeResult:
    """Result of speech synthesis from text."""
    success: bool
    audio_bytes: bytes = b""
    sample_rate: int = 24000
    encoding: str = "pcm_s16le"
    channels: int = 1
    latency_ms: float = 0.0
    error: Optional[str] = None
    raw_status: Optional[str] = None


def extract_pcm_from_wav_or_raw(raw_bytes: bytes, default_sample_rate: int = 16000) -> Tuple[bytes, int, int]:
    """
    Extracts raw 16-bit PCM bytes, sample rate, and channel count.
    Supports both raw PCM buffers and RIFF/WAV containers.
    """
    if len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                if sample_width != 2:
                    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes (expected 2 bytes / 16-bit)")
                pcm_data = wf.readframes(wf.getnframes())
                return pcm_data, sample_rate, channels
        except Exception as e:
            raise ValueError(f"Malformed WAV container: {str(e)}")
    
    return raw_bytes, default_sample_rate, 1


class AudioService:
    """
    High-level audio service orchestrating host DSP preprocessing,
    voice activity detection, speech-to-text (STT) transcription,
    and text-to-speech (TTS) synthesis.
    """
    def __init__(
        self,
        stt_provider: Optional[BaseSTTProvider] = None,
        tts_provider: Optional[BaseTTSProvider] = None,
        pipeline_config: Optional[AudioPipelineConfig] = None,
        settings: Optional[AppSettings] = None
    ):
        self.settings = settings or get_settings()
        self.stt_provider = stt_provider or get_stt_provider(settings=self.settings)
        self.tts_provider = tts_provider or get_tts_provider(settings=self.settings)
        self.dsp_pipeline = IntegratedAudioPipeline(config=pipeline_config or AudioPipelineConfig())

    async def process_and_transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        reference_audio_bytes: Optional[bytes] = None,
        run_dsp: bool = True
    ) -> AudioTranscribeResult:
        """
        Processes audio bytes through the software DSP chain (noise suppression, VAD, AGC)
        and sends the clean PCM audio to the configured STT provider.
        """
        t0 = time.time()
        if not audio_bytes or len(audio_bytes) == 0:
            return AudioTranscribeResult(
                success=False,
                transcript="",
                confidence=0.0,
                error="Audio buffer is empty",
                latency_ms=0.0
            )

        # 1. Parse WAV if container format
        try:
            pcm_bytes, detected_sr, detected_ch = extract_pcm_from_wav_or_raw(audio_bytes, default_sample_rate=sample_rate)
            sample_rate = detected_sr
            channels = detected_ch
        except Exception as e:
            return AudioTranscribeResult(
                success=False,
                transcript="",
                confidence=0.0,
                error=f"Malformed audio data: {str(e)}",
                latency_ms=round((time.time() - t0) * 1000.0, 2)
            )

        if channels != 1:
            return AudioTranscribeResult(
                success=False,
                transcript="",
                confidence=0.0,
                error=f"Unsupported channel count: got {channels}, expected 1 (mono)",
                latency_ms=round((time.time() - t0) * 1000.0, 2)
            )

        dsp_metrics_dict = None
        vad_active = False
        target_pcm = pcm_bytes

        # 2. Run DSP pipeline if enabled
        if run_dsp:
            try:
                mic_frame = AudioFrame.from_bytes(pcm_bytes, sample_rate=sample_rate)
                ref_frame = None
                if reference_audio_bytes and len(reference_audio_bytes) > 0:
                    ref_pcm, ref_sr, _ = extract_pcm_from_wav_or_raw(reference_audio_bytes, default_sample_rate=sample_rate)
                    ref_frame = AudioFrame.from_bytes(ref_pcm, sample_rate=ref_sr)

                dsp_res = self.dsp_pipeline.process_frame(mic_frame, ref_frame=ref_frame)
                if not dsp_res.accepted:
                    return AudioTranscribeResult(
                        success=False,
                        transcript="",
                        confidence=0.0,
                        error=f"DSP pipeline rejected frame: {dsp_res.rejection_reason}",
                        latency_ms=round((time.time() - t0) * 1000.0, 2)
                    )

                target_pcm = dsp_res.output_bytes or pcm_bytes
                vad_active = dsp_res.metrics.vad_speech_active
                dsp_metrics_dict = {
                    "input_rms": dsp_res.metrics.input_rms,
                    "output_rms": dsp_res.metrics.output_rms,
                    "rms_gain_db": dsp_res.metrics.rms_gain_db,
                    "vad_speech_active": dsp_res.metrics.vad_speech_active,
                    "aec_status": dsp_res.metrics.aec_status,
                    "clipping_detected": dsp_res.metrics.clipping_detected,
                    "stages_executed": dsp_res.metrics.stages_executed
                }
            except Exception as e:
                return AudioTranscribeResult(
                    success=False,
                    transcript="",
                    confidence=0.0,
                    error=f"DSP processing error: {str(e)}",
                    latency_ms=round((time.time() - t0) * 1000.0, 2)
                )

        # 3. Transcribe via STT Provider
        stt_res = await self.stt_provider.transcribe(target_pcm, sample_rate=sample_rate)
        elapsed_ms = (time.time() - t0) * 1000.0

        is_success = stt_res.success
        status_str = stt_res.status.value if hasattr(stt_res.status, "value") else str(stt_res.status)
        err_msg = "; ".join(stt_res.errors) if stt_res.errors else None

        return AudioTranscribeResult(
            success=is_success,
            transcript=stt_res.text,
            confidence=stt_res.confidence,
            dsp_metrics=dsp_metrics_dict,
            vad_speech_active=vad_active,
            latency_ms=round(elapsed_ms, 2),
            error=err_msg if not is_success else None,
            raw_status=status_str
        )

    async def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None
    ) -> SpeechSynthesizeResult:
        """
        Synthesizes speech audio from text using the configured TTS provider.
        """
        t0 = time.time()
        if not text or not text.strip():
            return SpeechSynthesizeResult(
                success=False,
                error="Text to synthesize is empty",
                latency_ms=0.0
            )

        tts_res = await self.tts_provider.synthesize(text, voice=voice)
        elapsed_ms = (time.time() - t0) * 1000.0

        is_success = tts_res.success
        status_str = tts_res.status.value if hasattr(tts_res.status, "value") else str(tts_res.status)
        err_msg = "; ".join(tts_res.errors) if tts_res.errors else None

        return SpeechSynthesizeResult(
            success=is_success,
            audio_bytes=tts_res.audio_bytes,
            sample_rate=tts_res.sample_rate,
            encoding=tts_res.audio_format,
            channels=tts_res.channels,
            latency_ms=round(elapsed_ms, 2),
            error=err_msg if not is_success else None,
            raw_status=status_str
        )


_audio_service_instance: Optional[AudioService] = None

def get_audio_service(
    settings: Optional[AppSettings] = None,
    stt_provider: Optional[BaseSTTProvider] = None,
    tts_provider: Optional[BaseTTSProvider] = None
) -> AudioService:
    """Singleton getter for AudioService."""
    global _audio_service_instance
    if _audio_service_instance is None:
        _audio_service_instance = AudioService(
            settings=settings,
            stt_provider=stt_provider,
            tts_provider=tts_provider
        )
    return _audio_service_instance

def reset_audio_service(service: Optional[AudioService] = None) -> None:
    """Resets or overrides AudioService singleton."""
    global _audio_service_instance
    _audio_service_instance = service
