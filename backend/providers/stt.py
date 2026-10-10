"""Speech-to-Text (STT) provider adapters and abstractions."""

import io
import time
import wave
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Union, List

import httpx

from backend.providers.contracts import STTResult, ProviderStatus


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wraps raw 16-bit PCM bytes in a standard WAV header container."""
    # Check if already a WAV container (starts with RIFF)
    if pcm_bytes.startswith(b"RIFF") and len(pcm_bytes) > 44:
        return pcm_bytes

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_io.getvalue()


class BaseSTTProvider(ABC):
    """Abstract interface for Speech-to-Text providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if provider has valid credentials and is ready."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_data: Union[bytes, memoryview],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        timeout_seconds: float = 10.0
    ) -> STTResult:
        """Transcribes raw audio bytes into text."""
        pass


class NullSTTProvider(BaseSTTProvider):
    """
    Honest null provider reporting STT is unavailable / not configured.
    Never invents transcription text when no provider exists.
    """

    def is_available(self) -> bool:
        return False

    async def transcribe(
        self,
        audio_data: Union[bytes, memoryview],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        timeout_seconds: float = 10.0
    ) -> STTResult:
        return STTResult(
            text="",
            confidence=0.0,
            status=ProviderStatus.PROVIDER_UNAVAILABLE,
            errors=["STT provider is not configured or unavailable in current runtime"]
        )


class MockSTTProvider(BaseSTTProvider):
    """
    Deterministic mock STT provider for offline unit and integration testing.
    Can be configured with fixed transcription or echo mode.
    """

    def __init__(
        self,
        canned_transcription: str = "Hello NextSight",
        confidence: float = 0.98,
        simulated_delay_s: float = 0.0
    ):
        self.canned_transcription = canned_transcription
        self.confidence = confidence
        self.simulated_delay_s = simulated_delay_s

    def is_available(self) -> bool:
        return True

    async def transcribe(
        self,
        audio_data: Union[bytes, memoryview],
        sample_rate: int = 16000,
        language: Optional[str] = "en",
        timeout_seconds: float = 10.0
    ) -> STTResult:
        t0 = time.time()
        if not audio_data or len(audio_data) == 0:
            return STTResult(
                text="",
                confidence=0.0,
                status=ProviderStatus.ERROR,
                errors=["Empty audio data provided for transcription"]
            )

        if self.simulated_delay_s > 0:
            if self.simulated_delay_s > timeout_seconds:
                await asyncio.sleep(timeout_seconds)
                return STTResult(
                    text="",
                    confidence=0.0,
                    status=ProviderStatus.TIMEOUT,
                    duration_ms=(time.time() - t0) * 1000.0,
                    errors=[f"STT transcription timed out after {timeout_seconds:.1f}s"]
                )
            await asyncio.sleep(self.simulated_delay_s)

        elapsed = (time.time() - t0) * 1000.0
        return STTResult(
            text=self.canned_transcription,
            confidence=self.confidence,
            language=language or "en",
            is_final=True,
            duration_ms=elapsed,
            status=ProviderStatus.MOCK_DEVELOPMENT
        )


class WhisperSTTAdapter(BaseSTTProvider):
    """
    Real API Adapter for OpenAI Whisper (and compatible API endpoints).
    Sends multipart audio payload to /v1/audio/transcriptions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "whisper-1",
        api_base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ):
        self._api_key = api_key
        self.model_name = model_name
        self.api_base_url = (api_base_url or "https://api.openai.com/v1").rstrip("/")
        self._custom_client = http_client

    def is_available(self) -> bool:
        return bool(self._api_key and self._api_key.strip())

    async def transcribe(
        self,
        audio_data: Union[bytes, memoryview],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        timeout_seconds: float = 15.0
    ) -> STTResult:
        t0 = time.time()
        if not self.is_available():
            return STTResult(
                text="",
                confidence=0.0,
                status=ProviderStatus.AUTH_FAILED,
                duration_ms=(time.time() - t0) * 1000.0,
                errors=["Whisper STT requires valid API key or credentials"]
            )

        raw_bytes = bytes(audio_data) if isinstance(audio_data, memoryview) else audio_data
        if not raw_bytes or len(raw_bytes) == 0:
            return STTResult(
                text="",
                confidence=0.0,
                status=ProviderStatus.ERROR,
                duration_ms=(time.time() - t0) * 1000.0,
                errors=["Empty audio data provided for transcription"]
            )

        # Wrap in valid WAV container
        wav_payload = pcm_to_wav_bytes(raw_bytes, sample_rate=sample_rate)
        url = f"{self.api_base_url}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        data_fields = {"model": self.model_name, "response_format": "verbose_json"}
        if language:
            data_fields["language"] = language

        files = {"file": ("audio.wav", wav_payload, "audio/wav")}

        try:
            client = self._custom_client or httpx.AsyncClient(timeout=timeout_seconds)
            try:
                response = await client.post(url, headers=headers, data=data_fields, files=files)
            finally:
                if self._custom_client is None:
                    await client.aclose()

            elapsed = (time.time() - t0) * 1000.0

            if response.status_code in (401, 403):
                return STTResult(
                    text="",
                    confidence=0.0,
                    status=ProviderStatus.AUTH_FAILED,
                    duration_ms=elapsed,
                    errors=[f"Authentication failed: HTTP {response.status_code}"]
                )
            elif response.status_code == 429:
                return STTResult(
                    text="",
                    confidence=0.0,
                    status=ProviderStatus.RATE_LIMITED,
                    duration_ms=elapsed,
                    errors=["Rate limit exceeded (HTTP 429)"]
                )
            elif response.status_code >= 400:
                return STTResult(
                    text="",
                    confidence=0.0,
                    status=ProviderStatus.ERROR,
                    duration_ms=elapsed,
                    errors=[f"API error HTTP {response.status_code}: {response.text[:200]}"]
                )

            res_json = response.json()
            transcribed_text = res_json.get("text", "").strip()
            detected_lang = res_json.get("language", language or "en")

            # Extract average confidence if segments available
            segments = res_json.get("segments", [])
            confidence = 1.0
            if segments:
                no_speech_probs = [s.get("no_speech_prob", 0.0) for s in segments if "no_speech_prob" in s]
                if no_speech_probs:
                    confidence = max(0.0, min(1.0, 1.0 - (sum(no_speech_probs) / len(no_speech_probs))))

            return STTResult(
                text=transcribed_text,
                confidence=round(confidence, 3),
                language=detected_lang,
                is_final=True,
                duration_ms=elapsed,
                status=ProviderStatus.OPERATIONAL
            )

        except (httpx.TimeoutException, asyncio.TimeoutError):
            return STTResult(
                text="",
                confidence=0.0,
                status=ProviderStatus.TIMEOUT,
                duration_ms=(time.time() - t0) * 1000.0,
                errors=[f"Whisper STT request timed out after {timeout_seconds:.1f}s"]
            )
        except Exception as e:
            return STTResult(
                text="",
                confidence=0.0,
                status=ProviderStatus.ERROR,
                duration_ms=(time.time() - t0) * 1000.0,
                errors=[f"Whisper STT connection failed: {str(e)}"]
            )
