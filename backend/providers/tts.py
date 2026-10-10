"""Text-to-Speech (TTS) provider adapters and abstractions."""

import asyncio
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from backend.providers.contracts import TTSResult, ProviderStatus


class BaseTTSProvider(ABC):
    """Abstract interface for Text-to-Speech synthesis providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if provider is operational and configured."""
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: int = 16000,
        timeout_seconds: float = 10.0
    ) -> TTSResult:
        """Synthesizes text into audio bytes (e.g. PCM s16le)."""
        pass


class NullTTSProvider(BaseTTSProvider):
    """
    Honest null provider reporting TTS is unavailable / deferred.
    Never invents fabricated audio bytes when unconfigured.
    """

    def is_available(self) -> bool:
        return False

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: int = 16000,
        timeout_seconds: float = 10.0
    ) -> TTSResult:
        return TTSResult(
            audio_bytes=b"",
            sample_rate=sample_rate,
            channels=1,
            audio_format="pcm_s16le",
            duration_ms=0.0,
            status=ProviderStatus.PROVIDER_UNAVAILABLE,
            errors=["TTS provider is not configured or deferred in current runtime"]
        )


class MockTTSProvider(BaseTTSProvider):
    """
    Deterministic mock TTS provider for offline testing.
    Generates synthetic PCM s16le sine tone audio representing synthesized speech.
    """

    def __init__(
        self,
        simulated_delay_s: float = 0.0,
        default_sample_rate: int = 16000
    ):
        self.simulated_delay_s = simulated_delay_s
        self.default_sample_rate = default_sample_rate

    def is_available(self) -> bool:
        return True

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: Optional[int] = None,
        timeout_seconds: float = 10.0
    ) -> TTSResult:
        t0 = time.time()
        sr = sample_rate or self.default_sample_rate

        if not text or not text.strip():
            return TTSResult(
                audio_bytes=b"",
                sample_rate=sr,
                status=ProviderStatus.ERROR,
                errors=["Empty text provided for TTS synthesis"]
            )

        if self.simulated_delay_s > 0:
            if self.simulated_delay_s > timeout_seconds:
                await asyncio.sleep(timeout_seconds)
                return TTSResult(
                    audio_bytes=b"",
                    sample_rate=sr,
                    status=ProviderStatus.TIMEOUT,
                    duration_ms=(time.time() - t0) * 1000.0,
                    errors=[f"TTS synthesis timed out after {timeout_seconds:.1f}s"]
                )
            await asyncio.sleep(self.simulated_delay_s)

        # Generate deterministic synthetic PCM audio corresponding to text length (~15 chars/sec)
        duration_s = max(0.2, min(5.0, len(text) * 0.06 / max(0.5, speed)))
        num_samples = int(sr * duration_s)
        
        # 440 Hz gentle sine tone normalized to int16
        t = np.linspace(0, duration_s, num_samples, endpoint=False)
        audio_float = 0.3 * np.sin(2 * np.pi * 440.0 * t)
        audio_int16 = (audio_float * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        elapsed = (time.time() - t0) * 1000.0
        return TTSResult(
            audio_bytes=audio_bytes,
            sample_rate=sr,
            channels=1,
            audio_format="pcm_s16le",
            duration_ms=elapsed,
            status=ProviderStatus.MOCK_DEVELOPMENT
        )


class APITTSAdapter(BaseTTSProvider):
    """
    Adapter for remote TTS APIs (e.g. OpenAI TTS, ElevenLabs, Edge-TTS).
    Fails closed if credentials or network are unconfigured.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "default_voice",
        api_base_url: Optional[str] = None
    ):
        self._api_key = api_key
        self.voice_id = voice_id
        self.api_base_url = api_base_url

    def is_available(self) -> bool:
        return bool(self._api_key and self._api_key.strip())

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: int = 16000,
        timeout_seconds: float = 10.0
    ) -> TTSResult:
        t0 = time.time()
        if not self.is_available():
            return TTSResult(
                audio_bytes=b"",
                sample_rate=sample_rate,
                status=ProviderStatus.AUTH_FAILED,
                duration_ms=(time.time() - t0) * 1000.0,
                errors=["API TTS provider requires valid API key or credentials"]
            )

        if not text or not text.strip():
            return TTSResult(
                audio_bytes=b"",
                sample_rate=sample_rate,
                status=ProviderStatus.ERROR,
                duration_ms=(time.time() - t0) * 1000.0,
                errors=["Empty text provided for TTS synthesis"]
            )

        return TTSResult(
            audio_bytes=b"",
            sample_rate=sample_rate,
            status=ProviderStatus.PROVIDER_UNAVAILABLE,
            duration_ms=(time.time() - t0) * 1000.0,
            errors=["TTS API endpoint connection not configured in software runtime"]
        )
