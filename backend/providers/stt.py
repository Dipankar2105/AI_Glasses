"""Speech-to-Text (STT) provider adapters and abstractions."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Optional, Union, List

from backend.providers.contracts import STTResult, ProviderStatus


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
    Adapter for Whisper STT (local or remote API).
    Requires configured API key or local model weights.
    Fails closed with honest structured error if unconfigured.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "whisper-1",
        api_base_url: Optional[str] = None
    ):
        self._api_key = api_key
        self.model_name = model_name
        self.api_base_url = api_base_url

    def is_available(self) -> bool:
        return bool(self._api_key and self._api_key.strip())

    async def transcribe(
        self,
        audio_data: Union[bytes, memoryview],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        timeout_seconds: float = 10.0
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

        if not audio_data or len(audio_data) == 0:
            return STTResult(
                text="",
                confidence=0.0,
                status=ProviderStatus.ERROR,
                duration_ms=(time.time() - t0) * 1000.0,
                errors=["Empty audio data provided for transcription"]
            )

        # In software-only / offline test environment without live credentials
        return STTResult(
            text="",
            confidence=0.0,
            status=ProviderStatus.PROVIDER_UNAVAILABLE,
            duration_ms=(time.time() - t0) * 1000.0,
            errors=["Whisper API endpoint connection not configured in software runtime"]
        )
