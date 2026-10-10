"""Factory and dependency injection helpers for AI Providers."""

import os
from typing import Optional

from backend.config.settings import AppSettings, get_settings
from backend.providers.contracts import ProviderStatus, STTResult, TTSResult, LLMResult
from backend.providers.stt import BaseSTTProvider, NullSTTProvider, MockSTTProvider, WhisperSTTAdapter
from backend.providers.tts import BaseTTSProvider, NullTTSProvider, MockTTSProvider, APITTSAdapter
from backend.providers.llm import BaseLLMProvider, NullLLMProvider, MockLLMProvider, APILLMAdapter


class ProviderFactory:
    """Instantiates configured AI providers based on environment settings."""

    @staticmethod
    def create_stt_provider(
        provider_name: Optional[str] = None,
        settings: Optional[AppSettings] = None
    ) -> BaseSTTProvider:
        cfg = settings or get_settings()
        name = (provider_name or os.getenv("NEXTSIGHT_STT_PROVIDER", "null")).lower().strip()

        if name == "mock":
            return MockSTTProvider()
        elif name == "whisper":
            api_key = os.getenv("OPENAI_API_KEY")
            return WhisperSTTAdapter(api_key=api_key)
        else:
            return NullSTTProvider()

    @staticmethod
    def create_tts_provider(
        provider_name: Optional[str] = None,
        settings: Optional[AppSettings] = None
    ) -> BaseTTSProvider:
        cfg = settings or get_settings()
        name = (provider_name or os.getenv("NEXTSIGHT_TTS_PROVIDER", "null")).lower().strip()

        if name == "mock":
            return MockTTSProvider()
        elif name in ("api", "openai", "elevenlabs"):
            api_key = os.getenv("TTS_API_KEY") or os.getenv("OPENAI_API_KEY")
            return APITTSAdapter(api_key=api_key)
        else:
            return NullTTSProvider()

    @staticmethod
    def create_llm_provider(
        provider_name: Optional[str] = None,
        settings: Optional[AppSettings] = None
    ) -> BaseLLMProvider:
        cfg = settings or get_settings()
        name = (provider_name or os.getenv("NEXTSIGHT_LLM_PROVIDER", "null")).lower().strip()

        if name == "mock":
            return MockLLMProvider()
        elif name in ("gemini", "openai", "claude", "api"):
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            return APILLMAdapter(provider_name=name, api_key=api_key)
        else:
            return NullLLMProvider()


def get_stt_provider(provider_name: Optional[str] = None) -> BaseSTTProvider:
    return ProviderFactory.create_stt_provider(provider_name)

def get_tts_provider(provider_name: Optional[str] = None) -> BaseTTSProvider:
    return ProviderFactory.create_tts_provider(provider_name)

def get_llm_provider(provider_name: Optional[str] = None) -> BaseLLMProvider:
    return ProviderFactory.create_llm_provider(provider_name)
