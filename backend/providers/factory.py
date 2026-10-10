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
        name = (
            provider_name
            or (settings.stt_provider if settings else None)
            or os.getenv("NEXTSIGHT_STT_PROVIDER")
            or (cfg.stt_provider if cfg else "null")
        ).lower().strip()

        if name == "mock":
            return MockSTTProvider()
        elif name in ("whisper", "openai"):
            api_key = (settings.openai_api_key if settings else None) or os.getenv("OPENAI_API_KEY") or (cfg.openai_api_key if cfg else None)
            return WhisperSTTAdapter(
                api_key=api_key,
                model_name=cfg.stt_model,
                api_base_url=cfg.stt_base_url
            )
        else:
            return NullSTTProvider()

    @staticmethod
    def create_tts_provider(
        provider_name: Optional[str] = None,
        settings: Optional[AppSettings] = None
    ) -> BaseTTSProvider:
        cfg = settings or get_settings()
        name = (
            provider_name
            or (settings.tts_provider if settings else None)
            or os.getenv("NEXTSIGHT_TTS_PROVIDER")
            or (cfg.tts_provider if cfg else "null")
        ).lower().strip()

        if name == "mock":
            return MockTTSProvider()
        elif name in ("api", "openai", "elevenlabs"):
            api_key = (
                (settings.tts_api_key or settings.openai_api_key if settings else None)
                or os.getenv("TTS_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or (cfg.tts_api_key or cfg.openai_api_key if cfg else None)
            )
            return APITTSAdapter(
                api_key=api_key,
                model_name=cfg.tts_model,
                voice_id=cfg.tts_voice,
                api_base_url=cfg.tts_base_url
            )
        else:
            return NullTTSProvider()

    @staticmethod
    def create_llm_provider(
        provider_name: Optional[str] = None,
        settings: Optional[AppSettings] = None
    ) -> BaseLLMProvider:
        cfg = settings or get_settings()
        name = (
            provider_name
            or (settings.llm_provider if settings else None)
            or os.getenv("NEXTSIGHT_LLM_PROVIDER")
            or (cfg.llm_provider if cfg else "null")
        ).lower().strip()

        if name == "mock":
            return MockLLMProvider()
        elif name in ("gemini", "openai", "claude", "anthropic", "api"):
            api_key = None
            if name == "gemini":
                api_key = (settings.gemini_api_key if settings else None) or os.getenv("GEMINI_API_KEY") or (cfg.gemini_api_key if cfg else None) or os.getenv("OPENAI_API_KEY")
            elif name in ("claude", "anthropic"):
                api_key = (settings.anthropic_api_key if settings else None) or os.getenv("ANTHROPIC_API_KEY") or (cfg.anthropic_api_key if cfg else None) or os.getenv("OPENAI_API_KEY")
            else:
                api_key = (settings.openai_api_key if settings else None) or os.getenv("OPENAI_API_KEY") or (cfg.openai_api_key if cfg else None)

            return APILLMAdapter(
                provider_name=name,
                api_key=api_key,
                model_name=cfg.llm_model,
                api_base_url=cfg.llm_base_url
            )
        else:
            return NullLLMProvider()


def get_stt_provider(provider_name: Optional[str] = None, settings: Optional[AppSettings] = None) -> BaseSTTProvider:
    return ProviderFactory.create_stt_provider(provider_name, settings)

def get_tts_provider(provider_name: Optional[str] = None, settings: Optional[AppSettings] = None) -> BaseTTSProvider:
    return ProviderFactory.create_tts_provider(provider_name, settings)

def get_llm_provider(provider_name: Optional[str] = None, settings: Optional[AppSettings] = None) -> BaseLLMProvider:
    return ProviderFactory.create_llm_provider(provider_name, settings)
