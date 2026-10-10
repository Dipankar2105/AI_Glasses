"""AI Provider Adapters for Speech-to-Text, Text-to-Speech, and LLM services."""

from backend.providers.contracts import (
    ProviderStatus,
    STTResult,
    TTSResult,
    LLMResult,
    LLMToolCall,
)
from backend.providers.stt import (
    BaseSTTProvider,
    NullSTTProvider,
    MockSTTProvider,
    WhisperSTTAdapter,
)
from backend.providers.tts import (
    BaseTTSProvider,
    NullTTSProvider,
    MockTTSProvider,
    APITTSAdapter,
)
from backend.providers.llm import (
    BaseLLMProvider,
    NullLLMProvider,
    MockLLMProvider,
    APILLMAdapter,
)
from backend.providers.factory import (
    ProviderFactory,
    get_stt_provider,
    get_tts_provider,
    get_llm_provider,
)

__all__ = [
    "ProviderStatus",
    "STTResult",
    "TTSResult",
    "LLMResult",
    "LLMToolCall",
    "BaseSTTProvider",
    "NullSTTProvider",
    "MockSTTProvider",
    "WhisperSTTAdapter",
    "BaseTTSProvider",
    "NullTTSProvider",
    "MockTTSProvider",
    "APITTSAdapter",
    "BaseLLMProvider",
    "NullLLMProvider",
    "MockLLMProvider",
    "APILLMAdapter",
    "ProviderFactory",
    "get_stt_provider",
    "get_tts_provider",
    "get_llm_provider",
]
