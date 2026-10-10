"""Structured contracts and data models for NextSight AI Provider Adapters."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any


class ProviderStatus(str, Enum):
    """Operational status declarations for AI providers."""
    OPERATIONAL = "OPERATIONAL"
    MOCK_DEVELOPMENT = "MOCK_DEVELOPMENT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    ERROR = "ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    AUTH_FAILED = "AUTH_FAILED"


@dataclass
class STTResult:
    """Structured response from Speech-to-Text provider."""
    text: str
    confidence: float = 1.0
    language: str = "en"
    is_final: bool = True
    duration_ms: float = 0.0
    status: ProviderStatus = ProviderStatus.OPERATIONAL
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status in (ProviderStatus.OPERATIONAL, ProviderStatus.MOCK_DEVELOPMENT) and len(self.errors) == 0


@dataclass
class TTSResult:
    """Structured response from Text-to-Speech provider."""
    audio_bytes: bytes = b""
    sample_rate: int = 16000
    channels: int = 1
    audio_format: str = "pcm_s16le"
    duration_ms: float = 0.0
    status: ProviderStatus = ProviderStatus.OPERATIONAL
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status in (ProviderStatus.OPERATIONAL, ProviderStatus.MOCK_DEVELOPMENT) and len(self.errors) == 0


@dataclass
class LLMToolCall:
    """Tool invocation requested by LLM."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class LLMResult:
    """Structured response from Large Language Model provider."""
    content: str
    status: ProviderStatus = ProviderStatus.OPERATIONAL
    tool_calls: List[LLMToolCall] = field(default_factory=list)
    model_name: Optional[str] = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status in (ProviderStatus.OPERATIONAL, ProviderStatus.MOCK_DEVELOPMENT) and len(self.errors) == 0
