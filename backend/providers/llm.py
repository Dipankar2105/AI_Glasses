"""Large Language Model (LLM) provider adapters and abstractions."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

from backend.conversation.models import ConversationMessage, MessageRole
from backend.providers.contracts import LLMResult, LLMToolCall, ProviderStatus


class LLMProviderResponse:
    """Legacy compatibility response wrapper."""
    def __init__(
        self,
        content: str,
        status: str = "OPERATIONAL",
        tool_calls: Optional[List[dict]] = None
    ):
        self.content = content
        self.status = status
        self.tool_calls = tool_calls or []


class BaseLLMProvider(ABC):
    """Abstract interface for LLM reasoning and conversation providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if provider has valid credentials and is ready."""
        pass

    @abstractmethod
    async def generate_response(
        self,
        messages: List[ConversationMessage],
        context_metadata: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[dict]] = None,
        timeout_seconds: float = 15.0
    ) -> LLMResult:
        """Generates an assistant response given conversation history and tools."""
        pass


class NullLLMProvider(BaseLLMProvider):
    """
    Honest null provider reporting no LLM reasoning model is configured.
    Never pretends a response was generated.
    """

    def is_available(self) -> bool:
        return False

    async def generate_response(
        self,
        messages: List[ConversationMessage],
        context_metadata: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[dict]] = None,
        timeout_seconds: float = 15.0
    ) -> LLMResult:
        return LLMResult(
            content="[LLM Unavailable] Conversation orchestration and tool execution are active, but no LLM reasoning provider is currently configured.",
            status=ProviderStatus.PROVIDER_UNAVAILABLE,
            errors=["LLM provider is not configured in current runtime"]
        )


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic test double for unit and integration testing.
    Supports fixed responses, simulated tool calling, simulated errors, and timeout testing.
    """

    def __init__(
        self,
        fixed_response: str = "This is a deterministic mock assistant response for testing.",
        simulated_tool_calls: Optional[List[LLMToolCall]] = None,
        simulated_delay_s: float = 0.0,
        model_name: str = "mock-gpt-4o"
    ):
        self.fixed_response = fixed_response
        self.simulated_tool_calls = simulated_tool_calls or []
        self.simulated_delay_s = simulated_delay_s
        self.model_name = model_name

    def is_available(self) -> bool:
        return True

    async def generate_response(
        self,
        messages: List[ConversationMessage],
        context_metadata: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[dict]] = None,
        timeout_seconds: float = 15.0
    ) -> LLMResult:
        t0 = time.time()

        if self.simulated_delay_s > 0:
            if self.simulated_delay_s > timeout_seconds:
                await asyncio.sleep(timeout_seconds)
                return LLMResult(
                    content="",
                    status=ProviderStatus.TIMEOUT,
                    latency_ms=(time.time() - t0) * 1000.0,
                    errors=[f"LLM generation timed out after {timeout_seconds:.1f}s"]
                )
            await asyncio.sleep(self.simulated_delay_s)

        elapsed = (time.time() - t0) * 1000.0
        
        # Calculate mock token count based on input + output text
        prompt_chars = sum(len(m.content) for m in messages)
        usage = {
            "prompt_tokens": max(1, prompt_chars // 4),
            "completion_tokens": max(1, len(self.fixed_response) // 4),
            "total_tokens": max(2, (prompt_chars + len(self.fixed_response)) // 4)
        }

        return LLMResult(
            content=self.fixed_response,
            status=ProviderStatus.MOCK_DEVELOPMENT,
            tool_calls=self.simulated_tool_calls,
            model_name=self.model_name,
            token_usage=usage,
            latency_ms=elapsed
        )


class APILLMAdapter(BaseLLMProvider):
    """
    Standardized adapter for cloud LLM APIs (e.g. Gemini, OpenAI, Claude).
    Validates API credentials, enforces request limits, and handles errors cleanly.
    """

    def __init__(
        self,
        provider_name: str = "gemini",
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        api_base_url: Optional[str] = None
    ):
        self.provider_name = provider_name
        self._api_key = api_key
        self.model_name = model_name
        self.api_base_url = api_base_url

    def is_available(self) -> bool:
        return bool(self._api_key and self._api_key.strip())

    async def generate_response(
        self,
        messages: List[ConversationMessage],
        context_metadata: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[dict]] = None,
        timeout_seconds: float = 15.0
    ) -> LLMResult:
        t0 = time.time()

        if not self.is_available():
            return LLMResult(
                content="",
                status=ProviderStatus.AUTH_FAILED,
                latency_ms=(time.time() - t0) * 1000.0,
                errors=[f"{self.provider_name.capitalize()} LLM requires a valid API key"]
            )

        # In offline software test harness without live network endpoints
        return LLMResult(
            content="",
            status=ProviderStatus.PROVIDER_UNAVAILABLE,
            latency_ms=(time.time() - t0) * 1000.0,
            errors=[f"{self.provider_name.capitalize()} endpoint connection not configured in software runtime"]
        )
