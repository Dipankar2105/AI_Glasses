"""Large Language Model (LLM) provider adapters and abstractions."""

import json
import time
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

import httpx

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
        model_name: str = "mock-gpt-4o",
        consume_tool_calls: bool = True
    ):
        self.fixed_response = fixed_response
        self.simulated_tool_calls = list(simulated_tool_calls) if simulated_tool_calls else []
        self.simulated_delay_s = simulated_delay_s
        self.model_name = model_name
        self.consume_tool_calls = consume_tool_calls

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

        current_tool_calls = list(self.simulated_tool_calls)
        if self.consume_tool_calls and self.simulated_tool_calls:
            self.simulated_tool_calls.clear()

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
            tool_calls=current_tool_calls,
            model_name=self.model_name,
            token_usage=usage,
            latency_ms=elapsed
        )


class APILLMAdapter(BaseLLMProvider):
    """
    Standardized adapter for OpenAI and OpenAI-compatible cloud LLM APIs (e.g. Gemini, OpenAI, Claude).
    Validates API credentials, serializes conversation history, parses tool calls, and handles errors cleanly.
    """

    def __init__(
        self,
        provider_name: str = "openai",
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        api_base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.provider_name = provider_name
        self._api_key = api_key
        self.model_name = model_name
        self.api_base_url = (api_base_url or "https://api.openai.com/v1").rstrip("/")
        self._custom_client = http_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        return bool(self._api_key and self._api_key.strip())

    def _format_messages(
        self,
        messages: List[ConversationMessage],
        context_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Formats conversation messages into standard OpenAI chat completion schema."""
        formatted: List[Dict[str, Any]] = []

        # If visual context metadata is present, inject system context prompt
        if context_metadata:
            ctx_summary = []
            if "scene_description" in context_metadata:
                ctx_summary.append(f"Visual Scene: {context_metadata['scene_description']}")
            if "detected_objects" in context_metadata:
                ctx_summary.append(f"Detected Objects: {', '.join(context_metadata['detected_objects'])}")
            if ctx_summary:
                formatted.append({
                    "role": "system",
                    "content": "NextSight Smart Glasses Context:\n" + "\n".join(ctx_summary)
                })

        for m in messages:
            if m.role == MessageRole.USER:
                formatted.append({"role": "user", "content": m.content})
            elif m.role == MessageRole.ASSISTANT:
                formatted.append({"role": "assistant", "content": m.content})
            elif m.role == MessageRole.SYSTEM:
                formatted.append({"role": "system", "content": m.content})
            elif m.role == MessageRole.TOOL_RESULT:
                formatted.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id or f"call_{m.tool_name or 'tool'}",
                    "content": m.content
                })

        return formatted

    def _format_tools(self, available_tools: Optional[List[dict]]) -> Optional[List[Dict[str, Any]]]:
        """Converts internal tool definitions into OpenAI function tool schemas."""
        if not available_tools:
            return None
        tools = []
        for t in available_tools:
            name = t.get("name")
            if not name:
                continue
            desc = t.get("description", "")
            schema = t.get("input_schema", {"type": "object", "properties": {}})
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": schema
                }
            })
        return tools if tools else None

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

        url = f"{self.api_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._format_messages(messages, context_metadata),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        tools_schema = self._format_tools(available_tools)
        if tools_schema:
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"

        try:
            client = self._custom_client or httpx.AsyncClient(timeout=timeout_seconds)
            try:
                response = await client.post(url, headers=headers, json=payload)
            finally:
                if self._custom_client is None:
                    await client.aclose()

            elapsed = (time.time() - t0) * 1000.0

            if response.status_code in (401, 403):
                return LLMResult(
                    content="",
                    status=ProviderStatus.AUTH_FAILED,
                    latency_ms=elapsed,
                    errors=[f"Authentication failed: HTTP {response.status_code}"]
                )
            elif response.status_code == 429:
                return LLMResult(
                    content="",
                    status=ProviderStatus.RATE_LIMITED,
                    latency_ms=elapsed,
                    errors=["Rate limit exceeded (HTTP 429)"]
                )
            elif response.status_code >= 400:
                return LLMResult(
                    content="",
                    status=ProviderStatus.ERROR,
                    latency_ms=elapsed,
                    errors=[f"API error HTTP {response.status_code}: {response.text[:200]}"]
                )

            res_json = response.json()
            choices = res_json.get("choices", [])
            if not choices:
                return LLMResult(
                    content="",
                    status=ProviderStatus.ERROR,
                    latency_ms=elapsed,
                    errors=["Empty choices array in API response"]
                )

            msg_obj = choices[0].get("message", {})
            content = msg_obj.get("content") or ""

            # Parse requested tool calls
            tool_calls: List[LLMToolCall] = []
            raw_tcs = msg_obj.get("tool_calls", [])
            for tc in raw_tcs:
                fn = tc.get("function", {})
                fn_name = fn.get("name")
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {}
                if fn_name:
                    tool_calls.append(LLMToolCall(
                        tool_name=fn_name,
                        arguments=args,
                        call_id=tc.get("id")
                    ))

            usage = res_json.get("usage", {})
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }

            return LLMResult(
                content=content,
                status=ProviderStatus.OPERATIONAL,
                tool_calls=tool_calls,
                model_name=self.model_name,
                token_usage=token_usage,
                latency_ms=elapsed
            )

        except (httpx.TimeoutException, asyncio.TimeoutError):
            return LLMResult(
                content="",
                status=ProviderStatus.TIMEOUT,
                latency_ms=(time.time() - t0) * 1000.0,
                errors=[f"LLM request timed out after {timeout_seconds:.1f}s"]
            )
        except Exception as e:
            return LLMResult(
                content="",
                status=ProviderStatus.ERROR,
                latency_ms=(time.time() - t0) * 1000.0,
                errors=[f"LLM API connection failed: {str(e)}"]
            )
