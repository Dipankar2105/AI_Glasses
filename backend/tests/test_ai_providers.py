"""Comprehensive tests for AI Provider Adapters (STT, TTS, LLM) including mock HTTP API behavior."""

import os
import sys
import json
import pytest
import asyncio
import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.conversation.models import ConversationMessage, MessageRole
from backend.providers import (
    ProviderStatus,
    STTResult,
    TTSResult,
    LLMResult,
    LLMToolCall,
    BaseSTTProvider,
    NullSTTProvider,
    MockSTTProvider,
    WhisperSTTAdapter,
    BaseTTSProvider,
    NullTTSProvider,
    MockTTSProvider,
    APITTSAdapter,
    BaseLLMProvider,
    NullLLMProvider,
    MockLLMProvider,
    APILLMAdapter,
    ProviderFactory,
    get_stt_provider,
    get_tts_provider,
    get_llm_provider,
)
from backend.providers.stt import pcm_to_wav_bytes


# ============================================================================
# 1. Speech-to-Text (STT) Provider Tests
# ============================================================================

@pytest.mark.anyio
async def test_null_stt_provider_reports_unavailable():
    provider = NullSTTProvider()
    assert provider.is_available() is False

    res = await provider.transcribe(b"dummy audio pcm data")
    assert res.success is False
    assert res.status == ProviderStatus.PROVIDER_UNAVAILABLE
    assert len(res.errors) > 0
    assert res.text == ""


@pytest.mark.anyio
async def test_mock_stt_provider_transcribes_valid_audio():
    provider = MockSTTProvider(canned_transcription="Recognized query", confidence=0.95)
    assert provider.is_available() is True

    # 1 second of 16kHz 16-bit PCM silence = 32000 bytes
    pcm_audio = b"\x00" * 32000
    res = await provider.transcribe(pcm_audio, sample_rate=16000)
    assert res.success is True
    assert res.status == ProviderStatus.MOCK_DEVELOPMENT
    assert res.text == "Recognized query"
    assert res.confidence == 0.95
    assert res.duration_ms >= 0.0


@pytest.mark.anyio
async def test_mock_stt_provider_empty_audio_rejected():
    provider = MockSTTProvider()
    res = await provider.transcribe(b"")
    assert res.success is False
    assert res.status == ProviderStatus.ERROR
    assert len(res.errors) > 0


@pytest.mark.anyio
async def test_mock_stt_provider_timeout_handling():
    provider = MockSTTProvider(simulated_delay_s=0.2)
    res = await provider.transcribe(b"\x00" * 100, timeout_seconds=0.05)
    assert res.success is False
    assert res.status == ProviderStatus.TIMEOUT
    assert len(res.errors) > 0


@pytest.mark.anyio
async def test_whisper_stt_adapter_fails_closed_without_credentials():
    provider = WhisperSTTAdapter(api_key=None)
    assert provider.is_available() is False

    res = await provider.transcribe(b"\x00" * 1000)
    assert res.success is False
    assert res.status == ProviderStatus.AUTH_FAILED
    assert "API key" in res.errors[0]


def test_pcm_to_wav_bytes_conversion():
    pcm_silence = b"\x00" * 1600
    wav_bytes = pcm_to_wav_bytes(pcm_silence, sample_rate=16000)
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes
    assert len(wav_bytes) == len(pcm_silence) + 44


@pytest.mark.anyio
async def test_whisper_stt_adapter_mock_http_success():
    mock_response_data = {
        "text": "Where is the nearest door?",
        "language": "english",
        "segments": [{"no_speech_prob": 0.05}]
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer test-key-123"
        return httpx.Response(200, json=mock_response_data)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)

    adapter = WhisperSTTAdapter(api_key="test-key-123", http_client=client)
    assert adapter.is_available() is True

    res = await adapter.transcribe(b"\x00" * 1600, sample_rate=16000)
    assert res.success is True
    assert res.text == "Where is the nearest door?"
    assert res.language == "english"
    assert res.confidence == 0.95
    assert res.status == ProviderStatus.OPERATIONAL


@pytest.mark.anyio
async def test_whisper_stt_adapter_mock_http_401_and_429():
    def handler_401(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Invalid API Key"})

    transport_401 = httpx.MockTransport(handler_401)
    client_401 = httpx.AsyncClient(transport=transport_401)
    adapter = WhisperSTTAdapter(api_key="bad-key", http_client=client_401)

    res_401 = await adapter.transcribe(b"\x00" * 1600)
    assert res_401.success is False
    assert res_401.status == ProviderStatus.AUTH_FAILED

    def handler_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "Rate limit exceeded"})

    transport_429 = httpx.MockTransport(handler_429)
    client_429 = httpx.AsyncClient(transport=transport_429)
    adapter_429 = WhisperSTTAdapter(api_key="valid-key", http_client=client_429)

    res_429 = await adapter_429.transcribe(b"\x00" * 1600)
    assert res_429.success is False
    assert res_429.status == ProviderStatus.RATE_LIMITED


# ============================================================================
# 2. Text-to-Speech (TTS) Provider Tests
# ============================================================================

@pytest.mark.anyio
async def test_null_tts_provider_reports_unavailable():
    provider = NullTTSProvider()
    assert provider.is_available() is False

    res = await provider.synthesize("Hello world")
    assert res.success is False
    assert res.status == ProviderStatus.PROVIDER_UNAVAILABLE
    assert len(res.errors) > 0
    assert len(res.audio_bytes) == 0


@pytest.mark.anyio
async def test_mock_tts_provider_synthesizes_valid_pcm():
    provider = MockTTSProvider(default_sample_rate=16000)
    assert provider.is_available() is True

    res = await provider.synthesize("Turn right in 200 meters")
    assert res.success is True
    assert res.status == ProviderStatus.MOCK_DEVELOPMENT
    assert len(res.audio_bytes) > 0
    # Must be 16-bit PCM (even number of bytes)
    assert len(res.audio_bytes) % 2 == 0
    assert res.sample_rate == 16000
    assert res.audio_format == "pcm_s16le"


@pytest.mark.anyio
async def test_mock_tts_provider_empty_text_rejected():
    provider = MockTTSProvider()
    res = await provider.synthesize("   ")
    assert res.success is False
    assert res.status == ProviderStatus.ERROR
    assert len(res.errors) > 0


@pytest.mark.anyio
async def test_mock_tts_provider_timeout_handling():
    provider = MockTTSProvider(simulated_delay_s=0.2)
    res = await provider.synthesize("Hello", timeout_seconds=0.05)
    assert res.success is False
    assert res.status == ProviderStatus.TIMEOUT
    assert len(res.errors) > 0


@pytest.mark.anyio
async def test_api_tts_adapter_fails_closed_without_credentials():
    provider = APITTSAdapter(api_key=None)
    assert provider.is_available() is False

    res = await provider.synthesize("Test speech synthesis")
    assert res.success is False
    assert res.status == ProviderStatus.AUTH_FAILED
    assert "API key" in res.errors[0]


@pytest.mark.anyio
async def test_api_tts_adapter_oversized_text_rejected():
    provider = APITTSAdapter(api_key="valid-dummy-key")
    long_text = "A" * 5000
    res = await provider.synthesize(long_text)
    assert res.success is False
    assert res.status == ProviderStatus.ERROR
    assert "exceeds maximum allowed length" in res.errors[0]


@pytest.mark.anyio
async def test_api_tts_adapter_mock_http_success():
    synthetic_pcm = b"\x10\x00" * 800

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer test-tts-key"
        body = json.loads(request.content.decode("utf-8"))
        assert body["response_format"] == "pcm"
        assert body["input"] == "Obstacle detected ahead"
        return httpx.Response(200, content=synthetic_pcm)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)
    adapter = APITTSAdapter(api_key="test-tts-key", http_client=client)

    res = await adapter.synthesize("Obstacle detected ahead")
    assert res.success is True
    assert res.status == ProviderStatus.OPERATIONAL
    assert res.audio_bytes == synthetic_pcm
    assert res.sample_rate == 24000
    assert res.audio_format == "pcm_s16le"


# ============================================================================
# 3. Large Language Model (LLM) Provider Tests
# ============================================================================

@pytest.mark.anyio
async def test_null_llm_provider_reports_unavailable():
    provider = NullLLMProvider()
    assert provider.is_available() is False

    messages = [ConversationMessage(role=MessageRole.USER, content="What is in front of me?")]
    res = await provider.generate_response(messages)
    assert res.success is False
    assert res.status == ProviderStatus.PROVIDER_UNAVAILABLE
    assert "[LLM Unavailable]" in res.content


@pytest.mark.anyio
async def test_mock_llm_provider_generates_response_with_tokens():
    mock_tools = [LLMToolCall(tool_name="analyze_vision_frame", arguments={"use_synthetic_frame": True})]
    provider = MockLLMProvider(
        fixed_response="I can see a notebook on the desk.",
        simulated_tool_calls=mock_tools
    )
    assert provider.is_available() is True

    messages = [ConversationMessage(role=MessageRole.USER, content="Describe the scene")]
    res = await provider.generate_response(messages)
    assert res.success is True
    assert res.status == ProviderStatus.MOCK_DEVELOPMENT
    assert res.content == "I can see a notebook on the desk."
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].tool_name == "analyze_vision_frame"
    assert "total_tokens" in res.token_usage
    assert res.token_usage["total_tokens"] > 0


@pytest.mark.anyio
async def test_mock_llm_provider_timeout_handling():
    provider = MockLLMProvider(simulated_delay_s=0.2)
    messages = [ConversationMessage(role=MessageRole.USER, content="Hello")]
    res = await provider.generate_response(messages, timeout_seconds=0.05)
    assert res.success is False
    assert res.status == ProviderStatus.TIMEOUT
    assert len(res.errors) > 0


@pytest.mark.anyio
async def test_api_llm_adapter_fails_closed_without_credentials():
    provider = APILLMAdapter(provider_name="gemini", api_key=None)
    assert provider.is_available() is False

    messages = [ConversationMessage(role=MessageRole.USER, content="Hello")]
    res = await provider.generate_response(messages)
    assert res.success is False
    assert res.status == ProviderStatus.AUTH_FAILED
    assert "API key" in res.errors[0]


@pytest.mark.anyio
async def test_api_llm_adapter_mock_http_success():
    mock_chat_res = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "You are facing an open doorway."
            }
        }],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 8,
            "total_tokens": 23
        }
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer test-llm-key"
        body = json.loads(request.content.decode("utf-8"))
        assert len(body["messages"]) == 1
        assert body["messages"][0]["content"] == "What is ahead?"
        return httpx.Response(200, json=mock_chat_res)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)
    adapter = APILLMAdapter(provider_name="openai", api_key="test-llm-key", http_client=client)

    messages = [ConversationMessage(role=MessageRole.USER, content="What is ahead?")]
    res = await adapter.generate_response(messages)
    assert res.success is True
    assert res.status == ProviderStatus.OPERATIONAL
    assert res.content == "You are facing an open doorway."
    assert res.token_usage["total_tokens"] == 23


@pytest.mark.anyio
async def test_api_llm_adapter_mock_http_tool_calls_parsing():
    mock_chat_res = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_12345",
                    "type": "function",
                    "function": {
                        "name": "analyze_vision_frame",
                        "arguments": json.dumps({"use_synthetic_frame": True})
                    }
                }]
            }
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32}
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=mock_chat_res)

    transport = httpx.MockTransport(mock_handler)
    client = httpx.AsyncClient(transport=transport)
    adapter = APILLMAdapter(provider_name="openai", api_key="test-llm-key", http_client=client)

    messages = [ConversationMessage(role=MessageRole.USER, content="Inspect the scene")]
    tools = [{
        "name": "analyze_vision_frame",
        "description": "Analyze vision frame",
        "input_schema": {"type": "object", "properties": {"use_synthetic_frame": {"type": "boolean"}}}
    }]
    res = await adapter.generate_response(messages, available_tools=tools)
    assert res.success is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].tool_name == "analyze_vision_frame"
    assert res.tool_calls[0].arguments == {"use_synthetic_frame": True}
    assert res.tool_calls[0].call_id == "call_12345"


# ============================================================================
# 4. Provider Factory & Environment Resolution Tests
# ============================================================================

def test_provider_factory_default_resolution():
    stt = ProviderFactory.create_stt_provider()
    assert isinstance(stt, NullSTTProvider)
    assert stt.is_available() is False

    tts = ProviderFactory.create_tts_provider()
    assert isinstance(tts, NullTTSProvider)
    assert tts.is_available() is False

    llm = ProviderFactory.create_llm_provider()
    assert isinstance(llm, NullLLMProvider)
    assert llm.is_available() is False


def test_provider_factory_mock_resolution(monkeypatch):
    monkeypatch.setenv("NEXTSIGHT_STT_PROVIDER", "mock")
    monkeypatch.setenv("NEXTSIGHT_TTS_PROVIDER", "mock")
    monkeypatch.setenv("NEXTSIGHT_LLM_PROVIDER", "mock")

    stt = get_stt_provider()
    assert isinstance(stt, MockSTTProvider)
    assert stt.is_available() is True

    tts = get_tts_provider()
    assert isinstance(tts, MockTTSProvider)
    assert tts.is_available() is True

    llm = get_llm_provider()
    assert isinstance(llm, MockLLMProvider)
    assert llm.is_available() is True
