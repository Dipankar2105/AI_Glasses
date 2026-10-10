"""Comprehensive tests for AI Provider Adapters (STT, TTS, LLM)."""

import os
import sys
import pytest
import asyncio

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
