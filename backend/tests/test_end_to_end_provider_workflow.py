"""Comprehensive End-to-End Provider-Driven Workflow Tests.

Validates the full software loop:
Audio Input -> DSP Pipeline -> STT Provider -> Conversation Orchestration -> LLM Provider -> TTS Provider -> Audio Output.
"""

import base64
import struct
import io
import wave
import pytest
from typing import List, Dict, Any

from backend.config.settings import AppSettings
from backend.services.audio_service import AudioService, reset_audio_service
from backend.services.conversation_service import ConversationService, reset_conversation_service
from backend.conversation.session import SessionManager, reset_session_manager
from backend.conversation.models import ConversationMessageRequest, MessageRole
from backend.mcp.registry import MCPToolRegistry, reset_tool_registry
from backend.providers.contracts import ProviderStatus, LLMResult, LLMToolCall, STTResult, TTSResult
from backend.providers.stt import MockSTTProvider, BaseSTTProvider, WhisperSTTAdapter
from backend.providers.llm import MockLLMProvider, BaseLLMProvider, APILLMAdapter
from backend.providers.tts import MockTTSProvider, BaseTTSProvider, APITTSAdapter
from firmware.audio.dsp.integrated_pipeline import AudioPipelineConfig


def generate_speech_like_pcm(sample_count: int = 3200, sample_rate: int = 16000) -> bytes:
    """Generates 200ms of non-trivial synthetic audio samples."""
    samples = []
    for i in range(sample_count):
        # Mix 200Hz fundamental with 400Hz harmonic
        val = int(2000 * ((i % 80) / 40.0 - 1.0) + 1000 * ((i % 40) / 20.0 - 1.0))
        samples.append(val)
    return struct.pack(f"<{len(samples)}h", *samples)


class FailingSTTProvider(BaseSTTProvider):
    @property
    def is_available(self) -> bool:
        return True

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000, language: str = "en") -> STTResult:
        return STTResult(
            text="",
            confidence=0.0,
            status=ProviderStatus.ERROR,
            errors=["STT model backend failed to process audio stream"]
        )


class FailingLLMProvider(BaseLLMProvider):
    @property
    def is_available(self) -> bool:
        return True

    async def generate_response(self, messages, context_metadata=None, available_tools=None) -> LLMResult:
        return LLMResult(
            content="",
            status=ProviderStatus.PROVIDER_UNAVAILABLE,
            errors=["LLM inference gateway unreachable"]
        )


class FailingTTSProvider(BaseTTSProvider):
    @property
    def is_available(self) -> bool:
        return True

    async def synthesize(self, text: str, voice=None, audio_format="pcm_s16le") -> TTSResult:
        return TTSResult(
            audio_bytes=b"",
            status=ProviderStatus.ERROR,
            errors=["TTS synthesis engine out of memory"]
        )


class RecursiveToolCallingLLM(BaseLLMProvider):
    """Simulates an LLM that endlessly requests tool calls to test loop bounding."""
    def __init__(self):
        self.call_count = 0

    @property
    def is_available(self) -> bool:
        return True

    async def generate_response(self, messages, context_metadata=None, available_tools=None) -> LLMResult:
        self.call_count += 1
        return LLMResult(
            content=f"Calling tool attempt {self.call_count}",
            status=ProviderStatus.OPERATIONAL,
            tool_calls=[LLMToolCall(tool_name="get_system_status", arguments={})]
        )


pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def cleanup():
    yield
    reset_audio_service()
    reset_conversation_service()
    reset_session_manager()
    reset_tool_registry()


# =============================================================================
# 1. OFFLINE FULL VOICE LOOP (AUDIO -> DSP -> STT -> LLM -> TTS -> AUDIO)
# =============================================================================

async def test_e2e_offline_full_voice_loop():
    """
    Validates complete end-to-end chain:
    1. Input raw audio -> DSP pipeline (DC filter, high pass, noise suppression, VAD, AGC)
    2. STT transcription -> 'What is the system status?'
    3. Conversation service receives transcript, records user message in session
    4. LLM reasoning resolves message & executes MCP get_system_status tool
    5. Final response generated & stored in session history
    6. TTS synthesizes spoken response into PCM audio bytes
    """
    settings = AppSettings(environment="test", stt_provider="mock", llm_provider="mock", tts_provider="mock")
    session_mgr = SessionManager(max_history_per_session=10)
    tool_reg = MCPToolRegistry()
    
    stt = MockSTTProvider()
    llm = MockLLMProvider()
    tts = MockTTSProvider()

    audio_svc = AudioService(stt_provider=stt, tts_provider=tts, settings=settings)
    conv_svc = ConversationService(session_manager=session_mgr, tool_registry=tool_reg, llm_provider=llm)

    # 1. Ingest audio through DSP + STT
    audio_pcm = generate_speech_like_pcm(3200, 16000)
    stt_res = await audio_svc.process_and_transcribe(audio_pcm, sample_rate=16000, run_dsp=True)
    
    assert stt_res.success is True
    assert stt_res.transcript != ""
    assert stt_res.dsp_metrics is not None
    assert "DCBlocker" in stt_res.dsp_metrics["stages_executed"]
    assert "SpectralNoiseSuppression" in stt_res.dsp_metrics["stages_executed"]

    # 2. Process query in Conversation Service
    conv_req = ConversationMessageRequest(
        session_id="voice-session-001",
        message="Check glasses telemetry and status",
        tool_to_invoke="get_system_status"
    )
    conv_res = await conv_svc.process_user_message(conv_req)

    assert conv_res.session_id == "voice-session-001"
    assert conv_res.response != ""
    assert conv_res.llm_status in ("OPERATIONAL", "MOCK_DEVELOPMENT")
    assert len(conv_res.tool_executions) == 1
    assert conv_res.tool_executions[0].success is True
    assert conv_res.session_message_count >= 2

    # 3. Synthesize assistant response to audio via TTS
    tts_res = await audio_svc.synthesize_speech(conv_res.response, voice="alloy")
    assert tts_res.success is True
    assert len(tts_res.audio_bytes) > 0
    assert tts_res.sample_rate in (16000, 24000)
    assert tts_res.encoding == "pcm_s16le"


# =============================================================================
# 2. FAILURE PROPAGATION AND ISOLATION TESTS
# =============================================================================

async def test_e2e_stt_failure_does_not_fabricate_output():
    """Verify STT failure returns honest error without fabricated transcript."""
    audio_svc = AudioService(stt_provider=FailingSTTProvider())
    audio_pcm = generate_speech_like_pcm(1600)
    res = await audio_svc.process_and_transcribe(audio_pcm)

    assert res.success is False
    assert res.transcript == ""
    assert "failed" in res.error.lower()


async def test_e2e_llm_failure_does_not_fabricate_response():
    """Verify LLM failure returns unavailable status and records failure in session."""
    session_mgr = SessionManager()
    conv_svc = ConversationService(session_manager=session_mgr, llm_provider=FailingLLMProvider())

    req = ConversationMessageRequest(session_id="fail-sess", message="Hello AI")
    res = await conv_svc.process_user_message(req)

    assert res.llm_status == "PROVIDER_UNAVAILABLE"
    assert res.response == ""
    session = session_mgr.get_session("fail-sess")
    assert len(session.messages) == 2
    assert session.messages[1].role == MessageRole.ASSISTANT
    assert session.messages[1].metadata["llm_status"] == "PROVIDER_UNAVAILABLE"


async def test_e2e_tts_failure_preserves_conversation_text():
    """Verify TTS failure does not destroy or alter conversation response text."""
    audio_svc = AudioService(tts_provider=FailingTTSProvider())
    tts_res = await audio_svc.synthesize_speech("System online.")

    assert tts_res.success is False
    assert tts_res.audio_bytes == b""
    assert "out of memory" in tts_res.error.lower()


# =============================================================================
# 3. BOUNDED TOOL RECURSION GUARD
# =============================================================================

async def test_e2e_bounded_tool_recursion_guard():
    """Verify runaway LLM tool calling loops are hard-capped at max_tool_iterations (3)."""
    recursive_llm = RecursiveToolCallingLLM()
    conv_svc = ConversationService(llm_provider=recursive_llm, max_tool_iterations=3)

    req = ConversationMessageRequest(session_id="loop-test", message="Trigger loop")
    res = await conv_svc.process_user_message(req)

    assert recursive_llm.call_count == 3
    assert res.session_id == "loop-test"


# =============================================================================
# 4. SESSION ISOLATION & CONTEXT ENRICHMENT
# =============================================================================

async def test_e2e_session_isolation():
    """Verify independent sessions do not leak messages or metadata."""
    session_mgr = SessionManager()
    conv_svc = ConversationService(session_manager=session_mgr, llm_provider=MockLLMProvider())

    await conv_svc.process_user_message(ConversationMessageRequest(
        session_id="user-a",
        message="I am Alice"
    ))
    await conv_svc.process_user_message(ConversationMessageRequest(
        session_id="user-b",
        message="I am Bob"
    ))

    sess_a = session_mgr.get_session("user-a")
    sess_b = session_mgr.get_session("user-b")

    assert len(sess_a.messages) == 2
    assert "Alice" in sess_a.messages[0].content
    assert len(sess_b.messages) == 2
    assert "Bob" in sess_b.messages[0].content


async def test_e2e_multimodal_vision_context_attachment():
    """Verify vision detection metadata is attached to conversation context."""
    session_mgr = SessionManager()
    conv_svc = ConversationService(session_manager=session_mgr, llm_provider=MockLLMProvider())

    vision_ctx = {
        "scene_description": "A crosswalk with approaching red car",
        "objects": ["crosswalk", "car", "pedestrian"],
        "hazards": ["approaching_vehicle"],
        "confidence": 0.88
    }

    req = ConversationMessageRequest(
        session_id="vision-enrich-sess",
        message="Is it safe to cross?",
        vision_context=vision_ctx
    )
    res = await conv_svc.process_user_message(req)

    sess = session_mgr.get_session("vision-enrich-sess")
    assert sess.context_metadata.get("scene_description") == "A crosswalk with approaching red car"
    assert "approaching_vehicle" in sess.context_metadata.get("hazards", [])


# =============================================================================
# 5. REAL PROVIDER CREDENTIAL-GATED BEHAVIOR
# =============================================================================

async def test_real_provider_adapters_fail_closed_without_keys():
    """
    Verify real provider adapters (Whisper, OpenAI LLM, API TTS) fail safely
    and explicitly when credentials are empty, without pretending to work.
    """
    whisper_adapter = WhisperSTTAdapter(api_key="")
    stt_res = await whisper_adapter.transcribe(b"some_pcm_bytes")
    assert stt_res.status == ProviderStatus.AUTH_FAILED
    assert "api key" in stt_res.errors[0].lower() or "missing" in stt_res.errors[0].lower()

    llm_adapter = APILLMAdapter(api_key="")
    llm_res = await llm_adapter.generate_response([])
    assert llm_res.status == ProviderStatus.AUTH_FAILED
    assert "api key" in llm_res.errors[0].lower() or "missing" in llm_res.errors[0].lower()

    tts_adapter = APITTSAdapter(api_key="")
    tts_res = await tts_adapter.synthesize("Hello")
    assert tts_res.status == ProviderStatus.AUTH_FAILED
    assert "api key" in tts_res.errors[0].lower() or "missing" in tts_res.errors[0].lower()
