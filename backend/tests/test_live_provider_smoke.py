"""Opt-in live AI Provider integration smoke tests (credential-gated).

These tests run against real external APIs ONLY when NEXTSIGHT_LIVE_TEST=1
and corresponding API keys are explicitly supplied in the execution environment.
In standard unit/CI test runs, these tests are skipped to ensure 100% offline determinism
and prevent accidental paid API costs.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.providers.stt import WhisperSTTAdapter
from backend.providers.llm import APILLMAdapter
from backend.providers.tts import APITTSAdapter
from backend.providers.contracts import ProviderStatus
from backend.conversation.models import ConversationMessage, MessageRole

LIVE_TEST_ENABLED = os.getenv("NEXTSIGHT_LIVE_TEST", "0").lower() in ("1", "true", "yes")


@pytest.mark.anyio
@pytest.mark.skipif(not LIVE_TEST_ENABLED or not os.getenv("OPENAI_API_KEY"), reason="Requires NEXTSIGHT_LIVE_TEST=1 and OPENAI_API_KEY")
async def test_live_openai_whisper_stt_smoke():
    api_key = os.getenv("OPENAI_API_KEY")
    adapter = WhisperSTTAdapter(api_key=api_key)
    assert adapter.is_available() is True

    # 1 second of 16kHz silence
    pcm_silence = b"\x00" * 32000
    res = await adapter.transcribe(pcm_silence, sample_rate=16000)
    assert res.status in (ProviderStatus.OPERATIONAL, ProviderStatus.ERROR)


@pytest.mark.anyio
@pytest.mark.skipif(not LIVE_TEST_ENABLED or not os.getenv("OPENAI_API_KEY"), reason="Requires NEXTSIGHT_LIVE_TEST=1 and OPENAI_API_KEY")
async def test_live_openai_llm_smoke():
    api_key = os.getenv("OPENAI_API_KEY")
    adapter = APILLMAdapter(api_key=api_key, model_name="gpt-4o-mini")
    assert adapter.is_available() is True

    messages = [ConversationMessage(role=MessageRole.USER, content="Hello NextSight, reply with one word: 'READY'")]
    res = await adapter.generate_response(messages=messages)
    assert res.status == ProviderStatus.OPERATIONAL
    assert len(res.content) > 0


@pytest.mark.anyio
@pytest.mark.skipif(not LIVE_TEST_ENABLED or not os.getenv("OPENAI_API_KEY"), reason="Requires NEXTSIGHT_LIVE_TEST=1 and OPENAI_API_KEY")
async def test_live_openai_tts_smoke():
    api_key = os.getenv("OPENAI_API_KEY")
    adapter = APITTSAdapter(api_key=api_key, model_name="tts-1", voice_id="alloy")
    assert adapter.is_available() is True

    res = await adapter.synthesize(text="Testing NextSight text to speech.")
    assert res.status == ProviderStatus.OPERATIONAL
    assert len(res.audio_bytes) > 0
