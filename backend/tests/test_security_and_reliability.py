"""Security, Resource Bounds, and API Reliability Tests."""

import time
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config.settings import AppSettings
from backend.conversation.session import SessionManager, reset_session_manager
from backend.conversation.models import MessageRole
from backend.services.conversation_service import ConversationService, reset_conversation_service
from backend.providers.llm import MockLLMProvider
from backend.mcp.registry import MCPToolRegistry


@pytest.fixture(autouse=True)
def cleanup():
    yield
    reset_session_manager()
    reset_conversation_service()


def test_public_endpoints_accessible_without_auth():
    """Verify health, ready, and capabilities endpoints remain accessible without auth."""
    settings = AppSettings(environment="test", api_key="secret-master-key-12345")
    app = create_app(settings)
    client = TestClient(app)

    res_h = client.get("/health")
    assert res_h.status_code == 200

    res_r = client.get("/ready")
    assert res_r.status_code == 200

    res_c = client.get("/api/v1/capabilities")
    assert res_c.status_code == 200


def test_api_key_auth_enforced_when_configured():
    """Verify protected endpoints require valid X-API-Key or Bearer token when configured."""
    settings = AppSettings(environment="test", api_key="secret-master-key-12345", llm_provider="mock")
    app = create_app(settings)
    app.state.conversation_service = ConversationService(llm_provider=MockLLMProvider())
    client = TestClient(app)

    # 1. Unauthenticated request -> 401
    res_unauth = client.post("/api/v1/conversation/message", json={"message": "Hello"})
    assert res_unauth.status_code == 401
    assert "UNAUTHORIZED" in res_unauth.json()["error"]

    # 2. Invalid key -> 401
    res_bad = client.post(
        "/api/v1/conversation/message",
        json={"message": "Hello"},
        headers={"X-API-Key": "wrong-key"}
    )
    assert res_bad.status_code == 401

    # 3. Valid X-API-Key header -> 200
    res_valid_hdr = client.post(
        "/api/v1/conversation/message",
        json={"message": "Hello"},
        headers={"X-API-Key": "secret-master-key-12345"}
    )
    assert res_valid_hdr.status_code == 200

    # 4. Valid Authorization Bearer header -> 200
    res_valid_bearer = client.post(
        "/api/v1/conversation/message",
        json={"message": "Hello"},
        headers={"Authorization": "Bearer secret-master-key-12345"}
    )
    assert res_valid_bearer.status_code == 200


def test_request_payload_size_limit_middleware():
    """Verify Content-Length exceeding configured max_request_body_bytes is rejected with 413."""
    settings = AppSettings(environment="test", max_request_body_bytes=1000)  # 1 KB limit
    app = create_app(settings)
    client = TestClient(app)

    res = client.post(
        "/api/v1/audio/transcribe",
        json={"audio_base64": "A" * 2000},
        headers={"Content-Length": "2050"}
    )
    assert res.status_code == 413
    assert "PAYLOAD_TOO_LARGE" in res.json()["error"]


def test_rate_limiting_enforcement():
    """Verify exceeding rate limit triggers 429 Too Many Requests."""
    settings = AppSettings(environment="test", rate_limit_per_minute=5)
    app = create_app(settings)
    client = TestClient(app)

    # First 5 requests should pass
    for _ in range(5):
        res = client.get("/health")
        assert res.status_code == 200

    # 6th request within same minute should be rate limited
    res_limited = client.get("/health")
    assert res_limited.status_code == 429
    assert res_limited.json()["error"] == "RATE_LIMIT_EXCEEDED"
    assert "Retry-After" in res_limited.headers


def test_session_history_bounding_prevents_memory_leak():
    """Verify session manager enforces max_history limit and retains system prompt."""
    mgr = SessionManager(max_history_per_session=5)
    sess_id = "bounded-sess"

    # Add system prompt
    mgr.add_message(sess_id, MessageRole.SYSTEM, "You are NextSight AI")

    # Add 10 user/assistant turns
    for i in range(10):
        mgr.add_message(sess_id, MessageRole.USER, f"User message {i}")
        mgr.add_message(sess_id, MessageRole.ASSISTANT, f"Assistant reply {i}")

    session = mgr.get_session(sess_id)
    assert len(session.messages) == 5
    # System prompt at index 0 must be preserved
    assert session.messages[0].role == MessageRole.SYSTEM
    assert session.messages[0].content == "You are NextSight AI"
    # Oldest user messages discarded, newest retained
    assert session.messages[-1].content == "Assistant reply 9"


@pytest.mark.anyio
async def test_mcp_tool_execution_timeout_and_sandboxing():
    """Verify MCP tool registry handles execution timeouts and isolates failures."""
    registry = MCPToolRegistry()

    # Tool that exceeds timeout
    import asyncio
    from backend.mcp.registry import MCPTool

    async def slow_tool():
        await asyncio.sleep(0.5)
        return "done"

    registry.register_tool(MCPTool(
        name="slow_tool",
        description="Slow test tool",
        input_schema={"type": "object", "properties": {}},
        handler=slow_tool
    ))

    res = await registry.execute_tool("slow_tool", arguments={}, timeout_seconds=0.05)
    assert res.is_error is True
    assert "timed out" in res.error_message


def test_secrets_filtering_and_safe_error_messages(client=None):
    """Verify secrets and API keys are never printed in status or error responses."""
    settings = AppSettings(
        environment="test",
        openai_api_key="sk-proj-supersecretkey999888777",
        gemini_api_key="AIzaSySuperSecretGeminiKey12345",
        anthropic_api_key="sk-ant-SecretAnthropicKey555444"
    )
    app = create_app(settings)
    c = TestClient(app)

    res = c.get("/api/v1/providers/status")
    assert res.status_code == 200
    res_text = res.text

    assert "sk-proj" not in res_text
    assert "supersecret" not in res_text.lower()
    assert "AIzaSy" not in res_text
    assert "sk-ant" not in res_text
