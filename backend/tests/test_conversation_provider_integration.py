"""Integration tests for Conversation Orchestration with AI Providers, Tool Loops, and Session Isolation."""

import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.config.settings import AppSettings
from backend.app import create_app
from backend.conversation.models import (
    MessageRole,
    ConversationMessage,
    ConversationMessageRequest,
)
from backend.conversation.session import SessionManager, reset_session_manager
from backend.mcp.registry import MCPToolRegistry, MCPTool, get_tool_registry
from backend.providers import (
    MockLLMProvider,
    NullLLMProvider,
    LLMToolCall,
    ProviderStatus,
)
from backend.services.conversation_service import (
    ConversationService,
    reset_conversation_service,
)
from backend.services.vision_service import reset_vision_service
from backend.power.policy import reset_power_policy_manager, get_power_policy_manager
from backend.power.contracts import BatteryTelemetry, ThermalTelemetry


@pytest.fixture(autouse=True)
def cleanup():
    reset_conversation_service()
    reset_session_manager()
    reset_vision_service()
    reset_power_policy_manager()
    yield
    reset_conversation_service()
    reset_session_manager()
    reset_vision_service()
    reset_power_policy_manager()


@pytest.mark.anyio
async def test_conversation_with_mock_llm_tool_calling_loop():
    """LLM requesting a tool call executes the tool and receives result in session history."""
    mgr = SessionManager(max_history_per_session=10)
    registry = MCPToolRegistry()
    
    # Configure MockLLM to return a tool call on first turn
    mock_tools = [LLMToolCall(tool_name="get_system_status", arguments={})]
    llm = MockLLMProvider(
        fixed_response="The system is healthy and ready.",
        simulated_tool_calls=mock_tools
    )
    
    service = ConversationService(session_manager=mgr, tool_registry=registry, llm_provider=llm, max_tool_iterations=3)
    
    req = ConversationMessageRequest(message="Check status")
    resp = await service.process_user_message(req)
    
    assert resp.session_id is not None
    assert resp.llm_status == "MOCK_DEVELOPMENT"
    assert resp.response == "The system is healthy and ready."
    assert len(resp.tool_executions) == 1
    assert resp.tool_executions[0].tool_name == "get_system_status"
    assert resp.tool_executions[0].success is True
    
    # Verify session history contains USER -> TOOL_RESULT -> ASSISTANT
    session = mgr.get_session(resp.session_id)
    assert len(session.messages) >= 3
    roles = [m.role for m in session.messages]
    assert MessageRole.USER in roles
    assert MessageRole.TOOL_RESULT in roles
    assert MessageRole.ASSISTANT in roles


@pytest.mark.anyio
async def test_conversation_recursion_guard():
    """Unbounded tool calling from an LLM is capped at max_tool_iterations."""
    mgr = SessionManager()
    registry = MCPToolRegistry()
    
    # Mock LLM that perpetually asks to call get_system_status
    mock_tools = [LLMToolCall(tool_name="get_system_status", arguments={})]
    llm = MockLLMProvider(
        fixed_response="Looping response",
        simulated_tool_calls=mock_tools,
        consume_tool_calls=False
    )
    
    service = ConversationService(session_manager=mgr, tool_registry=registry, llm_provider=llm, max_tool_iterations=2)
    
    req = ConversationMessageRequest(message="Trigger loop")
    resp = await service.process_user_message(req)
    
    # Must terminate without infinite loop
    assert resp.session_id is not None
    assert len(resp.tool_executions) <= 2


@pytest.mark.anyio
async def test_session_isolation_across_multiple_users():
    """Different session IDs maintain isolated message histories and context."""
    mgr = SessionManager()
    service = ConversationService(session_manager=mgr, llm_provider=MockLLMProvider(fixed_response="Echo"))
    
    # Session 1
    s1_resp = await service.process_user_message(ConversationMessageRequest(
        session_id="user-session-alpha",
        message="Message from Alpha",
        vision_context={"user": "alpha"}
    ))
    
    # Session 2
    s2_resp = await service.process_user_message(ConversationMessageRequest(
        session_id="user-session-beta",
        message="Message from Beta",
        vision_context={"user": "beta"}
    ))
    
    s1 = mgr.get_session("user-session-alpha")
    s2 = mgr.get_session("user-session-beta")
    
    assert s1.session_id == "user-session-alpha"
    assert s2.session_id == "user-session-beta"
    assert s1.context_metadata.get("user") == "alpha"
    assert s2.context_metadata.get("user") == "beta"
    assert s1.messages[0].content == "Message from Alpha"
    assert s2.messages[0].content == "Message from Beta"


@pytest.mark.anyio
async def test_vision_tool_calls_actual_vision_service_and_respects_safety():
    """analyze_vision_frame tool executes through VisionService and respects power policy."""
    from backend.services.vision_service import get_vision_service
    power_mgr = get_power_policy_manager(require_verified_telemetry=True)
    _ = get_vision_service(power_manager=power_mgr)
    registry = MCPToolRegistry()
    
    # 1. With no telemetry injected, strict power manager should deny vision capture
    tool_res = await registry.execute_tool("analyze_vision_frame", arguments={"use_synthetic_frame": True})
    assert tool_res.is_error is False # Returns result object containing errors
    content = tool_res.content
    assert len(content) > 0
    vision_data = content[0].get("data")
    assert len(vision_data.get("errors", [])) > 0
    assert vision_data["errors"][0]["error"] == "BLOCKED_BY_POWER_POLICY"
    
    # 2. Inject valid fresh telemetry
    now = time.time()
    power_mgr.update_battery_telemetry(BatteryTelemetry(
        voltage_volts=4.0,
        percentage=80.0,
        timestamp=now
    ), current_time=now)
    power_mgr.update_thermal_telemetry(ThermalTelemetry(
        temperature_celsius=36.0,
        timestamp=now
    ), current_time=now)
    
    tool_res2 = await registry.execute_tool("analyze_vision_frame", arguments={"use_synthetic_frame": True})
    content2 = tool_res2.content
    vision_data2 = content2[0].get("data")
    assert len(vision_data2.get("errors", [])) == 0
    assert len(vision_data2.get("detections", [])) > 0

