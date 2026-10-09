import os
import sys
import pytest
import asyncio
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.config.settings import AppSettings
from backend.app import create_app
from backend.conversation.models import (
    MessageRole,
    ConversationMessage,
    ConversationMessageRequest
)
from backend.conversation.session import SessionManager, get_session_manager
from backend.mcp.models import MCPToolDefinition, MCPToolCallRequest
from backend.mcp.registry import MCPToolRegistry, MCPTool, get_tool_registry
from backend.mcp.server import MCPServer, get_mcp_server
from backend.services.conversation_service import (
    ConversationService,
    NullLLMProvider,
    MockLLMProvider
)

@pytest.fixture
def test_app():
    settings = AppSettings(
        app_name="NextSight Test Backend",
        version="0.7.0-test",
        environment="test",
        host="127.0.0.1",
        port=8000,
        log_level="DEBUG"
    )
    app = create_app(settings)
    return app

@pytest.fixture
def client(test_app):
    with TestClient(test_app) as client:
        yield client


# --- 1. Session Manager Tests ---

def test_session_creation_and_retrieval():
    mgr = SessionManager(max_history_per_session=5)
    s1 = mgr.get_or_create_session()
    assert s1.session_id is not None
    assert len(s1.messages) == 0

    s2 = mgr.get_session(s1.session_id)
    assert s2 is not None
    assert s2.session_id == s1.session_id

    # Unknown session returns None
    assert mgr.get_session("non-existent-id") is None

def test_session_message_bounding_and_trimming():
    mgr = SessionManager(max_history_per_session=4)
    s = mgr.get_or_create_session("session-bound-test")

    # Add 6 messages
    for i in range(6):
        mgr.add_message(s.session_id, MessageRole.USER, f"Message {i}")

    retrieved = mgr.get_session("session-bound-test")
    assert len(retrieved.messages) == 4
    # Ensure oldest messages were trimmed
    assert retrieved.messages[0].content == "Message 2"
    assert retrieved.messages[3].content == "Message 5"

def test_session_deletion():
    mgr = SessionManager()
    s = mgr.get_or_create_session("delete-me")
    assert mgr.delete_session("delete-me") is True
    assert mgr.get_session("delete-me") is None
    assert mgr.delete_session("delete-me") is False


# --- 2. MCP Tool Registry & Execution Tests ---

@pytest.mark.anyio
async def test_mcp_registry_builtin_tools():
    registry = MCPToolRegistry()
    tools = registry.list_tools()
    tool_names = [t.name for t in tools]

    assert "get_system_status" in tool_names
    assert "analyze_vision_frame" in tool_names
    assert "get_conversation_context" in tool_names

@pytest.mark.anyio
async def test_mcp_tool_execution_success():
    registry = MCPToolRegistry()
    res = await registry.execute_tool("get_system_status")
    assert res.is_error is False
    assert len(res.content) > 0
    assert "app_name" in res.content[0]["data"]
    assert res.latency_ms >= 0.0

@pytest.mark.anyio
async def test_mcp_tool_execution_unknown():
    registry = MCPToolRegistry()
    res = await registry.execute_tool("non_existent_tool")
    assert res.is_error is True
    assert "Unknown tool" in res.error_message

@pytest.mark.anyio
async def test_mcp_tool_execution_timeout():
    registry = MCPToolRegistry()
    
    async def slow_handler():
        await asyncio.sleep(0.5)
        return "done"

    registry.register_tool(MCPTool(
        name="slow_tool",
        description="Slow test tool",
        input_schema={},
        handler=slow_handler
    ))

    res = await registry.execute_tool("slow_tool", timeout_seconds=0.05)
    assert res.is_error is True
    assert "timed out" in res.error_message


# --- 3. MCP JSON-RPC Server Protocol Tests ---

@pytest.mark.anyio
async def test_mcp_jsonrpc_tools_list():
    server = MCPServer()
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    resp = await server.handle_jsonrpc(req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "tools" in resp["result"]
    assert len(resp["result"]["tools"]) >= 3

@pytest.mark.anyio
async def test_mcp_jsonrpc_tools_call():
    server = MCPServer()
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "get_system_status", "arguments": {}}
    }
    resp = await server.handle_jsonrpc(req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 2
    assert resp["result"]["is_error"] is False
    assert resp["result"]["tool"] == "get_system_status"

@pytest.mark.anyio
async def test_mcp_jsonrpc_invalid_method():
    server = MCPServer()
    req = {"jsonrpc": "2.0", "id": 3, "method": "unknown_rpc_method"}
    resp = await server.handle_jsonrpc(req)

    assert resp["jsonrpc"] == "2.0"
    assert "error" in resp
    assert resp["error"]["code"] == -32601


# --- 4. Conversation Service & Honest LLM Reporting Tests ---

@pytest.mark.anyio
async def test_conversation_service_with_null_llm():
    mgr = SessionManager()
    reg = MCPToolRegistry()
    service = ConversationService(session_manager=mgr, tool_registry=reg, llm_provider=NullLLMProvider())

    req = ConversationMessageRequest(message="Hello glasses!")
    resp = await service.process_user_message(req)

    assert resp.session_id is not None
    assert resp.llm_status == "PROVIDER_UNAVAILABLE"
    assert "no LLM reasoning provider" in resp.response
    assert resp.session_message_count == 2 # 1 User + 1 Assistant

@pytest.mark.anyio
async def test_conversation_service_with_tool_invocation():
    mgr = SessionManager()
    reg = MCPToolRegistry()
    service = ConversationService(session_manager=mgr, tool_registry=reg, llm_provider=NullLLMProvider())

    req = ConversationMessageRequest(
        message="What is the system status?",
        tool_to_invoke="get_system_status"
    )
    resp = await service.process_user_message(req)

    assert len(resp.tool_executions) == 1
    assert resp.tool_executions[0].tool_name == "get_system_status"
    assert resp.tool_executions[0].success is True
    assert resp.session_message_count == 3 # 1 User + 1 ToolResult + 1 Assistant


# --- 5. API Route Endpoints Tests ---

def test_api_send_conversation_message(client):
    payload = {"message": "Hello from API test"}
    response = client.post("/api/v1/conversation/message", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["llm_status"] == "PROVIDER_UNAVAILABLE"
    assert data["session_message_count"] == 2

def test_api_get_session_details(client):
    # Create message to spawn session
    post_res = client.post("/api/v1/conversation/message", json={"message": "First message"})
    session_id = post_res.json()["session_id"]

    # Retrieve session details
    get_res = client.get(f"/api/v1/conversation/sessions/{session_id}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["session_id"] == session_id
    assert data["message_count"] >= 2
    assert len(data["messages"]) >= 2

def test_api_get_nonexistent_session(client):
    response = client.get("/api/v1/conversation/sessions/invalid-session-999")
    assert response.status_code == 404

def test_api_delete_session(client):
    post_res = client.post("/api/v1/conversation/message", json={"message": "Temporary session"})
    session_id = post_res.json()["session_id"]

    del_res = client.delete(f"/api/v1/conversation/sessions/{session_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Confirm deletion
    get_res = client.get(f"/api/v1/conversation/sessions/{session_id}")
    assert get_res.status_code == 404

def test_api_list_mcp_tools(client):
    response = client.get("/api/v1/mcp/tools")
    assert response.status_code == 200
    tools = response.json()
    assert len(tools) >= 3
    names = [t["name"] for t in tools]
    assert "get_system_status" in names

def test_api_call_mcp_tool(client):
    payload = {"name": "get_system_status", "arguments": {}}
    response = client.post("/api/v1/mcp/tools/call", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tool_name"] == "get_system_status"
    assert data["is_error"] is False

def test_api_mcp_jsonrpc_protocol(client):
    payload = {"jsonrpc": "2.0", "id": 42, "method": "ping"}
    response = client.post("/api/v1/mcp/jsonrpc", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 42
    assert data["result"]["status"] == "pong"
