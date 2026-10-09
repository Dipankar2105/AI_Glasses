from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse

from backend.conversation.models import (
    ConversationMessageRequest,
    ConversationMessageResponse,
    SessionDetailResponse
)
from backend.conversation.session import SessionManager, get_session_manager
from backend.services.conversation_service import ConversationService, get_conversation_service
from backend.mcp.models import (
    MCPToolDefinition,
    MCPToolCallRequest,
    MCPToolCallResult,
    JSONRPCRequest,
    JSONRPCResponse
)
from backend.mcp.registry import MCPToolRegistry, get_tool_registry
from backend.mcp.server import MCPServer, get_mcp_server

router = APIRouter(prefix="/api/v1", tags=["Conversation & MCP"])

def get_app_conversation_service(request: Request) -> ConversationService:
    if hasattr(request.app.state, "conversation_service") and request.app.state.conversation_service is not None:
        return request.app.state.conversation_service
    return get_conversation_service()

def get_app_session_manager(request: Request) -> SessionManager:
    if hasattr(request.app.state, "session_manager") and request.app.state.session_manager is not None:
        return request.app.state.session_manager
    return get_session_manager()

def get_app_tool_registry(request: Request) -> MCPToolRegistry:
    if hasattr(request.app.state, "tool_registry") and request.app.state.tool_registry is not None:
        return request.app.state.tool_registry
    return get_tool_registry()

def get_app_mcp_server(request: Request) -> MCPServer:
    if hasattr(request.app.state, "mcp_server") and request.app.state.mcp_server is not None:
        return request.app.state.mcp_server
    return get_mcp_server()


# --- Conversation Endpoints ---

@router.post("/conversation/message", response_model=ConversationMessageResponse)
async def send_conversation_message(
    request: ConversationMessageRequest,
    service: ConversationService = Depends(get_app_conversation_service)
):
    """
    Sends a user query into a conversation session, optionally executes registered
    tools, and returns the contextual orchestrated assistant response.
    """
    return await service.process_user_message(request)


@router.get("/conversation/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_details(
    session_id: str,
    manager: SessionManager = Depends(get_app_session_manager)
):
    """
    Retrieves history and context metadata for a specific conversation session.
    """
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found"
        )
    return SessionDetailResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=len(session.messages),
        messages=session.messages,
        context_metadata=session.context_metadata
    )


@router.delete("/conversation/sessions/{session_id}")
async def delete_session(
    session_id: str,
    manager: SessionManager = Depends(get_app_session_manager)
):
    """
    Deletes an active conversation session from in-memory storage.
    """
    deleted = manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found"
        )
    return {"status": "deleted", "session_id": session_id}


# --- MCP Tool Endpoints ---

@router.get("/mcp/tools", response_model=List[MCPToolDefinition])
async def list_mcp_tools(
    registry: MCPToolRegistry = Depends(get_app_tool_registry)
):
    """
    Lists all registered Model Context Protocol (MCP) tools conforming to schema.
    """
    return registry.list_tools()


@router.post("/mcp/tools/call", response_model=MCPToolCallResult)
async def call_mcp_tool(
    request: MCPToolCallRequest,
    registry: MCPToolRegistry = Depends(get_app_tool_registry)
):
    """
    Direct execution of a registered MCP tool with argument validation and timeout.
    """
    return await registry.execute_tool(request.name, request.arguments)


@router.post("/mcp/jsonrpc")
async def handle_mcp_jsonrpc(
    request: Dict[str, Any],
    server: MCPServer = Depends(get_app_mcp_server)
):
    """
    Standard JSON-RPC 2.0 protocol endpoint for MCP clients.
    """
    return await server.handle_jsonrpc(request)
