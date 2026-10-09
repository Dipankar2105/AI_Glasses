from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class MCPToolDefinition(BaseModel):
    """MCP standard tool declaration."""
    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)

class MCPToolCallRequest(BaseModel):
    """Request to invoke an MCP tool."""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class MCPToolCallResult(BaseModel):
    """Result of an MCP tool invocation."""
    tool_name: str
    content: List[Dict[str, Any]] = Field(default_factory=list)
    is_error: bool = False
    error_message: Optional[str] = None
    latency_ms: float = 0.0

class JSONRPCRequest(BaseModel):
    """Standard JSON-RPC 2.0 request envelope used by MCP transports."""
    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    method: str
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)

class JSONRPCResponse(BaseModel):
    """Standard JSON-RPC 2.0 response envelope."""
    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
