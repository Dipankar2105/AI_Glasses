import json
from typing import Dict, Any, Optional

from backend.mcp.models import JSONRPCRequest, JSONRPCResponse
from backend.mcp.registry import MCPToolRegistry, get_tool_registry

class MCPServer:
    """
    Local Model Context Protocol (MCP) server dispatcher conforming to JSON-RPC 2.0.
    Enables standard protocol interactions for tool discovery and execution.
    """
    def __init__(self, registry: Optional[MCPToolRegistry] = None):
        self.registry = registry or get_tool_registry()

    async def handle_jsonrpc(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatches incoming JSON-RPC 2.0 requests to corresponding MCP handlers.
        """
        req_id = request_payload.get("id")
        method = request_payload.get("method")
        params = request_payload.get("params") or {}

        if not method or not isinstance(method, str):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: 'method' must be a valid string"
                }
            }

        # 1. tools/list
        if method == "tools/list":
            tool_defs = [t.model_dump() for t in self.registry.list_tools()]
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": tool_defs}
            }

        # 2. tools/call
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments") or {}
            if not tool_name:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: 'name' is required for tools/call"
                    }
                }

            result = await self.registry.execute_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tool": result.tool_name,
                    "content": result.content,
                    "is_error": result.is_error,
                    "error_message": result.error_message,
                    "latency_ms": result.latency_ms
                }
            }

        # 3. ping
        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"status": "pong"}
            }

        # 4. Unknown method
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: '{method}'. Supported methods: tools/list, tools/call, ping"
                }
            }


_mcp_server_instance: Optional[MCPServer] = None

def get_mcp_server() -> MCPServer:
    """Singleton getter for MCPServer."""
    global _mcp_server_instance
    if _mcp_server_instance is None:
        _mcp_server_instance = MCPServer()
    return _mcp_server_instance
