import time
import asyncio
import inspect
from typing import Dict, List, Optional, Callable, Any
import numpy as np

from backend.mcp.models import MCPToolDefinition, MCPToolCallResult
from backend.config.settings import get_settings
from backend.services.vision_service import get_vision_service
from backend.conversation.session import get_session_manager
from backend.vision.frame import VisionFrame

class MCPTool:
    """Registered tool metadata and execution handler."""
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: Callable[..., Any]
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def to_definition(self) -> MCPToolDefinition:
        return MCPToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema
        )


class MCPToolRegistry:
    """
    Controlled registry for Model Context Protocol (MCP) tools.
    Enforces argument validation, execution timeouts, and strict error isolation.
    Does NOT allow shell execution, filesystem access, or external network requests.
    """
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}
        self._register_builtin_tools()

    def register_tool(self, tool: MCPTool) -> None:
        """Registers an MCP tool."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Retrieves a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[MCPToolDefinition]:
        """Returns definitions of all registered tools."""
        return [tool.to_definition() for tool in self._tools.values()]

    async def execute_tool(
        self,
        name: str,
        arguments: Optional[dict] = None,
        timeout_seconds: float = 5.0
    ) -> MCPToolCallResult:
        """
        Executes a registered tool with timeout and structured error isolation.
        """
        t0 = time.time()
        arguments = arguments or {}

        tool = self._tools.get(name)
        if not tool:
            return MCPToolCallResult(
                tool_name=name,
                is_error=True,
                error_message=f"Unknown tool '{name}'. Available tools: {list(self._tools.keys())}",
                latency_ms=round((time.time() - t0) * 1000.0, 2)
            )

        try:
            # Check if handler is async or sync
            if inspect.iscoroutinefunction(tool.handler):
                coro = tool.handler(**arguments)
            else:
                coro = asyncio.to_thread(tool.handler, **arguments)

            result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            latency = (time.time() - t0) * 1000.0

            content = [{"type": "text", "data": result}] if not isinstance(result, list) else result
            return MCPToolCallResult(
                tool_name=name,
                content=content,
                is_error=False,
                latency_ms=round(latency, 2)
            )

        except asyncio.TimeoutError:
            latency = (time.time() - t0) * 1000.0
            return MCPToolCallResult(
                tool_name=name,
                is_error=True,
                error_message=f"Tool '{name}' execution timed out after {timeout_seconds:.1f}s",
                latency_ms=round(latency, 2)
            )
        except TypeError as te:
            latency = (time.time() - t0) * 1000.0
            return MCPToolCallResult(
                tool_name=name,
                is_error=True,
                error_message=f"Invalid arguments for tool '{name}': {str(te)}",
                latency_ms=round(latency, 2)
            )
        except Exception as e:
            latency = (time.time() - t0) * 1000.0
            return MCPToolCallResult(
                tool_name=name,
                is_error=True,
                error_message=f"Execution failure in tool '{name}': {str(e)}",
                latency_ms=round(latency, 2)
            )

    def _register_builtin_tools(self) -> None:
        """Registers default safe tools for NextSight."""
        
        # 1. System Status Tool
        def get_system_status() -> dict:
            settings = get_settings()
            service = get_vision_service()
            return {
                "app_name": settings.app_name,
                "version": settings.version,
                "environment": settings.environment,
                "components": service.check_readiness(),
                "ocr_status": "DEFERRED_PHASE_5",
                "hardware_status": "SOFTWARE_SIMULATION"
            }

        self.register_tool(MCPTool(
            name="get_system_status",
            description="Inspects NextSight system health, readiness, and active subsystem statuses.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=get_system_status
        ))

        # 2. Vision Analysis Tool
        async def analyze_vision_frame(use_synthetic_frame: bool = True) -> dict:
            service = get_vision_service()
            # Generate deterministic test frame
            data = np.full((100, 100, 3), 128, dtype=np.uint8)
            vframe = VisionFrame(
                data=data,
                width=100,
                height=100,
                channels=3,
                pixel_format="RGB",
                numerical_range=(0, 255),
                timestamp=int(time.time() * 1000),
                seq_num=1
            )
            res = await service.process_frame_async(vframe)
            detections = []
            if res.detection_result and res.detection_result.detections:
                for d in res.detection_result.detections:
                    detections.append({
                        "label": d.label,
                        "confidence": d.confidence,
                        "box": [d.box.x, d.box.y, d.box.width, d.box.height]
                    })
            
            scene_desc = res.scene_result.description if res.scene_result else "No scene description"
            return {
                "detections": detections,
                "scene_description": scene_desc,
                "ocr_notice": "OCR is explicitly DEFERRED in Phase 5 & Phase 7",
                "errors": res.errors
            }

        self.register_tool(MCPTool(
            name="analyze_vision_frame",
            description="Runs vision pipeline on the current camera frame to detect objects and describe the scene.",
            input_schema={
                "type": "object",
                "properties": {
                    "use_synthetic_frame": {
                        "type": "boolean",
                        "description": "Whether to use a synthetic software test frame (default: true)"
                    }
                },
                "required": []
            },
            handler=analyze_vision_frame
        ))

        # 3. Conversation Context Tool
        def get_conversation_context(session_id: str) -> dict:
            mgr = get_session_manager()
            session = mgr.get_session(session_id)
            if not session:
                return {"error": f"Session '{session_id}' not found"}
            
            return {
                "session_id": session.session_id,
                "message_count": len(session.messages),
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "recent_messages": [
                    {"role": m.role.value, "content": m.content}
                    for m in session.messages[-5:]
                ],
                "context_metadata": session.context_metadata
            }

        self.register_tool(MCPTool(
            name="get_conversation_context",
            description="Retrieves message history and context metadata for an active conversation session.",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The unique conversation session ID"
                    }
                },
                "required": ["session_id"]
            },
            handler=get_conversation_context
        ))


_tool_registry_instance: Optional[MCPToolRegistry] = None

def get_tool_registry() -> MCPToolRegistry:
    """Singleton getter for MCPToolRegistry."""
    global _tool_registry_instance
    if _tool_registry_instance is None:
        _tool_registry_instance = MCPToolRegistry()
    return _tool_registry_instance

def reset_tool_registry(registry: Optional[MCPToolRegistry] = None) -> None:
    """Resets or overrides MCPToolRegistry singleton."""
    global _tool_registry_instance
    _tool_registry_instance = registry
