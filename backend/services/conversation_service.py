import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

from backend.conversation.models import (
    MessageRole,
    ConversationMessage,
    ConversationSession,
    ConversationMessageRequest,
    ConversationMessageResponse,
    ToolExecutionSummary
)
from backend.conversation.session import SessionManager, get_session_manager
from backend.mcp.registry import MCPToolRegistry, get_tool_registry

class LLMProviderResponse:
    def __init__(
        self,
        content: str,
        status: str = "OPERATIONAL",
        tool_calls: Optional[List[dict]] = None
    ):
        self.content = content
        self.status = status
        self.tool_calls = tool_calls or []

class BaseLLMProvider(ABC):
    """Provider-neutral LLM abstraction for future Gemini/Claude/Local integrations."""
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    async def generate_response(
        self,
        messages: List[ConversationMessage],
        context_metadata: Dict[str, Any],
        available_tools: List[dict]
    ) -> LLMProviderResponse:
        pass


class NullLLMProvider(BaseLLMProvider):
    """
    Default provider in Phase 7.
    Honestly reports that no external or local LLM reasoning model is configured yet.
    """
    def is_available(self) -> bool:
        return False

    async def generate_response(
        self,
        messages: List[ConversationMessage],
        context_metadata: Dict[str, Any],
        available_tools: List[dict]
    ) -> LLMProviderResponse:
        return LLMProviderResponse(
            content="[LLM Unavailable] Conversation orchestration and tool execution are active, but no LLM reasoning provider is currently configured.",
            status="PROVIDER_UNAVAILABLE"
        )


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic test double for unit and integration testing only.
    Clearly marked as TEST_DOUBLE.
    """
    def __init__(self, fixed_response: str = "This is a deterministic mock assistant response for testing."):
        self.fixed_response = fixed_response

    def is_available(self) -> bool:
        return True

    async def generate_response(
        self,
        messages: List[ConversationMessage],
        context_metadata: Dict[str, Any],
        available_tools: List[dict]
    ) -> LLMProviderResponse:
        return LLMProviderResponse(
            content=self.fixed_response,
            status="MOCK_DEVELOPMENT"
        )


class ConversationService:
    """
    Orchestrates contextual conversations for NextSight.
    Coordinates: Session History <-> Context Enrichment <-> MCP Tool Execution <-> LLM Reasoning.
    """
    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        tool_registry: Optional[MCPToolRegistry] = None,
        llm_provider: Optional[BaseLLMProvider] = None
    ):
        self.session_manager = session_manager or get_session_manager()
        self.tool_registry = tool_registry or get_tool_registry()
        self.llm_provider = llm_provider or NullLLMProvider()

    async def process_user_message(
        self,
        request: ConversationMessageRequest
    ) -> ConversationMessageResponse:
        """
        Processes an incoming user message through the orchestration pipeline.
        """
        t0 = time.time()

        # 1. Retrieve or initialize session
        session = self.session_manager.get_or_create_session(request.session_id)
        session_id = session.session_id

        # 2. Update vision context if provided
        if request.vision_context:
            self.session_manager.update_context(session_id, request.vision_context)

        # 3. Record user message in session history
        user_msg = self.session_manager.add_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message,
            metadata={"vision_context_present": request.vision_context is not None}
        )

        # 4. Execute tool if requested or triggered
        tool_summaries: List[ToolExecutionSummary] = []
        if request.tool_to_invoke:
            tool_res = await self.tool_registry.execute_tool(
                name=request.tool_to_invoke,
                arguments=request.tool_arguments or {}
            )
            
            # Record tool result in session history
            tool_content_str = str(tool_res.content) if not tool_res.is_error else tool_res.error_message
            self.session_manager.add_message(
                session_id=session_id,
                role=MessageRole.TOOL_RESULT,
                content=tool_content_str,
                tool_name=request.tool_to_invoke,
                metadata={"is_error": tool_res.is_error}
            )

            tool_summaries.append(ToolExecutionSummary(
                tool_name=request.tool_to_invoke,
                arguments=request.tool_arguments or {},
                success=not tool_res.is_error,
                result=tool_res.content if not tool_res.is_error else None,
                error=tool_res.error_message if tool_res.is_error else None,
                latency_ms=tool_res.latency_ms
            ))

        # 5. Invoke LLM Provider
        available_tools = [t.model_dump() for t in self.tool_registry.list_tools()]
        llm_res = await self.llm_provider.generate_response(
            messages=session.messages,
            context_metadata=session.context_metadata,
            available_tools=available_tools
        )

        # 6. Record assistant response in session history
        assistant_msg = self.session_manager.add_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=llm_res.content,
            metadata={"llm_status": llm_res.status}
        )

        latency_ms = (time.time() - t0) * 1000.0

        return ConversationMessageResponse(
            session_id=session_id,
            message_id=assistant_msg.id,
            response=llm_res.content,
            llm_status=llm_res.status,
            tool_executions=tool_summaries,
            session_message_count=len(session.messages),
            latency_ms=round(latency_ms, 2),
            timestamp=time.time()
        )


_conversation_service_instance: Optional[ConversationService] = None

def get_conversation_service() -> ConversationService:
    """Singleton getter for ConversationService."""
    global _conversation_service_instance
    if _conversation_service_instance is None:
        _conversation_service_instance = ConversationService()
    return _conversation_service_instance

def reset_conversation_service(service: Optional[ConversationService] = None) -> None:
    """Resets or overrides ConversationService singleton."""
    global _conversation_service_instance
    _conversation_service_instance = service
