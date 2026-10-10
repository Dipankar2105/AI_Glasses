"""Conversation orchestration service integrating Session History, MCP Tools, and AI Providers."""

import time
from typing import List, Dict, Optional, Any, Union

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
from backend.providers.contracts import ProviderStatus, LLMResult, LLMToolCall
from backend.providers.llm import BaseLLMProvider, NullLLMProvider, MockLLMProvider, LLMProviderResponse
from backend.providers.factory import get_llm_provider


class ConversationService:
    """
    Orchestrates contextual conversations for NextSight.
    Coordinates: Session History <-> Context Enrichment <-> MCP Tool Execution <-> LLM Reasoning.
    Enforces bounded tool recursion loops (max 3 iterations) and timeout protection.
    """
    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        tool_registry: Optional[MCPToolRegistry] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        max_tool_iterations: int = 3
    ):
        self.session_manager = session_manager or get_session_manager()
        self.tool_registry = tool_registry or get_tool_registry()
        self.llm_provider = llm_provider or get_llm_provider()
        self.max_tool_iterations = max_tool_iterations

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

        # 4. Execute explicit tool if requested by client
        tool_summaries: List[ToolExecutionSummary] = []
        if request.tool_to_invoke:
            tool_res = await self.tool_registry.execute_tool(
                name=request.tool_to_invoke,
                arguments=request.tool_arguments or {}
            )
            
            # Record tool result in session history
            tool_content_str = str(tool_res.content) if not tool_res.is_error else str(tool_res.error_message)
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

        # 5. Invoke LLM Provider with tool recursion guard
        available_tools = [t.model_dump() for t in self.tool_registry.list_tools()]
        iteration = 0
        final_content = ""
        final_status = "PROVIDER_UNAVAILABLE"

        while iteration < self.max_tool_iterations:
            iteration += 1
            llm_res = await self.llm_provider.generate_response(
                messages=session.messages,
                context_metadata=session.context_metadata,
                available_tools=available_tools
            )

            # Handle both legacy LLMProviderResponse and modern LLMResult
            if isinstance(llm_res, LLMResult):
                final_content = llm_res.content
                final_status = llm_res.status.value if hasattr(llm_res.status, "value") else str(llm_res.status)
                tool_calls = llm_res.tool_calls
            else:
                final_content = llm_res.content
                final_status = llm_res.status
                tool_calls = getattr(llm_res, "tool_calls", [])

            # Check if LLM requested additional tool calls and we have remaining iterations
            if tool_calls and iteration < self.max_tool_iterations:
                for tc in tool_calls:
                    tc_name = tc.tool_name if hasattr(tc, "tool_name") else tc.get("name")
                    tc_args = tc.arguments if hasattr(tc, "arguments") else tc.get("arguments", {})
                    if tc_name:
                        t_res = await self.tool_registry.execute_tool(name=tc_name, arguments=tc_args)
                        t_content_str = str(t_res.content) if not t_res.is_error else str(t_res.error_message)
                        self.session_manager.add_message(
                            session_id=session_id,
                            role=MessageRole.TOOL_RESULT,
                            content=t_content_str,
                            tool_name=tc_name,
                            metadata={"is_error": t_res.is_error}
                        )
                        tool_summaries.append(ToolExecutionSummary(
                            tool_name=tc_name,
                            arguments=tc_args,
                            success=not t_res.is_error,
                            result=t_res.content if not t_res.is_error else None,
                            error=t_res.error_message if t_res.is_error else None,
                            latency_ms=t_res.latency_ms
                        ))
                continue
            else:
                break

        # 6. Record assistant response in session history
        assistant_msg = self.session_manager.add_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=final_content,
            metadata={"llm_status": final_status}
        )

        latency_ms = (time.time() - t0) * 1000.0

        return ConversationMessageResponse(
            session_id=session_id,
            message_id=assistant_msg.id,
            response=final_content,
            llm_status=final_status,
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
