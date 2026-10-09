import time
import uuid
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

class ConversationMessage(BaseModel):
    """A single message within a conversation session."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: float = Field(default_factory=time.time)
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConversationSession(BaseModel):
    """Session container with chronological message history and context metadata."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    messages: List[ConversationMessage] = Field(default_factory=list)
    context_metadata: Dict[str, Any] = Field(default_factory=dict)

# API Request & Response Models
class ConversationMessageRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Existing session ID, or null to generate a new session")
    message: str = Field(..., min_length=1, description="User message text")
    tool_to_invoke: Optional[str] = Field(default=None, description="Optional explicit MCP tool to invoke")
    tool_arguments: Optional[Dict[str, Any]] = Field(default=None, description="Arguments for tool invocation")
    vision_context: Optional[Dict[str, Any]] = Field(default=None, description="Optional vision metadata from current frame")

class ToolExecutionSummary(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    result: Any
    error: Optional[str] = None
    latency_ms: float

class ConversationMessageResponse(BaseModel):
    session_id: str
    message_id: str
    response: str
    llm_status: str # e.g. "PROVIDER_UNAVAILABLE", "MOCK_DEVELOPMENT", "OPERATIONAL"
    tool_executions: List[ToolExecutionSummary] = Field(default_factory=list)
    session_message_count: int
    latency_ms: float
    timestamp: float = Field(default_factory=time.time)

class SessionDetailResponse(BaseModel):
    session_id: str
    created_at: float
    updated_at: float
    message_count: int
    messages: List[ConversationMessage]
    context_metadata: Dict[str, Any]
