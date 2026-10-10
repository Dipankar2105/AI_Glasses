import time
import uuid
from enum import Enum
from typing import List, Dict, Optional, Any, Union
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

class ImageQualityAssessment(BaseModel):
    """Host image quality assessment metrics (distinguished from semantic object recognition)."""
    brightness: Optional[float] = None
    dynamic_range: Optional[float] = None
    sharpness: Optional[float] = None
    is_blurry: Optional[bool] = None
    is_underexposed: Optional[bool] = None
    is_overexposed: Optional[bool] = None

class VisionContextMetadata(BaseModel):
    """Explicit strongly-typed schema for multimodal vision context."""
    model_config = {"extra": "allow"}

    scene_description: Optional[str] = Field(default=None, description="Semantic summary of the visual scene")
    objects: List[str] = Field(default_factory=list, description="Detected object labels")
    hazards: List[str] = Field(default_factory=list, description="Detected environmental hazards")
    image_quality: Optional[ImageQualityAssessment] = Field(default=None, description="Image quality metrics")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    timestamp_ms: Optional[float] = Field(default=None, description="Capture timestamp in ms")
    seq_num: Optional[int] = Field(default=None, description="Frame sequence number")
    is_mock_frame: bool = Field(default=False, description="Whether context originated from synthetic test frame")

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
    vision_context: Optional[Union[VisionContextMetadata, Dict[str, Any]]] = Field(
        default=None, description="Optional strongly-typed or dictionary vision metadata from current frame"
    )

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
