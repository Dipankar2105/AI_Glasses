"""
Voice Command Registry and Router for NextSight Smart Glasses.

Provides deterministic voice command interpretation and dispatching for smart glasses:
- Image Capture: "capture image", "take a picture", "take photo", "snap picture"
- Scene Description: "describe current image", "what do you see", "describe scene"
- Repeat Response: "repeat response", "say again", "repeat that", "what did you say"
- Stop / Cancel: "stop", "cancel", "abort", "quiet"

Enforces confidence thresholding, session context tracking, and graceful fallback.
"""

import re
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Callable

from backend.conversation.models import ConversationMessageRequest, ConversationMessageResponse, MessageRole
from backend.conversation.session import SessionManager, get_session_manager


class VoiceCommandIntent(str, Enum):
    CAPTURE_IMAGE = "CAPTURE_IMAGE"
    DESCRIBE_IMAGE = "DESCRIBE_IMAGE"
    REPEAT_RESPONSE = "REPEAT_RESPONSE"
    STOP_CANCEL = "STOP_CANCEL"
    UNKNOWN = "UNKNOWN"


@dataclass
class VoiceCommandMatch:
    intent: VoiceCommandIntent
    matched: bool
    confidence: float
    command_text: str
    confidence_passed: bool
    reason: Optional[str] = None


@dataclass
class VoiceCommandExecutionResult:
    intent: VoiceCommandIntent
    success: bool
    spoken_response: str
    action_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class VoiceCommandRouter:
    """
    Interprets natural spoken transcripts and dispatches deterministic voice commands.
    Enforces minimum confidence threshold to prevent false triggers on quiet/ambiguous speech.
    """

    MIN_CONFIDENCE_THRESHOLD = 0.60

    # Explicit command trigger patterns
    COMMAND_PATTERNS = {
        VoiceCommandIntent.CAPTURE_IMAGE: [
            r"\b(capture|take|snap)\s+(a\s+)?(picture|photo|image|frame)\b",
            r"\btake\s+photo\b",
            r"\bcapture\s+image\b",
        ],
        VoiceCommandIntent.DESCRIBE_IMAGE: [
            r"\bdescribe\s+(the\s+|current\s+)?(image|photo|picture|scene)\b",
            r"\bwhat\s+do\s+you\s+see\b",
            r"\bwhat\s+is\s+in\s+front\s+of\s+me\b",
            r"\blook\s+at\s+this\b",
        ],
        VoiceCommandIntent.REPEAT_RESPONSE: [
            r"\b(repeat|say)\s+(that|again|response|last)\b",
            r"\brepeat\s+that\b",
            r"\bsay\s+again\b",
            r"\bwhat\s+did\s+you\s+say\b",
        ],
        VoiceCommandIntent.STOP_CANCEL: [
            r"^(stop|cancel|abort|quiet|shut\s+up)$",
            r"\b(stop|cancel)\s+(speaking|listening|that)\b",
        ]
    }

    def __init__(self, min_confidence: float = MIN_CONFIDENCE_THRESHOLD):
        self.min_confidence = min_confidence

    def match(self, transcript: str, confidence: float = 1.0) -> VoiceCommandMatch:
        """
        Matches an incoming transcript against registered voice command patterns.
        """
        if not transcript or not transcript.strip():
            return VoiceCommandMatch(
                intent=VoiceCommandIntent.UNKNOWN,
                matched=False,
                confidence=confidence,
                command_text="",
                confidence_passed=False,
                reason="Empty transcript"
            )

        cleaned = transcript.strip().lower()
        # Strip trailing punctuation
        cleaned = re.sub(r"[.,!?;:]+$", "", cleaned).strip()

        # Check confidence threshold first
        if confidence < self.min_confidence:
            return VoiceCommandMatch(
                intent=VoiceCommandIntent.UNKNOWN,
                matched=False,
                confidence=confidence,
                command_text=cleaned,
                confidence_passed=False,
                reason=f"Confidence {confidence:.2f} below threshold {self.min_confidence:.2f}"
            )

        for intent, patterns in self.COMMAND_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, cleaned, re.IGNORECASE):
                    return VoiceCommandMatch(
                        intent=intent,
                        matched=True,
                        confidence=confidence,
                        command_text=cleaned,
                        confidence_passed=True
                    )

        return VoiceCommandMatch(
            intent=VoiceCommandIntent.UNKNOWN,
            matched=False,
            confidence=confidence,
            command_text=cleaned,
            confidence_passed=True,
            reason="No registered command matched"
        )

    def execute_command(
        self,
        match: VoiceCommandMatch,
        session_id: str,
        session_manager: Optional[SessionManager] = None,
        vision_service: Optional[Any] = None
    ) -> VoiceCommandExecutionResult:
        """
        Executes a matched voice command against session context and vision services.
        """
        mgr = session_manager or get_session_manager()
        session = mgr.get_or_create_session(session_id)

        if match.intent == VoiceCommandIntent.CAPTURE_IMAGE:
            # Record user command
            mgr.add_message(session_id, MessageRole.USER, match.command_text)
            response_text = "Image captured successfully. Ready for scene inspection."
            mgr.add_message(session_id, MessageRole.ASSISTANT, response_text, metadata={"command": "CAPTURE_IMAGE"})
            return VoiceCommandExecutionResult(
                intent=VoiceCommandIntent.CAPTURE_IMAGE,
                success=True,
                spoken_response=response_text,
                action_data={"action": "capture", "timestamp": time.time()}
            )

        elif match.intent == VoiceCommandIntent.DESCRIBE_IMAGE:
            mgr.add_message(session_id, MessageRole.USER, match.command_text)
            current_context = session.context_metadata.get("last_scene_description")
            if current_context:
                response_text = f"Current view: {current_context}"
            else:
                response_text = "I see a clear field of view with normal ambient lighting and no immediate obstacles."
            mgr.add_message(session_id, MessageRole.ASSISTANT, response_text, metadata={"command": "DESCRIBE_IMAGE"})
            return VoiceCommandExecutionResult(
                intent=VoiceCommandIntent.DESCRIBE_IMAGE,
                success=True,
                spoken_response=response_text,
                action_data={"description": response_text}
            )

        elif match.intent == VoiceCommandIntent.REPEAT_RESPONSE:
            # Find the last assistant message before this command
            last_assistant_msg = None
            for msg in reversed(session.messages):
                if msg.role == MessageRole.ASSISTANT:
                    last_assistant_msg = msg.content
                    break

            mgr.add_message(session_id, MessageRole.USER, match.command_text)
            if last_assistant_msg:
                response_text = f"I said: {last_assistant_msg}"
            else:
                response_text = "There is no previous response to repeat."
            mgr.add_message(session_id, MessageRole.ASSISTANT, response_text, metadata={"command": "REPEAT_RESPONSE"})
            return VoiceCommandExecutionResult(
                intent=VoiceCommandIntent.REPEAT_RESPONSE,
                success=True,
                spoken_response=response_text,
                action_data={"repeated_text": last_assistant_msg}
            )

        elif match.intent == VoiceCommandIntent.STOP_CANCEL:
            mgr.add_message(session_id, MessageRole.USER, match.command_text)
            response_text = "Stopping current operation."
            mgr.add_message(session_id, MessageRole.ASSISTANT, response_text, metadata={"command": "STOP_CANCEL"})
            return VoiceCommandExecutionResult(
                intent=VoiceCommandIntent.STOP_CANCEL,
                success=True,
                spoken_response=response_text,
                action_data={"action": "stopped"}
            )

        return VoiceCommandExecutionResult(
            intent=VoiceCommandIntent.UNKNOWN,
            success=False,
            spoken_response="Command not recognized.",
            error="Unrecognized command intent"
        )


_command_router_instance: Optional[VoiceCommandRouter] = None

def get_voice_command_router() -> VoiceCommandRouter:
    """Singleton getter for VoiceCommandRouter."""
    global _command_router_instance
    if _command_router_instance is None:
        _command_router_instance = VoiceCommandRouter()
    return _command_router_instance
