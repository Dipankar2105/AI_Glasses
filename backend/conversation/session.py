import time
import threading
from typing import Dict, List, Optional, Any

from backend.conversation.models import ConversationSession, ConversationMessage, MessageRole

class SessionManager:
    """
    Thread-safe in-memory session manager for NextSight conversations.
    Enforces configurable history window bounds to prevent unbounded memory growth.
    """
    def __init__(self, max_history_per_session: int = 20):
        self._sessions: Dict[str, ConversationSession] = {}
        self._lock = threading.RLock()
        self.max_history = max_history_per_session

    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationSession:
        """Retrieves existing session or creates a new one."""
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            
            new_session = ConversationSession(session_id=session_id) if session_id else ConversationSession()
            self._sessions[new_session.session_id] = new_session
            return new_session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieves an existing session by ID, returning None if not found."""
        with self._lock:
            return self._sessions.get(session_id)

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> ConversationMessage:
        """Appends a message to a session and trims oldest non-system messages if limit exceeded."""
        with self._lock:
            session = self.get_or_create_session(session_id)
            msg = ConversationMessage(
                role=role,
                content=content,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                metadata=metadata or {}
            )
            session.messages.append(msg)
            session.updated_at = time.time()

            # Bound history size while preserving system prompt if present at index 0
            if len(session.messages) > self.max_history:
                if session.messages[0].role == MessageRole.SYSTEM:
                    session.messages = [session.messages[0]] + session.messages[-(self.max_history - 1):]
                else:
                    session.messages = session.messages[-self.max_history:]

            return msg

    def update_context(self, session_id: str, context: Dict[str, Any]) -> None:
        """Updates context metadata (e.g. vision objects, location) for a session."""
        with self._lock:
            session = self.get_or_create_session(session_id)
            session.context_metadata.update(context)
            session.updated_at = time.time()

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session from memory. Returns True if deleted, False if not found."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> List[str]:
        """Returns all active session IDs."""
        with self._lock:
            return list(self._sessions.keys())

    def clear(self) -> None:
        """Clears all sessions from memory."""
        with self._lock:
            self._sessions.clear()


_session_manager_instance: Optional[SessionManager] = None

def get_session_manager() -> SessionManager:
    """Singleton getter for SessionManager."""
    global _session_manager_instance
    if _session_manager_instance is None:
        _session_manager_instance = SessionManager()
    return _session_manager_instance

def reset_session_manager(manager: Optional[SessionManager] = None) -> None:
    """Resets or overrides SessionManager singleton."""
    global _session_manager_instance
    _session_manager_instance = manager
