import time
from typing import Optional, Dict, Any, Callable

from backend.motion.contracts import (
    HeadGestureType,
    TouchGestureType,
    InteractionIntent,
    GestureEvent,
    InteractionResult
)
from backend.services.vision_service import VisionService, get_vision_service
from backend.conversation.session import SessionManager, get_session_manager

class InteractionDispatcher:
    """
    Decoupled interaction dispatcher mapping recognized gestures to safe application intents.
    Features:
      - Clean mapping matrix from gesture events to application intents.
      - Service availability validation before dispatch.
      - Cooldown enforcement per intent type.
      - Honest error reporting without false claims of execution.
    """
    def __init__(
        self,
        vision_service: Optional[VisionService] = None,
        session_manager: Optional[SessionManager] = None,
        intent_cooldown_seconds: float = 0.5
    ):
        self.vision_service = vision_service or get_vision_service()
        self.session_manager = session_manager or get_session_manager()
        self.cooldown_sec = intent_cooldown_seconds
        self._last_intent_times: Dict[InteractionIntent, float] = {}

    def map_gesture_to_intent(self, gesture: GestureEvent) -> InteractionIntent:
        """Determines corresponding InteractionIntent from a recognized gesture."""
        g_type = gesture.gesture_type

        # Touch Mappings
        if g_type == TouchGestureType.TAP.value:
            return InteractionIntent.CONFIRM
        elif g_type == TouchGestureType.DOUBLE_TAP.value:
            return InteractionIntent.TRIGGER_SCENE_ANALYSIS
        elif g_type == TouchGestureType.LONG_PRESS.value:
            return InteractionIntent.DISMISS_OR_CANCEL

        # Head Motion Mappings
        elif g_type == HeadGestureType.NOD.value:
            return InteractionIntent.CONFIRM
        elif g_type == HeadGestureType.SHAKE.value:
            return InteractionIntent.DISMISS_OR_CANCEL
        elif g_type == HeadGestureType.TILT_HEAD.value:
            return InteractionIntent.START_LISTENING

        return InteractionIntent.NONE

    async def dispatch_gesture(
        self,
        gesture: GestureEvent,
        active_session_id: Optional[str] = None
    ) -> InteractionResult:
        """
        Processes a gesture event, maps it to an intent, and safely executes the corresponding handler.
        """
        now = time.time()
        intent = self.map_gesture_to_intent(gesture)

        if intent == InteractionIntent.NONE:
            return InteractionResult(
                intent=InteractionIntent.NONE,
                triggered_by=gesture.gesture_type,
                success=False,
                action_taken="NO_ACTION",
                target_service="NONE",
                error="No intent mapped for gesture",
                timestamp=now
            )

        # Check cooldown for this specific intent
        last_time = self._last_intent_times.get(intent, -999.0)
        if (now - last_time) < self.cooldown_sec:
            return InteractionResult(
                intent=intent,
                triggered_by=gesture.gesture_type,
                success=False,
                action_taken="COOLDOWN_SUPPRESSED",
                target_service="NONE",
                error=f"Intent {intent.value} suppressed by cooldown ({self.cooldown_sec:.1f}s)",
                timestamp=now
            )

        self._last_intent_times[intent] = now

        # Execute intent
        if intent == InteractionIntent.TRIGGER_SCENE_ANALYSIS:
            readiness = self.vision_service.check_readiness() if self.vision_service else {}
            if readiness.get("vision_pipeline") == "READY":
                return InteractionResult(
                    intent=intent,
                    triggered_by=gesture.gesture_type,
                    success=True,
                    action_taken="SCENE_ANALYSIS_DISPATCHED",
                    target_service="VisionService",
                    timestamp=now
                )
            else:
                return InteractionResult(
                    intent=intent,
                    triggered_by=gesture.gesture_type,
                    success=False,
                    action_taken="DISPATCH_ABORTED",
                    target_service="VisionService",
                    error="Vision service is not ready",
                    timestamp=now
                )

        elif intent == InteractionIntent.START_LISTENING:
            session = self.session_manager.get_or_create_session(active_session_id)
            return InteractionResult(
                intent=intent,
                triggered_by=gesture.gesture_type,
                success=True,
                action_taken=f"LISTENING_SESSION_INITIALIZED:{session.session_id}",
                target_service="SessionManager",
                timestamp=now
            )

        elif intent == InteractionIntent.CONFIRM:
            return InteractionResult(
                intent=intent,
                triggered_by=gesture.gesture_type,
                success=True,
                action_taken="CONFIRMATION_SIGNAL_EMITTED",
                target_service="InteractionState",
                timestamp=now
            )

        elif intent == InteractionIntent.DISMISS_OR_CANCEL:
            return InteractionResult(
                intent=intent,
                triggered_by=gesture.gesture_type,
                success=True,
                action_taken="DISMISS_SIGNAL_EMITTED",
                target_service="InteractionState",
                timestamp=now
            )

        return InteractionResult(
            intent=intent,
            triggered_by=gesture.gesture_type,
            success=False,
            action_taken="UNHANDLED_INTENT",
            target_service="NONE",
            error="Intent execution not implemented",
            timestamp=now
        )


_dispatcher_instance: Optional[InteractionDispatcher] = None

def get_interaction_dispatcher() -> InteractionDispatcher:
    """Singleton getter for InteractionDispatcher."""
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = InteractionDispatcher()
    return _dispatcher_instance

def reset_interaction_dispatcher(dispatcher: Optional[InteractionDispatcher] = None) -> None:
    """Resets or overrides InteractionDispatcher singleton."""
    global _dispatcher_instance
    _dispatcher_instance = dispatcher
