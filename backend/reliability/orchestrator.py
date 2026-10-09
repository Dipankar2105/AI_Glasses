"""End-to-end integration orchestrator connecting device simulation, protocol framing, motion/touch processing, power policies, conversation, MCP tools, and vision services."""

import time
from typing import Dict, Any, Optional
from backend.protocol.contracts import PacketType
from backend.protocol.framing import ProtocolFraming
from backend.protocol.device_sim import SimulatedESP32Device
from backend.motion.contracts import HeadGestureType, TouchGestureType, InteractionIntent
from backend.motion.dispatcher import InteractionDispatcher
from backend.motion.processor import MotionProcessor
from backend.motion.touch import TouchProcessor
from backend.power.policy import PowerThermalPolicyManager
from backend.power.contracts import WorkloadPriority, OperatingState
from backend.services.conversation_service import ConversationService
from backend.services.vision_service import VisionService
from backend.mcp.server import MCPServer
from backend.mcp.registry import MCPToolRegistry
from backend.reliability.circuit_breaker import CircuitBreaker


class SystemIntegrationOrchestrator:
    """Coordinates end-to-end data flow between glasses device and backend subsystems."""

    def __init__(
        self,
        conversation_service: Optional[ConversationService] = None,
        vision_service: Optional[VisionService] = None,
        mcp_server: Optional[MCPServer] = None,
        power_manager: Optional[PowerThermalPolicyManager] = None,
    ) -> None:
        self.conv_service = conversation_service or ConversationService()
        self.vision_service = vision_service or VisionService()
        self.mcp_server = mcp_server or MCPServer()
        self.power_manager = power_manager or PowerThermalPolicyManager()

        self.motion_processor = MotionProcessor()
        self.touch_processor = TouchProcessor()
        self.dispatcher = InteractionDispatcher(
            vision_service=self.vision_service,
            session_manager=self.conv_service.session_manager if hasattr(self.conv_service, "session_manager") else None,
        )

        self.vision_circuit = CircuitBreaker(name="vision_service", failure_threshold=3, recovery_timeout_s=5.0)

    async def process_incoming_device_packet_async(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Async version of packet processing."""
        return self._process_packet_internal(raw_bytes, is_async=True)

    def process_incoming_device_packet(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Process an incoming framed packet from the device through the entire backend pipeline."""
        return self._process_packet_internal(raw_bytes, is_async=False)

    def _process_packet_internal(self, raw_bytes: bytes, is_async: bool = False) -> Dict[str, Any]:
        decoded = ProtocolFraming.decode_message(raw_bytes)
        if not decoded.is_valid:
            return {
                "status": "REJECTED",
                "error": decoded.error_detail,
                "packet_type": "INVALID",
            }

        p_type = decoded.header.packet_type

        # 1. Telemetry Packet -> Update Power / Thermal policies
        if p_type == PacketType.TELEMETRY:
            import json
            try:
                telem_dict = json.loads(decoded.payload.decode("utf-8"))
                from backend.power.contracts import BatteryTelemetry, ThermalTelemetry
                bat_pct = float(telem_dict.get("battery_pct", 100.0))
                temp_c = float(telem_dict.get("temp_c", 30.0))
                ts = decoded.header.timestamp_ms / 1000.0

                bat_status = self.power_manager.update_battery_telemetry(
                    BatteryTelemetry(voltage_volts=3.3 + (bat_pct / 100.0) * 0.9, percentage=bat_pct, timestamp=ts),
                    current_time=ts,
                )
                therm_status = self.power_manager.update_thermal_telemetry(
                    ThermalTelemetry(temperature_celsius=temp_c, timestamp=ts),
                    current_time=ts,
                )
                return {
                    "status": "PROCESSED",
                    "packet_type": "TELEMETRY",
                    "battery_status": bat_status.value,
                    "thermal_status": therm_status.value,
                    "operating_state": self.power_manager.requested_state.value,
                }
            except Exception as e:
                return {"status": "ERROR", "error": f"Failed to parse telemetry: {e}"}

        # 2. Motion / Touch Gestures -> Interaction Dispatcher
        if p_type in (PacketType.MOTION_EVENT, PacketType.TOUCH_EVENT):
            import json
            try:
                event_dict = json.loads(decoded.payload.decode("utf-8"))
                from backend.motion.contracts import GestureEvent
                g_name = event_dict.get("gesture", "")
                ts = decoded.header.timestamp_ms / 1000.0

                event = GestureEvent(
                    gesture_type=g_name,
                    confidence=float(event_dict.get("confidence", 1.0)),
                    start_time=ts,
                    end_time=ts,
                )

                # Check power permission for vision triggering
                if g_name == "DOUBLE_TAP":
                    can_vision, reason = self.power_manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE)
                    if not can_vision:
                        return {
                            "status": "BLOCKED_BY_POWER_POLICY",
                            "reason": reason,
                            "gesture": g_name,
                        }

                import asyncio
                import inspect
                if is_async:
                    # coroutine dispatch
                    dispatch_result = asyncio.run(self.dispatcher.dispatch_gesture(event))
                else:
                    # sync runner for dispatch_gesture
                    import inspect
                    if inspect.iscoroutinefunction(self.dispatcher.dispatch_gesture):
                        dispatch_result = asyncio.run(self.dispatcher.dispatch_gesture(event))
                    else:
                        dispatch_result = self.dispatcher.dispatch_gesture(event)

                return {
                    "status": "DISPATCHED",
                    "intent": dispatch_result.intent.value,
                    "success": dispatch_result.success,
                    "action_taken": dispatch_result.action_taken,
                    "target_service": dispatch_result.target_service,
                    "error": dispatch_result.error,
                }
            except Exception as e:
                return {"status": "ERROR", "error": f"Failed to process gesture event: {e}"}

        # 3. Audio / Heartbeat
        return {
            "status": "ACK",
            "packet_type": p_type.name,
            "seq": decoded.header.sequence_number,
        }
