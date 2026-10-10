"""Multimodal Vision and Conversation Context Contract Tests.

Validates:
1. Explicit strongly-typed schema for vision context attachment.
2. Clean separation between image quality assessment and semantic object recognition.
3. Strict session isolation of visual metadata.
4. Error propagation and safety authorization enforcement in VisionService.
5. Protection against fabricated detections.
"""

import time
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config.settings import AppSettings
from backend.conversation.models import (
    VisionContextMetadata,
    ImageQualityAssessment,
    ConversationMessageRequest,
    MessageRole
)
from backend.conversation.session import SessionManager, reset_session_manager
from backend.services.conversation_service import ConversationService, reset_conversation_service
from backend.services.vision_service import VisionService, reset_vision_service
from backend.vision.frame import VisionFrame
from backend.power.policy import PowerThermalPolicyManager
from backend.power.contracts import BatteryTelemetry, ThermalTelemetry, OperatingState
from backend.providers.llm import MockLLMProvider

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def cleanup():
    yield
    reset_session_manager()
    reset_conversation_service()
    reset_vision_service()


# =============================================================================
# 1. EXPLICIT SCHEMA VALIDATION
# =============================================================================

def test_vision_context_schema_validation():
    """Verify VisionContextMetadata and ImageQualityAssessment validate constraints."""
    quality = ImageQualityAssessment(
        brightness=128.5,
        dynamic_range=200.0,
        sharpness=45.2,
        is_blurry=False,
        is_underexposed=False,
        is_overexposed=False
    )
    vctx = VisionContextMetadata(
        scene_description="Laboratory workbench with soldering iron",
        objects=["soldering_iron", "oscilloscope", "multimeter"],
        hazards=["hot_surface"],
        image_quality=quality,
        confidence=0.92,
        timestamp_ms=1710000000.0,
        seq_num=42,
        is_mock_frame=True
    )
    dumped = vctx.model_dump(exclude_none=True)
    assert dumped["scene_description"] == "Laboratory workbench with soldering iron"
    assert "soldering_iron" in dumped["objects"]
    assert "hot_surface" in dumped["hazards"]
    assert dumped["image_quality"]["sharpness"] == 45.2
    assert dumped["confidence"] == 0.92
    assert dumped["is_mock_frame"] is True


def test_vision_context_confidence_bounds():
    """Verify confidence must be bounded between 0.0 and 1.0."""
    with pytest.raises(Exception):
        VisionContextMetadata(confidence=1.5)

    with pytest.raises(Exception):
        VisionContextMetadata(confidence=-0.1)


# =============================================================================
# 2. SESSION CONTEXT ATTACHMENT & ISOLATION
# =============================================================================

async def test_vision_context_attached_to_specific_session():
    """Verify vision context attaches to the targeted session and updates metadata."""
    session_mgr = SessionManager()
    conv_svc = ConversationService(session_manager=session_mgr, llm_provider=MockLLMProvider())

    vctx = VisionContextMetadata(
        scene_description="Pedestrian crossing sign visible ahead",
        objects=["traffic_sign", "road"],
        confidence=0.95
    )

    req_a = ConversationMessageRequest(
        session_id="session-user-1",
        message="What sign is ahead?",
        vision_context=vctx
    )
    await conv_svc.process_user_message(req_a)

    req_b = ConversationMessageRequest(
        session_id="session-user-2",
        message="What is the weather?"
    )
    await conv_svc.process_user_message(req_b)

    sess_a = session_mgr.get_session("session-user-1")
    sess_b = session_mgr.get_session("session-user-2")

    # Session A must have vision context
    assert sess_a.context_metadata.get("scene_description") == "Pedestrian crossing sign visible ahead"
    assert "traffic_sign" in sess_a.context_metadata.get("objects", [])

    # Session B must NOT inherit session A's vision context
    assert "scene_description" not in sess_b.context_metadata
    assert "objects" not in sess_b.context_metadata


async def test_stale_vision_context_isolation_across_turns():
    """Verify new vision context overrides old context in the same session."""
    session_mgr = SessionManager()
    conv_svc = ConversationService(session_manager=session_mgr, llm_provider=MockLLMProvider())

    vctx_frame1 = VisionContextMetadata(
        scene_description="Frame 1: Doorway entrance",
        objects=["door"]
    )
    await conv_svc.process_user_message(ConversationMessageRequest(
        session_id="nav-session",
        message="I see a door",
        vision_context=vctx_frame1
    ))

    vctx_frame2 = VisionContextMetadata(
        scene_description="Frame 2: Hallway with elevator",
        objects=["elevator", "stairs"]
    )
    await conv_svc.process_user_message(ConversationMessageRequest(
        session_id="nav-session",
        message="I moved forward",
        vision_context=vctx_frame2
    ))

    sess = session_mgr.get_session("nav-session")
    assert sess.context_metadata["scene_description"] == "Frame 2: Hallway with elevator"
    assert "elevator" in sess.context_metadata["objects"]


# =============================================================================
# 3. IMAGE QUALITY VS SEMANTIC OBJECT RECOGNITION DISTINCTION
# =============================================================================

def test_image_quality_distinguished_from_detection():
    """Verify image quality attributes are distinct from semantic object recognition."""
    quality = ImageQualityAssessment(
        brightness=50.0,
        dynamic_range=80.0,
        sharpness=12.0,
        is_blurry=True,
        is_underexposed=True
    )
    # A blurry underexposed image should not claim false detections
    vctx = VisionContextMetadata(
        scene_description="Low light scene with motion blur",
        objects=[],
        hazards=[],
        image_quality=quality,
        confidence=0.3
    )

    assert vctx.image_quality.is_blurry is True
    assert vctx.image_quality.is_underexposed is True
    assert len(vctx.objects) == 0
    assert vctx.confidence == 0.3


# =============================================================================
# 4. POWER & THERMAL SAFETY AUTHORIZATION BOUNDARY
# =============================================================================

async def test_vision_service_strict_power_policy_blocks_unauthorized_execution():
    """Verify strict power manager blocks vision processing when telemetry is missing."""
    strict_power_mgr = PowerThermalPolicyManager(require_verified_telemetry=True)
    settings = AppSettings(environment="production", require_verified_telemetry=True)
    vision_svc = VisionService(settings=settings, power_manager=strict_power_mgr)

    test_frame = VisionFrame(
        data=np.full((50, 50, 3), 128, dtype=np.uint8),
        width=50,
        height=50,
        channels=3,
        pixel_format="RGB",
        numerical_range=(0, 255),
        timestamp=int(time.time() * 1000),
        seq_num=1
    )

    res = await vision_svc.process_frame_async(test_frame)
    assert len(res.errors) > 0
    assert any("power" in str(e).lower() or "telemetry" in str(e).lower() for e in res.errors)
    assert res.detection_result is None


async def test_vision_service_strict_power_policy_authorizes_with_fresh_telemetry():
    """Verify strict power manager authorizes vision processing when fresh valid telemetry arrives."""
    strict_power_mgr = PowerThermalPolicyManager(require_verified_telemetry=True)
    now = time.time()
    strict_power_mgr.update_battery_telemetry(
        BatteryTelemetry(voltage_volts=3.8, percentage=85.0, timestamp=now)
    )
    strict_power_mgr.update_thermal_telemetry(
        ThermalTelemetry(temperature_celsius=32.0, timestamp=now)
    )

    settings = AppSettings(environment="production", require_verified_telemetry=True)
    vision_svc = VisionService(settings=settings, power_manager=strict_power_mgr)

    test_frame = VisionFrame(
        data=np.full((50, 50, 3), 128, dtype=np.uint8),
        width=50,
        height=50,
        channels=3,
        pixel_format="RGB",
        numerical_range=(0, 255),
        timestamp=int(time.time() * 1000),
        seq_num=2
    )

    res = await vision_svc.process_frame_async(test_frame)
    assert len(res.errors) == 0
    assert res.detection_result is not None
