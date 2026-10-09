import os
import time
import base64
import numpy as np
import cv2
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse

from backend.config.settings import AppSettings, get_settings
from backend.services.vision_service import VisionService, get_vision_service
from backend.api.models import (
    HealthResponse,
    ReadinessResponse,
    CapabilitiesResponse,
    VisionProcessRequest,
    VisionProcessResponse,
    DetectionOutput,
    DetectionBox,
    SceneOutput
)
from backend.vision.frame import VisionFrame

router = APIRouter()
START_TIME = time.time()

def get_app_settings(request: Request) -> AppSettings:
    """Dependency provider preferring app.state.settings."""
    if hasattr(request.app.state, "settings") and request.app.state.settings is not None:
        return request.app.state.settings
    return get_settings()

def get_app_vision_service(request: Request) -> VisionService:
    """Dependency provider preferring app.state.vision_service."""
    if hasattr(request.app.state, "vision_service") and request.app.state.vision_service is not None:
        return request.app.state.vision_service
    return get_vision_service()

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(settings: AppSettings = Depends(get_app_settings)):
    """Liveness probe returning process status and uptime."""
    now = time.time()
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.version,
        environment=settings.environment,
        timestamp=now,
        uptime_seconds=round(now - START_TIME, 2),
        pid=os.getpid()
    )


@router.get("/ready", response_model=ReadinessResponse, tags=["Health"])
async def readiness_check(service: VisionService = Depends(get_app_vision_service)):
    """Readiness probe evaluating initialization of internal pipelines and engines."""
    components = service.check_readiness()
    # Ready if core pipeline and detector are initialized
    is_ready = (
        components.get("vision_pipeline") == "READY" and
        components.get("object_detector") == "READY" and
        components.get("scene_analyzer") == "READY"
    )
    return ReadinessResponse(
        status="READY" if is_ready else "NOT_READY",
        components=components,
        timestamp=time.time()
    )


@router.get("/api/v1/capabilities", response_model=CapabilitiesResponse, tags=["Capabilities"])
async def get_capabilities(settings: AppSettings = Depends(get_app_settings)):
    """
    Honest declaration of implemented, unavailable, and deferred capabilities.
    Reflects actual software/hardware status without false claims.
    """
    return CapabilitiesResponse(
        implemented=[
            "fastapi_backend_server",
            "request_id_correlation_middleware",
            "centralized_settings_validation",
            "vision_pipeline_color_conversion",
            "vision_pipeline_resize_contrast",
            "ai_engine_registry",
            "vision_orchestrator",
            "async_threadpool_orchestration_with_timeout"
        ],
        unavailable=[
            "physical_camera_sensor",
            "physical_i2s_microphone",
            "physical_imu_sensor",
            "physical_touch_sensor",
            "local_gpu_cuda_acceleration"
        ],
        deferred=[
            "ocr_production_routing (ON_HOLD in Phase 5)",
            "llm_reasoning_and_conversation (Phase 7)",
            "tts_speech_audio_output (Phase 6)",
            "cloud_gemini_multimodal_api (Future Phase)"
        ],
        hardware_status={
            "camera": settings.hardware_camera_status,
            "audio_mic": settings.hardware_audio_status,
            "hal_driver": "MOCKABLE_IN_SOFTWARE"
        },
        ai_models_status={
            "object_detector": "OPERATIONAL (Mock/Deterministic)",
            "scene_analyzer": "OPERATIONAL (Mock/Deterministic)",
            "ocr_pipeline": settings.ocr_status,
            "llm_assistant": settings.llm_reasoning_status
        }
    )


@router.post("/api/v1/vision/process", response_model=VisionProcessResponse, tags=["Vision"])
async def process_vision_frame(
    request: VisionProcessRequest,
    service: VisionService = Depends(get_app_vision_service)
):
    """
    Processes an incoming vision frame through the pipeline and registered AI engines.
    Accepts Base64 image payload or synthetic mock frame request.
    """
    t0 = time.time()
    
    if request.use_mock_frame or not request.image_base64:
        # Create deterministic synthetic test frame
        data = np.full((100, 100, 3), 128, dtype=np.uint8)
        w, h = 100, 100
    else:
        try:
            img_bytes = base64.b64decode(request.image_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Failed to decode image from base64 buffer")
            data = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = data.shape
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed image payload: {str(e)}"
            )

    vframe = VisionFrame(
        data=data,
        width=w,
        height=h,
        channels=3,
        pixel_format="RGB",
        numerical_range=(0, 255),
        timestamp=int(time.time() * 1000),
        seq_num=request.seq_num
    )

    result = await service.process_frame_async(vframe)
    elapsed_ms = (time.time() - t0) * 1000.0

    detections = []
    if result.detection_result and result.detection_result.detections:
        for d in result.detection_result.detections:
            detections.append(DetectionOutput(
                label=d.label,
                confidence=d.confidence,
                box=DetectionBox(
                    x=d.box.x,
                    y=d.box.y,
                    width=d.box.width,
                    height=d.box.height
                )
            ))

    scene_out = None
    if result.scene_result:
        scene_out = SceneOutput(
            description=result.scene_result.description,
            objects=result.scene_result.objects,
            hazards=result.scene_result.hazards,
            context=result.scene_result.context,
            confidence=result.scene_result.confidence
        )

    return VisionProcessResponse(
        success=len(result.errors) == 0,
        seq_num=result.seq_num,
        timestamp=result.timestamp,
        detections=detections,
        scene=scene_out,
        ocr_notice="OCR is explicitly DEFERRED in Phase 5",
        errors=result.errors,
        latency_ms=round(elapsed_ms, 2)
    )
