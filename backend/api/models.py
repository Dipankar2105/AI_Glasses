from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    version: str
    environment: str
    timestamp: float
    uptime_seconds: float
    pid: int

class ReadinessResponse(BaseModel):
    status: str # "READY" or "NOT_READY"
    components: Dict[str, str]
    timestamp: float

class CapabilitiesResponse(BaseModel):
    implemented: List[str]
    unavailable: List[str]
    deferred: List[str]
    hardware_status: Dict[str, str]
    ai_models_status: Dict[str, str]

class DetectionBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class DetectionOutput(BaseModel):
    label: str
    confidence: float
    box: DetectionBox

class SceneOutput(BaseModel):
    description: str
    objects: List[str]
    hazards: List[str]
    context: str
    confidence: float

class VisionProcessRequest(BaseModel):
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded JPEG/PNG image")
    width: Optional[int] = Field(default=None, description="Image width")
    height: Optional[int] = Field(default=None, description="Image height")
    seq_num: int = Field(default=1, description="Frame sequence counter")
    use_mock_frame: bool = Field(default=False, description="Use deterministic synthetic frame if true")

class VisionProcessResponse(BaseModel):
    success: bool
    seq_num: int
    timestamp: int
    detections: List[DetectionOutput]
    scene: Optional[SceneOutput]
    ocr_notice: str = "OCR is explicitly DEFERRED in Phase 5"
    errors: List[Dict[str, Any]]
    latency_ms: float
    request_id: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
