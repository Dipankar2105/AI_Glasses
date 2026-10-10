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

class AudioTranscribeRequest(BaseModel):
    audio_base64: Optional[str] = Field(default=None, description="Base64 encoded PCM/WAV audio bytes")
    sample_rate: int = Field(default=16000, description="Audio sample rate (Hz), default 16000")
    channels: int = Field(default=1, description="Audio channels (1=mono)")
    reference_audio_base64: Optional[str] = Field(default=None, description="Optional reference audio for AEC")
    run_dsp: bool = Field(default=True, description="Whether to run integrated DSP pipeline before transcription")

class AudioTranscribeResponse(BaseModel):
    success: bool
    transcript: str = ""
    confidence: float = 0.0
    vad_speech_active: bool = False
    dsp_metrics: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    request_id: Optional[str] = None

class SpeechSynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="Text string to synthesize")
    voice: Optional[str] = Field(default=None, description="Optional voice identifier")
    format: Optional[str] = Field(default="pcm", description="Output audio format (pcm, wav)")

class SpeechSynthesizeResponse(BaseModel):
    success: bool
    audio_base64: str = Field(default="", description="Base64 encoded synthesized audio bytes")
    sample_rate: int = 24000
    encoding: str = "pcm_s16le"
    channels: int = 1
    latency_ms: float = 0.0
    error: Optional[str] = None
    request_id: Optional[str] = None

class ProviderStatusInfo(BaseModel):
    provider_type: str
    model: Optional[str] = None
    configured: bool
    available: bool
    status: str
    offline_mode: bool
    api_key_configured: bool

class ProvidersStatusResponse(BaseModel):
    stt: ProviderStatusInfo
    llm: ProviderStatusInfo
    tts: ProviderStatusInfo
    environment: str
    timestamp: float

class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
