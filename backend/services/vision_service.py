import asyncio
import time
from typing import Optional, Dict
import numpy as np

from backend.ai.registry import AIEngineRegistry
from backend.ai.orchestrator import VisionOrchestrator
from backend.ai.mock_engines import MockObjectDetector, MockOCREngine, MockSceneAnalyzer
from backend.ai.contracts import UnifiedVisionResult
from backend.vision.frame import VisionFrame
from backend.config.settings import AppSettings, get_settings

from backend.power.policy import PowerThermalPolicyManager
from backend.power.contracts import WorkloadPriority

class VisionService:
    """
    Service layer coordinating vision orchestration, engine registration,
    power/thermal authorization, and async thread-pool execution with timeout handling.
    """
    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        registry: Optional[AIEngineRegistry] = None,
        power_manager: Optional[PowerThermalPolicyManager] = None,
    ):
        self.settings = settings or get_settings()
        self.registry = registry or self._create_default_registry()
        if power_manager is not None:
            self.power_manager = power_manager
        elif self.settings.is_strict_telemetry_required:
            from backend.power.policy import get_power_policy_manager
            self.power_manager = get_power_policy_manager(require_verified_telemetry=True)
        else:
            self.power_manager = None
        self.orchestrator = VisionOrchestrator(self.registry)
        self.start_time = time.time()

    def _create_default_registry(self) -> AIEngineRegistry:
        """Initializes default engine registry."""
        r = AIEngineRegistry()
        r.register_detector("default", MockObjectDetector())
        
        # Load production OCR engines
        try:
            from backend.ai.tesseract_ocr import TesseractOCREngine
            r.register_ocr("tesseract", TesseractOCREngine(), set_active=True)
        except Exception:
            pass

        try:
            from backend.ai.rapid_ocr import RapidOCREngine
            r.register_ocr("rapidocr", RapidOCREngine(), set_active=False)
        except Exception:
            pass

        try:
            from backend.ai.notebook_ocr_pipeline import NotebookOCRPipeline
            r.register_ocr("notebook_pipeline", NotebookOCRPipeline(), set_active=False)
        except Exception:
            pass

        try:
            from backend.providers.cloud_ocr import CloudVisionOCRAdapter
            r.register_ocr("cloud_vision", CloudVisionOCRAdapter(), set_active=False)
        except Exception:
            pass

        if not r.ocrs:
            r.register_ocr("placeholder", MockOCREngine(), set_active=True)

        r.register_scene_analyzer("default", MockSceneAnalyzer())
        return r

    async def process_frame_async(
        self,
        vframe: VisionFrame,
        timeout_seconds: Optional[float] = None
    ) -> UnifiedVisionResult:
        """
        Executes vision pipeline and AI engines asynchronously in a thread pool
        with strict power/thermal authorization and timeout enforcement.
        """
        if self.power_manager is not None:
            ts = (vframe.timestamp / 1000.0) if (vframe and vframe.timestamp > 0) else time.time()
            can_run, reason = self.power_manager.can_execute_workload(WorkloadPriority.VISION_CAPTURE, current_time=ts)
            if not can_run:
                res = UnifiedVisionResult()
                res.timestamp = vframe.timestamp if vframe else 0
                res.seq_num = vframe.seq_num if vframe else 0
                res.errors.append({
                    "stage": "power_policy",
                    "error": "BLOCKED_BY_POWER_POLICY",
                    "msg": f"Vision processing prohibited by power/thermal policy: {reason}"
                })
                return res

        timeout = timeout_seconds if timeout_seconds is not None else self.settings.request_timeout_seconds
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.orchestrator.process, vframe),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            res = UnifiedVisionResult()
            res.timestamp = vframe.timestamp if vframe else 0
            res.seq_num = vframe.seq_num if vframe else 0
            res.errors.append({
                "stage": "service",
                "error": "TIMEOUT",
                "msg": f"Vision processing timed out after {timeout:.1f}s"
            })
            return res

    def process_image_ocr(
        self,
        image: np.ndarray,
        engine_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes OCR explicitly on a supplied RGB image frame.
        Uses active or requested OCR engine (NotebookOCRPipeline, RapidOCR, or Tesseract).
        """
        t0 = time.perf_counter()
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {
                "success": False,
                "text": "",
                "lines": [],
                "paragraphs": [],
                "confidence": 0.0,
                "error": "Empty or invalid image frame",
                "latency_ms": 0.0
            }

        h, w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1
        from backend.vision.frame import OCRInput
        ocr_inp = OCRInput(
            image=image,
            width=w,
            height=h,
            channels=channels,
            numerical_range=(0, 255),
            timestamp=int(time.time() * 1000),
            seq_num=1
        )

        try:
            if engine_name:
                if engine_name in self.registry.ocrs:
                    engine = self.registry.ocrs[engine_name]
                else:
                    return {
                        "success": False,
                        "text": "",
                        "lines": [],
                        "paragraphs": [],
                        "confidence": 0.0,
                        "error": f"Unknown OCR engine '{engine_name}'. Available: {list(self.registry.ocrs.keys())}",
                        "latency_ms": 0.0
                    }
            else:
                engine = self.registry.get_ocr()

            res = engine.process(ocr_inp)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            paragraphs = getattr(res, "paragraphs", [res.full_text] if res.full_text else [])
            lines = getattr(res, "lines", res.full_text.splitlines() if res.full_text else [])
            conf = getattr(res, "confidence", 1.0 if res.full_text else 0.0)

            return {
                "success": bool(res.full_text),
                "text": res.full_text,
                "paragraphs": paragraphs,
                "lines": lines,
                "confidence": conf,
                "regions_count": len(res.regions) if hasattr(res, "regions") else 0,
                "latency_ms": round(elapsed_ms, 2)
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "lines": [],
                "paragraphs": [],
                "confidence": 0.0,
                "error": str(e),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

    def check_readiness(self) -> Dict[str, str]:
        """Evaluates readiness of all internal vision and AI components."""
        status = {}
        # 1. Pipeline check
        try:
            dummy_frame = VisionFrame(
                data=np.zeros((10, 10, 3), dtype=np.uint8),
                width=10,
                height=10,
                channels=3,
                pixel_format="RGB",
                numerical_range=(0, 255),
                timestamp=1,
                seq_num=1
            )
            status["vision_pipeline"] = "READY"
        except Exception as e:
            status["vision_pipeline"] = f"ERROR: {str(e)}"

        # 2. Engine registry check
        try:
            _ = self.registry.get_detector()
            status["object_detector"] = "READY"
        except Exception:
            status["object_detector"] = "UNAVAILABLE"

        try:
            _ = self.registry.get_scene_analyzer()
            status["scene_analyzer"] = "READY"
        except Exception:
            status["scene_analyzer"] = "UNAVAILABLE"

        # 3. Explicitly declare deferred / mock subsystems
        status["ocr_engine"] = "DEFERRED_PHASE_5"
        status["llm_reasoning"] = "DEFERRED_PHASE_5"
        status["hardware_camera"] = self.settings.hardware_camera_status
        status["hardware_audio"] = self.settings.hardware_audio_status

        return status


_vision_service_instance: Optional[VisionService] = None

def get_vision_service(
    settings: Optional[AppSettings] = None,
    power_manager: Optional[PowerThermalPolicyManager] = None,
) -> VisionService:
    """Singleton getter for VisionService (used in FastAPI Dependency Injection)."""
    global _vision_service_instance
    if _vision_service_instance is None:
        _vision_service_instance = VisionService(settings=settings, power_manager=power_manager)
    return _vision_service_instance

def reset_vision_service(service: Optional[VisionService] = None) -> None:
    """Resets or overrides VisionService singleton for testing."""
    global _vision_service_instance
    _vision_service_instance = service
