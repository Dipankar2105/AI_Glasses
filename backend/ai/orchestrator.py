import time
from backend.ai.registry import AIEngineRegistry
from backend.ai.errors import AIVisionError, AIVisionErrorStatus
from backend.vision.frame import VisionFrame
from backend.ai.contracts import UnifiedVisionResult
from backend.vision.pipeline import VisionPipeline

class VisionOrchestrator:
    def __init__(self, registry: AIEngineRegistry):
        self.registry = registry
        self.pipeline = VisionPipeline() # Phase 4 connector

    def process(self, vframe: VisionFrame) -> UnifiedVisionResult:
        start_time = time.time()
        result = UnifiedVisionResult()
        
        # Input Validation
        if not vframe or vframe.width <= 0 or vframe.height <= 0 or vframe.data is None or len(vframe.data) == 0:
            result.errors.append({"stage": "orchestrator", "error": "INVALID_INPUT", "msg": "Invalid VisionFrame input"})
            return result
            
        result.timestamp = vframe.timestamp
        result.seq_num = vframe.seq_num

        # Object Detection
        try:
            det_input = self.pipeline.create_detection_input(vframe)
            detector = self.registry.get_detector()
            res = detector.process(det_input)
            # Output validation
            if not isinstance(res.detections, list):
                raise AIVisionError(AIVisionErrorStatus.INVALID_ENGINE_OUTPUT, "Detections must be a list")
            result.detection_result = res
        except AIVisionError as e:
            result.errors.append({"stage": "detector", "error": e.status.name, "msg": str(e)})
        except Exception as e:
            result.errors.append({"stage": "detector", "error": "ENGINE_FAILURE", "msg": str(e)})

        # OCR
        try:
            ocr_input = self.pipeline.create_ocr_input(vframe)
            ocr = self.registry.get_ocr()
            res = ocr.process(ocr_input)
            result.ocr_result = res
        except AIVisionError as e:
            result.errors.append({"stage": "ocr", "error": e.status.name, "msg": str(e)})
        except Exception as e:
            result.errors.append({"stage": "ocr", "error": "ENGINE_FAILURE", "msg": str(e)})

        # Scene Analysis
        try:
            scene_input = self.pipeline.create_scene_analysis_input(vframe)
            scene = self.registry.get_scene_analyzer()
            res = scene.process(scene_input)
            result.scene_result = res
        except AIVisionError as e:
            result.errors.append({"stage": "scene", "error": e.status.name, "msg": str(e)})
        except Exception as e:
            result.errors.append({"stage": "scene", "error": "ENGINE_FAILURE", "msg": str(e)})

        result.processing_metadata["start_time"] = start_time
        result.processing_metadata["end_time"] = time.time()
        result.processing_metadata["elapsed"] = result.processing_metadata["end_time"] - start_time
        
        return result
