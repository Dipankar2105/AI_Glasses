from typing import Dict, Optional
from backend.ai.contracts import ObjectDetectorEngine, OCREngine, SceneAnalyzerEngine
from backend.ai.errors import AIVisionError, AIVisionErrorStatus

class AIEngineRegistry:
    def __init__(self):
        self.detectors: Dict[str, ObjectDetectorEngine] = {}
        self.ocrs: Dict[str, OCREngine] = {}
        self.scenes: Dict[str, SceneAnalyzerEngine] = {}
        
        self.active_detector: Optional[str] = None
        self.active_ocr: Optional[str] = None
        self.active_scene: Optional[str] = None

    def register_detector(self, name: str, engine: ObjectDetectorEngine, set_active: bool = True):
        self.detectors[name] = engine
        if set_active:
            self.active_detector = name

    def register_ocr(self, name: str, engine: OCREngine, set_active: bool = True):
        self.ocrs[name] = engine
        if set_active:
            self.active_ocr = name

    def register_scene_analyzer(self, name: str, engine: SceneAnalyzerEngine, set_active: bool = True):
        self.scenes[name] = engine
        if set_active:
            self.active_scene = name

    def get_detector(self) -> ObjectDetectorEngine:
        if not self.active_detector or self.active_detector not in self.detectors:
            raise AIVisionError(AIVisionErrorStatus.ENGINE_UNAVAILABLE, "No active ObjectDetectorEngine")
        return self.detectors[self.active_detector]

    def get_ocr(self) -> OCREngine:
        if not self.active_ocr or self.active_ocr not in self.ocrs:
            raise AIVisionError(AIVisionErrorStatus.ENGINE_UNAVAILABLE, "No active OCREngine")
        return self.ocrs[self.active_ocr]

    def get_scene_analyzer(self) -> SceneAnalyzerEngine:
        if not self.active_scene or self.active_scene not in self.scenes:
            raise AIVisionError(AIVisionErrorStatus.ENGINE_UNAVAILABLE, "No active SceneAnalyzerEngine")
        return self.scenes[self.active_scene]

    def load_production_ocr(self, set_active: bool = True):
        from backend.ai.tesseract_ocr import TesseractOCREngine
        self.register_ocr("tesseract", TesseractOCREngine(), set_active=set_active)
