from typing import List, Dict, Optional
from backend.vision.frame import OCRInput, DetectionInput, SceneAnalysisInput

def validate_confidence(c: float) -> float:
    if not (0.0 <= c <= 1.0):
        raise ValueError("Confidence must be between 0.0 and 1.0")
    return c

class BoundingBox:
    def __init__(self, x: float, y: float, w: float, h: float):
        # Coordinates should be normalized [0.0, 1.0]
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            raise ValueError("BoundingBox coordinates must be normalized [0.0, 1.0]")
        self.x = x
        self.y = y
        self.width = w
        self.height = h

class Detection:
    def __init__(self, label: str, confidence: float, box: BoundingBox):
        self.label = label
        self.confidence = validate_confidence(confidence)
        self.box = box

class DetectionResult:
    def __init__(self, detections: List[Detection], metadata: dict = None):
        self.detections = detections
        self.metadata = metadata or {}

class OCRTextRegion:
    def __init__(self, text: str, confidence: float, box: BoundingBox, order: int = 0):
        self.text = text
        self.confidence = validate_confidence(confidence)
        self.box = box
        self.order = order

class OCRResult:
    def __init__(self, regions: List[OCRTextRegion], full_text: str, metadata: dict = None):
        self.regions = regions
        self.full_text = full_text
        self.metadata = metadata or {}

class SceneAnalysisResult:
    def __init__(self, description: str, objects: List[str], hazards: List[str], context: str, confidence: float, metadata: dict = None):
        self.description = description
        self.objects = objects
        self.hazards = hazards
        self.context = context
        self.confidence = validate_confidence(confidence)
        self.metadata = metadata or {}

class UnifiedVisionResult:
    def __init__(self):
        self.detection_result: Optional[DetectionResult] = None
        self.ocr_result: Optional[OCRResult] = None
        self.scene_result: Optional[SceneAnalysisResult] = None
        self.timestamp: int = 0
        self.seq_num: int = 0
        self.processing_metadata: dict = {}
        self.errors: List[dict] = []
        
# Engine Interfaces
class ObjectDetectorEngine:
    def process(self, input_data: DetectionInput) -> DetectionResult:
        raise NotImplementedError

class OCREngine:
    def process(self, input_data: OCRInput) -> OCRResult:
        raise NotImplementedError

class SceneAnalyzerEngine:
    def process(self, input_data: SceneAnalysisInput) -> SceneAnalysisResult:
        raise NotImplementedError
