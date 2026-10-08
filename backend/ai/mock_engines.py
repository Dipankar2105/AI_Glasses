from backend.ai.contracts import (ObjectDetectorEngine, DetectionResult, Detection, BoundingBox,
                                  OCREngine, OCRResult, OCRTextRegion,
                                  SceneAnalyzerEngine, SceneAnalysisResult)
from backend.vision.frame import DetectionInput, OCRInput, SceneAnalysisInput
from backend.ai.errors import AIVisionError, AIVisionErrorStatus

class MockObjectDetector(ObjectDetectorEngine):
    def process(self, input_data: DetectionInput) -> DetectionResult:
        if input_data.dimensions[0] == 0:
            raise AIVisionError(AIVisionErrorStatus.INVALID_INPUT, "Dimensions are zero")
        return DetectionResult([
            Detection("mock_object", 0.95, BoundingBox(0.1, 0.1, 0.5, 0.5))
        ])

class MockOCREngine(OCREngine):
    def process(self, input_data: OCRInput) -> OCRResult:
        if input_data.dimensions[0] == 0:
            raise AIVisionError(AIVisionErrorStatus.INVALID_INPUT, "Dimensions are zero")
        return OCRResult(
            regions=[OCRTextRegion("mock text", 0.9, BoundingBox(0.2, 0.2, 0.3, 0.1))],
            full_text="mock text"
        )

class MockSceneAnalyzer(SceneAnalyzerEngine):
    def process(self, input_data: SceneAnalysisInput) -> SceneAnalysisResult:
        if input_data.dimensions[0] == 0:
            raise AIVisionError(AIVisionErrorStatus.INVALID_INPUT, "Dimensions are zero")
        return SceneAnalysisResult(
            description="A mock scene",
            objects=["mock_object"],
            hazards=["none"],
            context="indoor",
            confidence=0.85
        )

class FailingMockEngine:
    def process(self, input_data):
        raise AIVisionError(AIVisionErrorStatus.ENGINE_FAILURE, "Simulated engine failure")
