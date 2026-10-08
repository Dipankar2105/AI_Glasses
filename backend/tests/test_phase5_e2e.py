import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.vision.mock_camera import DeterministicMockCamera
from backend.vision.pipeline import VisionPipeline
from backend.ai.registry import AIEngineRegistry
from backend.ai.orchestrator import VisionOrchestrator
from backend.ai.mock_engines import MockObjectDetector, MockOCREngine, MockSceneAnalyzer

def test_phase5_e2e():
    # 1. Setup Phase 3 Mock Hardware
    cam = DeterministicMockCamera(2, 2)
    raw_frame = cam.rgb_frame()
    
    # 2. Setup Phase 4 Pipeline
    pipeline = VisionPipeline()
    vframe = pipeline.process(raw_frame)
    
    # 3. Setup Phase 5 AI
    r = AIEngineRegistry()
    r.register_detector("mock", MockObjectDetector())
    r.register_ocr("mock", MockOCREngine())
    r.register_scene_analyzer("mock", MockSceneAnalyzer())
    orchestrator = VisionOrchestrator(r)
    
    # 4. E2E Execution
    res = orchestrator.process(vframe)
    
    # 5. Validation
    assert len(res.errors) == 0
    assert res.timestamp == raw_frame.timestamp
    assert res.seq_num == raw_frame.seq_num
    assert res.detection_result.detections[0].label == "mock_object"
    assert res.ocr_result.full_text == "mock text"
    assert res.scene_result.description == "A mock scene"
    
    print("test_phase5_e2e PASS")

if __name__ == "__main__":
    test_phase5_e2e()
