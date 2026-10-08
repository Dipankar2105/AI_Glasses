import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.ai.registry import AIEngineRegistry
from backend.ai.orchestrator import VisionOrchestrator
from backend.ai.mock_engines import MockObjectDetector, MockOCREngine, MockSceneAnalyzer, FailingMockEngine
from backend.vision.frame import VisionFrame

def test_orchestrator():
    r = AIEngineRegistry()
    r.register_detector("mock", MockObjectDetector())
    r.register_ocr("mock", MockOCREngine())
    r.register_scene_analyzer("mock", MockSceneAnalyzer())
    
    o = VisionOrchestrator(r)
    vframe = VisionFrame([0], 1, 1, 1, "GRAY", (0,1), 1000, 1)
    
    res = o.process(vframe)
    assert len(res.errors) == 0
    assert res.detection_result is not None
    assert res.ocr_result is not None
    assert res.scene_result is not None

def test_orchestrator_failure_isolation():
    r = AIEngineRegistry()
    r.register_detector("mock", MockObjectDetector())
    r.register_ocr("mock", FailingMockEngine())
    r.register_scene_analyzer("mock", MockSceneAnalyzer())
    
    o = VisionOrchestrator(r)
    vframe = VisionFrame([0], 1, 1, 1, "GRAY", (0,1), 1000, 1)
    
    res = o.process(vframe)
    assert len(res.errors) == 1
    assert res.errors[0]["stage"] == "ocr"
    assert res.errors[0]["error"] == "ENGINE_FAILURE"
    assert res.ocr_result is None
    assert res.detection_result is not None
    assert res.scene_result is not None

if __name__ == "__main__":
    test_orchestrator()
    test_orchestrator_failure_isolation()
    print("test_vision_orchestrator PASS")
