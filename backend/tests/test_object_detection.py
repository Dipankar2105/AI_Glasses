import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.ai.contracts import Detection, BoundingBox, DetectionResult
from backend.ai.mock_engines import MockObjectDetector
from backend.vision.frame import DetectionInput

def test_mock_detector():
    det = MockObjectDetector()
    inp = DetectionInput([0], 1, 1, 1, (0,1), 0, 0)
    res = det.process(inp)
    assert len(res.detections) == 1
    assert res.detections[0].label == "mock_object"
    assert res.detections[0].confidence == 0.95

if __name__ == "__main__":
    test_mock_detector()
    print("test_object_detection PASS")
