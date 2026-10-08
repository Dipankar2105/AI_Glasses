import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.ai.mock_engines import MockSceneAnalyzer
from backend.vision.frame import SceneAnalysisInput

def test_mock_scene():
    sce = MockSceneAnalyzer()
    inp = SceneAnalysisInput([0], 1, 1, 1, 0, 0)
    res = sce.process(inp)
    assert res.description == "A mock scene"

if __name__ == "__main__":
    test_mock_scene()
    print("test_scene_analysis PASS")
