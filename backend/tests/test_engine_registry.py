import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.ai.registry import AIEngineRegistry
from backend.ai.mock_engines import MockObjectDetector
from backend.ai.errors import AIVisionError

def test_registry():
    r = AIEngineRegistry()
    try:
        r.get_detector()
        assert False
    except AIVisionError:
        pass
    
    r.register_detector("mock", MockObjectDetector())
    d = r.get_detector()
    assert isinstance(d, MockObjectDetector)

if __name__ == "__main__":
    test_registry()
    print("test_engine_registry PASS")
