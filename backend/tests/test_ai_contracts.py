import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.ai.contracts import validate_confidence, BoundingBox

def test_confidence():
    assert validate_confidence(0.5) == 0.5
    try:
        validate_confidence(1.5)
        assert False
    except ValueError:
        pass
    try:
        validate_confidence(-0.1)
        assert False
    except ValueError:
        pass

def test_bounding_box():
    b = BoundingBox(0.0, 0.0, 1.0, 1.0)
    assert b.x == 0.0
    try:
        BoundingBox(1.1, 0.0, 1.0, 1.0)
        assert False
    except ValueError:
        pass

if __name__ == "__main__":
    test_confidence()
    test_bounding_box()
    print("test_ai_contracts PASS")
