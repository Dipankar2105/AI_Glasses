import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.ai.mock_engines import MockOCREngine
from backend.vision.frame import OCRInput

def test_mock_ocr():
    ocr = MockOCREngine()
    inp = OCRInput([0], 1, 1, 1, (0,1), 0, 0)
    res = ocr.process(inp)
    assert res.full_text == "mock text"

if __name__ == "__main__":
    test_mock_ocr()
    print("test_ocr PASS")
