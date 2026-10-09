import sys, os
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.ai.tesseract_ocr import TesseractOCREngine
from backend.vision.frame import OCRInput
from backend.ai.errors import AIVisionError, AIVisionErrorStatus
import pytesseract

def test_tesseract_ocr_invalid_input():
    engine = TesseractOCREngine()
    
    # Missing Image
    with pytest.raises(AIVisionError) as e:
        engine.process(OCRInput(None, 4, 4, 1, (0, 255), 100, 1))
    assert e.value.status == AIVisionErrorStatus.INVALID_INPUT
    
    # Float32 instead of uint8
    img_f32 = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(AIVisionError) as e:
        engine.process(OCRInput(img_f32, 4, 4, 1, (0.0, 1.0), 100, 1))
    assert e.value.status == AIVisionErrorStatus.INVALID_INPUT
    
    # Invalid channel
    img_bad_c = np.zeros((4, 4, 2), dtype=np.uint8)
    with pytest.raises(AIVisionError) as e:
        engine.process(OCRInput(img_bad_c, 4, 4, 2, (0, 255), 100, 1))
    assert e.value.status == AIVisionErrorStatus.INVALID_INPUT

@patch('pytesseract.image_to_data')
def test_tesseract_ocr_mocked_success(mock_image_to_data):
    # Setup mock return data
    mock_image_to_data.return_value = {
        'level': [1, 2, 3],
        'text': ['', 'Hello', 'World'],
        'conf': ['-1', '95', '98'],
        'left': [0, 10, 50],
        'top': [0, 20, 20],
        'width': [100, 30, 40],
        'height': [100, 15, 15]
    }
    
    engine = TesseractOCREngine()
    img = np.zeros((100, 100), dtype=np.uint8)
    inp = OCRInput(img, 100, 100, 1, (0, 255), 12345, 99)
    
    res = engine.process(inp)
    
    assert res.full_text == "Hello World"
    assert len(res.regions) == 2
    
    reg1 = res.regions[0]
    assert reg1.text == "Hello"
    assert np.isclose(reg1.confidence, 0.95)
    assert np.isclose(reg1.box.x, 10/100)
    assert np.isclose(reg1.box.y, 20/100)
    assert np.isclose(reg1.box.width, 30/100)
    assert np.isclose(reg1.box.height, 15/100)
    
    assert res.metadata["engine"] == "pytesseract"
    assert res.metadata["timestamp"] == 12345
    assert res.metadata["seq_num"] == 99

@patch('pytesseract.image_to_data')
def test_tesseract_ocr_mocked_no_tesseract(mock_image_to_data):
    mock_image_to_data.side_effect = pytesseract.TesseractNotFoundError()
    
    engine = TesseractOCREngine()
    img = np.zeros((10, 10), dtype=np.uint8)
    inp = OCRInput(img, 10, 10, 1, (0, 255), 100, 1)
    
    with pytest.raises(AIVisionError) as e:
        engine.process(inp)
    assert e.value.status == AIVisionErrorStatus.ENGINE_FAILURE
    assert "Tesseract executable not found" in str(e.value)

def has_tesseract():
    try:
        pytesseract.get_tesseract_version()
        return True
    except:
        return False

@pytest.mark.skipif(not has_tesseract(), reason="Tesseract executable not found")
def test_tesseract_ocr_genuine_integration():
    # Create a synthetic image with text "HELLO"
    img_pil = Image.new('L', (200, 50), color=255)
    d = ImageDraw.Draw(img_pil)
    d.text((10, 10), "HELLO", fill=0)
    
    img_np = np.array(img_pil)
    
    engine = TesseractOCREngine()
    inp = OCRInput(img_np, 200, 50, 1, (0, 255), 100, 1)
    
    res = engine.process(inp)
    assert "HELLO" in res.full_text.upper()
    assert len(res.regions) >= 1
