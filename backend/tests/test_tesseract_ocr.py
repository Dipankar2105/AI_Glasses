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
        _ = TesseractOCREngine()
        pytesseract.get_tesseract_version()
        return True
    except Exception:
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
    assert 0.0 <= res.regions[0].confidence <= 1.0
    assert 0.0 <= res.regions[0].box.x <= 1.0
    assert 0.0 <= res.regions[0].box.y <= 1.0
    assert 0.0 <= res.regions[0].box.width <= 1.0
    assert 0.0 <= res.regions[0].box.height <= 1.0

@pytest.mark.skipif(not has_tesseract(), reason="Tesseract executable not found")
def test_tesseract_ocr_genuine_blank_image():
    # Blank white image should not invent text or fake regions
    img_np = np.full((100, 100), 255, dtype=np.uint8)
    engine = TesseractOCREngine()
    inp = OCRInput(img_np, 100, 100, 1, (0, 255), 100, 1)
    
    res = engine.process(inp)
    assert res.full_text == ""
    assert len(res.regions) == 0

@pytest.mark.skipif(not has_tesseract(), reason="Tesseract executable not found")
def test_tesseract_ocr_genuine_rgb_and_numbers():
    # Test RGB image with digits
    img_pil = Image.new('RGB', (200, 50), color=(255, 255, 255))
    d = ImageDraw.Draw(img_pil)
    d.text((10, 10), "98765", fill=(0, 0, 0))
    img_np = np.array(img_pil)
    
    engine = TesseractOCREngine()
    inp = OCRInput(img_np, 200, 50, 3, (0, 255), 100, 1)
    
    res = engine.process(inp)
    assert "98765" in res.full_text
    assert len(res.regions) >= 1

@pytest.mark.skipif(not has_tesseract(), reason="Tesseract executable not found")
def test_tesseract_ocr_orchestrator_e2e():
    from firmware.hal.camera import CameraFrame
    from backend.vision.pipeline import VisionPipeline
    from backend.ai.registry import AIEngineRegistry
    from backend.ai.orchestrator import VisionOrchestrator
    from backend.ai.mock_engines import MockObjectDetector, MockSceneAnalyzer
    
    # 1. Create a CameraFrame with synthetic text "HELLO"
    img_pil = Image.new('RGB', (200, 50), color=(255, 255, 255))
    d = ImageDraw.Draw(img_pil)
    d.text((10, 10), "HELLO", fill=(0, 0, 0))
    arr = np.array(img_pil)
    
    cam_frame = CameraFrame(200, 50, "RGB", arr.tobytes(), 54321, 42)
    
    # 2. VisionPipeline processes CameraFrame -> VisionFrame
    pipeline = VisionPipeline()
    vframe = pipeline.process(cam_frame)
    
    # 3. Setup Orchestrator with genuine Tesseract
    registry = AIEngineRegistry()
    registry.register_detector("mock", MockObjectDetector())
    registry.load_production_ocr(set_active=True)
    registry.register_scene_analyzer("mock", MockSceneAnalyzer())
    
    orchestrator = VisionOrchestrator(registry)
    
    # 4. Process through Orchestrator
    res = orchestrator.process(vframe)
    
    # 5. Verify genuine OCR reached UnifiedVisionResult
    assert len(res.errors) == 0
    assert res.timestamp == 54321
    assert res.seq_num == 42
    assert res.ocr_result is not None
    assert "HELLO" in res.ocr_result.full_text.upper()
    assert len(res.ocr_result.regions) >= 1
    assert res.ocr_result.metadata["engine"] == "pytesseract"
