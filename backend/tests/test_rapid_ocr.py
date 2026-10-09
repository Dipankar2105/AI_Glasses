import pytest
import numpy as np
from PIL import Image, ImageDraw

from backend.ai.rapid_ocr import RapidOCREngine
from backend.vision.frame import OCRInput
from backend.ai.errors import AIVisionError, AIVisionErrorStatus
from backend.ai.contracts import OCREngine, OCRResult

def test_rapid_ocr_initialization():
    engine = RapidOCREngine()
    assert isinstance(engine, OCREngine)
    assert hasattr(engine, '_engine')

def test_rapid_ocr_valid_rgb_fixture():
    # Render synthetic text fixture
    img = Image.new('RGB', (300, 80), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 25), "HELLO WORLD", fill=(0, 0, 0))
    arr = np.array(img, dtype=np.uint8)

    inp = OCRInput(
        image=arr,
        width=300,
        height=80,
        channels=3,
        numerical_range=(0, 255),
        timestamp=1000,
        seq_num=1
    )

    engine = RapidOCREngine()
    result = engine.process(inp)

    assert isinstance(result, OCRResult)
    assert "HELLO" in result.full_text.upper() or "WORLD" in result.full_text.upper()
    assert len(result.regions) > 0
    
    # Check normalized bounding boxes
    for reg in result.regions:
        assert 0.0 <= reg.box.x <= 1.0
        assert 0.0 <= reg.box.y <= 1.0
        assert 0.0 <= reg.box.width <= 1.0
        assert 0.0 <= reg.box.height <= 1.0
        assert 0.0 <= reg.confidence <= 1.0

def test_rapid_ocr_grayscale_input():
    img = Image.new('L', (300, 80), color=255)
    d = ImageDraw.Draw(img)
    d.text((20, 25), "AI VISION", fill=0)
    arr = np.array(img, dtype=np.uint8)

    inp = OCRInput(
        image=arr,
        width=300,
        height=80,
        channels=1,
        numerical_range=(0, 255),
        timestamp=1000,
        seq_num=2
    )

    engine = RapidOCREngine()
    result = engine.process(inp)
    assert isinstance(result, OCRResult)
    assert "AI" in result.full_text.upper() or "VISION" in result.full_text.upper()

def test_rapid_ocr_blank_image():
    # Blank white image should produce empty result without error
    arr = np.ones((100, 200, 3), dtype=np.uint8) * 255
    inp = OCRInput(
        image=arr,
        width=200,
        height=100,
        channels=3,
        numerical_range=(0, 255),
        timestamp=1000,
        seq_num=3
    )
    engine = RapidOCREngine()
    result = engine.process(inp)
    assert isinstance(result, OCRResult)
    assert result.full_text == ""
    assert len(result.regions) == 0

def test_rapid_ocr_invalid_inputs():
    engine = RapidOCREngine()

    # None input
    with pytest.raises(AIVisionError) as exc_info:
        engine.process(None)
    assert exc_info.value.status == AIVisionErrorStatus.INVALID_INPUT

    # Empty array
    inp_empty = OCRInput(
        image=np.array([], dtype=np.uint8),
        width=0,
        height=0,
        channels=3,
        numerical_range=(0, 255),
        timestamp=1000,
        seq_num=4
    )
    with pytest.raises(AIVisionError) as exc_info:
        engine.process(inp_empty)
    assert exc_info.value.status == AIVisionErrorStatus.INVALID_INPUT

def test_rapid_ocr_model_reuse():
    # Successive calls should reuse the same loaded engine instance
    engine = RapidOCREngine()
    img = Image.new('RGB', (200, 60), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 20), "REUSE", fill=(0, 0, 0))
    arr = np.array(img, dtype=np.uint8)

    inp = OCRInput(
        image=arr,
        width=200,
        height=60,
        channels=3,
        numerical_range=(0, 255),
        timestamp=1000,
        seq_num=5
    )

    res1 = engine.process(inp)
    res2 = engine.process(inp)

    assert "REUSE" in res1.full_text.upper()
    assert "REUSE" in res2.full_text.upper()
    assert res1.metadata["engine"] == "RapidOCR"
