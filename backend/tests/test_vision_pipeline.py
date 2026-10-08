import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from firmware.hal.camera import CameraFrame
from backend.vision.pipeline import VisionPipeline, VisionPipelineError, ValidationStatus

def test_validation():
    p = VisionPipeline()
    assert p.validate_frame(CameraFrame(0, 0, "RGB", b"", 0, 0)) == ValidationStatus.UNAVAILABLE
    assert p.validate_frame(CameraFrame(2, 2, "RGB", b"123", 0, 0)) == ValidationStatus.INVALID # bad length
    assert p.validate_frame(CameraFrame(2, 2, "YUV", b"1234", 0, 0)) == ValidationStatus.UNSUPPORTED
    assert p.validate_frame(CameraFrame(2, 2, "GRAYSCALE", b"1234", 0, 0)) == ValidationStatus.VALID

def test_preprocessing():
    p = VisionPipeline()
    
    # RGB to Gray
    rgb = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
    gray = p.rgb_to_grayscale(rgb, 2, 2)
    assert len(gray) == 4
    assert gray[0] == 76 # ~0.299*255
    assert gray[1] == 149 # ~0.587*255
    assert gray[2] == 29 # ~0.114*255
    
    # Resize
    raw = bytes([10, 20, 30, 40])
    res, w, h = p.resize(raw, 2, 2, 1, 1, 1)
    assert w == 1 and h == 1 and res[0] == 10
    
    # ROI
    roi, w, h = p.extract_roi(raw, 2, 2, 1, 1, 1, 1, 1)
    assert w == 1 and h == 1 and roi[0] == 40
    try:
        p.extract_roi(raw, 2, 2, 1, 5, 5, 1, 1)
        assert False
    except VisionPipelineError:
        pass
        
    # Contrast Norm
    cn = p.contrast_normalize([0.1, 0.9])
    assert cn == [0.0, 1.0]

def test_quality():
    p = VisionPipeline()
    q_blank = p.evaluate_quality([0.0]*10)
    assert q_blank["very_dark"] and q_blank["is_uniform"] and q_blank["unusable"]
    
    q_bright = p.evaluate_quality([1.0]*10)
    assert q_bright["very_bright"] and q_bright["is_uniform"] and q_bright["unusable"]
    
    q_good = p.evaluate_quality([0.0, 0.5, 1.0])
    assert not q_good["unusable"]

if __name__ == "__main__":
    test_validation()
    test_preprocessing()
    test_quality()
    print("test_vision_pipeline PASS")
