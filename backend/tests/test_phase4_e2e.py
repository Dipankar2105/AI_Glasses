import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from backend.vision.pipeline import VisionPipeline
from backend.vision.mock_camera import DeterministicMockCamera

def test_phase4_e2e():
    cam = DeterministicMockCamera(2, 2)
    pipeline = VisionPipeline()
    
    # Valid Checkerboard
    frame = cam.checkerboard_frame()
    vframe = pipeline.process(frame)
    
    # Verify Metadata
    assert vframe.width == 2
    assert vframe.channels == 1
    assert vframe.timestamp == frame.timestamp
    assert vframe.seq_num == frame.seq_num
    
    # Verify actual image data
    assert vframe.data.dtype == np.uint8
    assert vframe.data.shape == (2, 2)
    
    # Inputs
    ocr = pipeline.create_ocr_input(vframe, roi=(0,0,1,1))
    assert ocr.dimensions == (2, 2)
    assert ocr.roi == (0,0,1,1)
    
    det = pipeline.create_detection_input(vframe)
    assert det.normalization == (0, 255)
    
    sce = pipeline.create_scene_analysis_input(vframe)
    assert sce.preprocessing_metadata is not None
    
    print("test_phase4_e2e PASS")

if __name__ == "__main__":
    test_phase4_e2e()

def test_phase4_integration_rgb_path():
    pipeline = VisionPipeline()
    cam = DeterministicMockCamera(8, 8)
    frame = cam.rgb_frame()
    
    # 1. Process (uint8 RGB)
    vframe1 = pipeline.process(frame)
    assert vframe1.data.dtype == np.uint8
    assert vframe1.channels == 3
    
    # 2. Resize
    vframe2 = pipeline.resize_frame(vframe1, 4, 4)
    assert vframe2.width == 4 and vframe2.height == 4
    assert vframe2.channels == 3
    
    # 3. Grayscale
    vframe3 = pipeline.convert_format(vframe2, "GRAYSCALE")
    assert vframe3.channels == 1
    assert vframe3.pixel_format == "GRAYSCALE"
    
    # 4. Normalize
    vframe4 = pipeline.normalize_frame(vframe3)
    assert vframe4.data.dtype == np.float32
    assert vframe4.numerical_range == (0.0, 1.0)
    
    # 5. Quality Analysis
    q_res = pipeline.analyze_quality(vframe4)
    assert q_res.width == 4 and q_res.height == 4
    assert q_res.channels == 1
    
    # Verify metadata and invariants
    assert vframe4.timestamp == frame.timestamp
    assert vframe4.seq_num == frame.seq_num
    assert "operations" in vframe4.preprocessing_metadata
    assert "resize" in vframe4.preprocessing_metadata["operations"]
    assert "convert_format" in vframe4.preprocessing_metadata["operations"]
    assert "normalize" in vframe4.preprocessing_metadata["operations"]

def test_phase4_integration_grayscale_path():
    pipeline = VisionPipeline()
    cam = DeterministicMockCamera(8, 8)
    frame = cam.checkerboard_frame()
    
    # 1. Process
    vframe1 = pipeline.process(frame)
    assert vframe1.pixel_format == "GRAYSCALE"
    
    # 2. Resize
    vframe2 = pipeline.resize_frame(vframe1, 4, 4)
    
    # 3. Normalize
    vframe3 = pipeline.normalize_frame(vframe2)
    assert vframe3.data.dtype == np.float32
    
    # 4. Contrast normalization
    vframe4 = pipeline.contrast_normalize(vframe3)
    assert vframe4.data.dtype == np.float32
    
    # 5. Quality Analysis (verify read-only)
    original_data = vframe4.data.copy()
    q_res = pipeline.analyze_quality(vframe4)
    assert np.array_equal(vframe4.data, original_data)
    
    # Check consistency
    assert vframe4.width == 4
    assert vframe4.seq_num == frame.seq_num

def test_phase4_integration_invalid_input():
    pipeline = VisionPipeline()
    from backend.vision.frame import VisionFrame
    from backend.vision.pipeline import VisionPipelineError
    
    # Try normalize on float32 (should fail)
    img_float = np.zeros((4, 4), dtype=np.float32)
    vframe_bad1 = VisionFrame(img_float, 4, 4, 1, "GRAYSCALE", (0.0, 1.0), 1, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.normalize_frame(vframe_bad1)
        
    # Try convert format with invalid target
    vframe_good = VisionFrame(np.zeros((4, 4), dtype=np.uint8), 4, 4, 1, "GRAYSCALE", (0, 255), 1, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.convert_format(vframe_good, "INVALID")

import pytest
