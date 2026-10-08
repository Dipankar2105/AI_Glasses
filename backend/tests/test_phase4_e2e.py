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
