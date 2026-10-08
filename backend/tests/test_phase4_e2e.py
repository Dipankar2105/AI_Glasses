import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.vision.pipeline import VisionPipeline
from backend.vision.mock_camera import DeterministicMockCamera

def test_phase4_e2e():
    cam = DeterministicMockCamera(2, 2)
    pipeline = VisionPipeline()
    
    # Valid Checkerboard
    frame = cam.checkerboard_frame()
    vframe = pipeline.process(frame, target_w=2, target_h=2, roi=(0,0,2,2), apply_contrast_norm=True)
    
    # Verify Metadata
    assert vframe.width == 2
    assert vframe.channels == 1
    assert vframe.timestamp == frame.timestamp
    assert vframe.seq_num == frame.seq_num
    assert "quality" in vframe.preprocessing_metadata
    
    # Inputs
    ocr = pipeline.create_ocr_input(vframe, roi=(0,0,1,1))
    assert ocr.dimensions == (2, 2)
    assert ocr.roi == (0,0,1,1)
    
    det = pipeline.create_detection_input(vframe)
    assert det.normalization == (0.0, 1.0)
    
    sce = pipeline.create_scene_analysis_input(vframe)
    assert sce.preprocessing_metadata is not None
    
    print("test_phase4_e2e PASS")

if __name__ == "__main__":
    test_phase4_e2e()
