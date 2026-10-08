import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pytest
from backend.vision.pipeline import VisionPipeline, VisionPipelineError
from backend.vision.mock_camera import DeterministicMockCamera
from firmware.hal.camera import CameraFrame

def test_pipeline_identity_grayscale():
    mock_cam = DeterministicMockCamera(width=8, height=8)
    frame = mock_cam.checkerboard_frame()
    pipeline = VisionPipeline()
    vframe = pipeline.process(frame)
    
    assert vframe.width == 8
    assert vframe.height == 8
    assert vframe.channels == 1
    assert vframe.pixel_format == "GRAYSCALE"
    assert vframe.data.dtype == np.uint8
    assert vframe.data.shape == (8, 8)
    
    expected_data = np.frombuffer(frame.data, dtype=np.uint8).reshape((8, 8))
    assert np.array_equal(vframe.data, expected_data)

def test_pipeline_identity_rgb():
    mock_cam = DeterministicMockCamera(width=8, height=8)
    frame = mock_cam.rgb_frame()
    pipeline = VisionPipeline()
    vframe = pipeline.process(frame)
    
    assert vframe.channels == 3
    assert vframe.pixel_format == "RGB"
    assert vframe.data.shape == (8, 8, 3)
    
    expected_data = np.frombuffer(frame.data, dtype=np.uint8).reshape((8, 8, 3))
    assert np.array_equal(vframe.data, expected_data)

def test_pipeline_metadata_preservation():
    frame = CameraFrame(width=4, height=4, pixel_format="GRAYSCALE", data=bytes([0]*16), timestamp=999, seq_num=42)
    pipeline = VisionPipeline()
    vframe = pipeline.process(frame)
    
    assert vframe.timestamp == 999
    assert vframe.seq_num == 42
    assert vframe.width == 4
    assert vframe.height == 4

def test_pipeline_invalid_inputs():
    pipeline = VisionPipeline()
    
    # Empty dimensions
    f1 = CameraFrame(width=0, height=0, pixel_format="GRAYSCALE", data=bytes([]), timestamp=1, seq_num=1)
    with pytest.raises(VisionPipelineError):
        pipeline.process(f1)
        
    # None data
    f2 = CameraFrame(width=4, height=4, pixel_format="GRAYSCALE", data=None, timestamp=1, seq_num=1)
    with pytest.raises(VisionPipelineError):
        pipeline.process(f2)
        
    # Dimension mismatch (width * height != len(data))
    f3 = CameraFrame(width=4, height=4, pixel_format="GRAYSCALE", data=bytes([0]*15), timestamp=1, seq_num=1)
    with pytest.raises(VisionPipelineError):
        pipeline.process(f3)
        
    # RGB Dimension mismatch
    f4 = CameraFrame(width=4, height=4, pixel_format="RGB", data=bytes([0]*47), timestamp=1, seq_num=1)
    with pytest.raises(VisionPipelineError):
        pipeline.process(f4)
