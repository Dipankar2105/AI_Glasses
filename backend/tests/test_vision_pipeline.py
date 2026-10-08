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

def test_resize_grayscale_downscale():
    pipeline = VisionPipeline()
    img = np.tile(np.linspace(0, 255, 8, dtype=np.uint8), (8, 1))
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 8, 8, 1, "GRAYSCALE", (0, 255), 100, 1)
    
    resized = pipeline.resize_frame(vframe, 4, 4)
    assert resized.data.shape == (4, 4)
    assert resized.data.dtype == np.uint8
    assert resized.width == 4 and resized.height == 4
    # Check pixels (nearest neighbor of 8x8 to 4x4 will just sample every 2nd pixel)
    assert np.array_equal(resized.data, img[::2, ::2])

def test_resize_rgb_downscale():
    pipeline = VisionPipeline()
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(0, 255, 8, dtype=np.uint8), (8, 1)) # R
    img[:, :, 1] = 128 # G
    img[:, :, 2] = 255 # B
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 8, 8, 3, "RGB", (0, 255), 100, 1)
    
    resized = pipeline.resize_frame(vframe, 4, 4)
    assert resized.data.shape == (4, 4, 3)
    assert resized.channels == 3
    assert np.array_equal(resized.data[:, :, 0], img[::2, ::2, 0])
    assert np.all(resized.data[:, :, 1] == 128)
    assert np.all(resized.data[:, :, 2] == 255)

def test_resize_identity():
    pipeline = VisionPipeline()
    img = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 8, 8, 1, "GRAYSCALE", (0, 255), 100, 1)
    
    resized = pipeline.resize_frame(vframe, 8, 8)
    assert np.array_equal(resized.data, img)
    assert resized is vframe # Exact same object for identity

def test_resize_metadata_preservation():
    pipeline = VisionPipeline()
    img = np.zeros((8, 8), dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 8, 8, 1, "GRAYSCALE", (0, 255), 12345, 99, {"original_size": (8, 8)})
    
    resized = pipeline.resize_frame(vframe, 4, 4)
    assert resized.timestamp == 12345
    assert resized.seq_num == 99
    assert resized.pixel_format == "GRAYSCALE"
    assert resized.numerical_range == (0, 255)
    assert "resize" in resized.preprocessing_metadata
    assert resized.preprocessing_metadata["resize"]["output_size"] == (4, 4)
    assert resized.preprocessing_metadata["original_size"] == (8, 8) # Preserved

def test_resize_invalid_dimensions():
    pipeline = VisionPipeline()
    img = np.zeros((8, 8), dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 8, 8, 1, "GRAYSCALE", (0, 255), 100, 1)
    
    with pytest.raises(VisionPipelineError):
        pipeline.resize_frame(vframe, 0, 10)
    with pytest.raises(VisionPipelineError):
        pipeline.resize_frame(vframe, 10, 0)
    with pytest.raises(VisionPipelineError):
        pipeline.resize_frame(vframe, -1, 10)
    with pytest.raises(VisionPipelineError):
        pipeline.resize_frame(vframe, 10, -1)

def test_resize_malformed_input():
    pipeline = VisionPipeline()
    img = np.zeros((8, 8), dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    
    # Missing data
    vframe1 = VisionFrame(None, 8, 8, 1, "GRAYSCALE", (0, 255), 100, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.resize_frame(vframe1, 4, 4)
        
    # Unsupported dimensionality (1D)
    img_1d = np.zeros((64,), dtype=np.uint8)
    vframe2 = VisionFrame(img_1d, 8, 8, 1, "GRAYSCALE", (0, 255), 100, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.resize_frame(vframe2, 4, 4)
        
    # Dimension mismatch
    img_mismatch = np.zeros((10, 10), dtype=np.uint8)
    vframe3 = VisionFrame(img_mismatch, 8, 8, 1, "GRAYSCALE", (0, 255), 100, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.resize_frame(vframe3, 4, 4)
