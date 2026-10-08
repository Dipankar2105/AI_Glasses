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

def test_convert_rgb_to_grayscale():
    pipeline = VisionPipeline()
    img = np.zeros((5, 1, 3), dtype=np.uint8)
    img[0, 0] = [255, 0, 0]     # Red -> 76
    img[1, 0] = [0, 255, 0]     # Green -> 150
    img[2, 0] = [0, 0, 255]     # Blue -> 29
    img[3, 0] = [255, 255, 255] # White -> 255
    img[4, 0] = [0, 0, 0]       # Black -> 0
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 1, 5, 3, "RGB", (0, 255), 100, 1)
    
    gray = pipeline.convert_format(vframe, "GRAYSCALE")
    # wait, data shape removed
    assert gray.data.shape == (5, 1)
    assert gray.channels == 1
    assert gray.pixel_format == "GRAYSCALE"
    
    expected = np.array([[76], [150], [29], [255], [0]], dtype=np.uint8)
    assert np.array_equal(gray.data, expected)

def test_convert_grayscale_to_rgb():
    pipeline = VisionPipeline()
    img = np.array([[0, 127], [200, 255]], dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 2, 2, 1, "GRAYSCALE", (0, 255), 100, 1)
    
    rgb = pipeline.convert_format(vframe, "RGB")
    assert rgb.data.shape == (2, 2, 3)
    assert rgb.channels == 3
    assert rgb.pixel_format == "RGB"
    
    assert np.array_equal(rgb.data[:, :, 0], img)
    assert np.array_equal(rgb.data[:, :, 1], img)
    assert np.array_equal(rgb.data[:, :, 2], img)

def test_convert_rgb_identity():
    pipeline = VisionPipeline()
    img = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 4, 4, 3, "RGB", (0, 255), 100, 1)
    
    rgb = pipeline.convert_format(vframe, "RGB")
    assert np.array_equal(rgb.data, img)
    assert rgb is vframe

def test_convert_grayscale_identity():
    pipeline = VisionPipeline()
    img = np.random.randint(0, 256, (4, 4), dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 4, 4, 1, "GRAYSCALE", (0, 255), 100, 1)
    
    gray = pipeline.convert_format(vframe, "GRAYSCALE")
    assert np.array_equal(gray.data, img)
    assert gray is vframe

def test_convert_channel_distinctness():
    pipeline = VisionPipeline()
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:, :, 0] = 255 # Only red
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 2, 2, 3, "RGB", (0, 255), 100, 1)
    
    gray = pipeline.convert_format(vframe, "GRAYSCALE")
    # Red only is ~76
    assert np.all(gray.data == 76)

def test_convert_metadata_preservation():
    pipeline = VisionPipeline()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 4, 4, 3, "RGB", (0, 255), 12345, 99, {"operations": ["resize"]})
    
    gray = pipeline.convert_format(vframe, "GRAYSCALE")
    assert gray.timestamp == 12345
    assert gray.seq_num == 99
    assert gray.numerical_range == (0, 255)
    assert "convert_format" in gray.preprocessing_metadata["operations"]
    assert "resize" in gray.preprocessing_metadata["operations"]
    assert gray.preprocessing_metadata["conversion"]["input_format"] == "RGB"
    assert gray.preprocessing_metadata["conversion"]["output_format"] == "GRAYSCALE"

def test_convert_invalid_input():
    pipeline = VisionPipeline()
    from backend.vision.frame import VisionFrame
    
    vframe1 = VisionFrame(None, 4, 4, 1, "GRAYSCALE", (0, 255), 100, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.convert_format(vframe1, "RGB")
        
    img_1d = np.zeros((16,), dtype=np.uint8)
    vframe2 = VisionFrame(img_1d, 4, 4, 1, "GRAYSCALE", (0, 255), 100, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.convert_format(vframe2, "RGB")
        
    img_bad_rgb = np.zeros((4, 4, 4), dtype=np.uint8)
    vframe3 = VisionFrame(img_bad_rgb, 4, 4, 3, "RGB", (0, 255), 100, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.convert_format(vframe3, "GRAYSCALE")


def test_normalize_grayscale():
    pipeline = VisionPipeline()
    img = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 2, 2, 1, "GRAYSCALE", (0, 255), 100, 1)
    
    norm = pipeline.normalize_frame(vframe)
    assert norm.data.shape == (2, 2)
    assert norm.data.dtype == np.float32
    assert norm.channels == 1
    assert norm.numerical_range == (0.0, 1.0)
    assert "normalize" in norm.preprocessing_metadata["operations"]
    
    expected = np.array([[0.0, 64/255.0], [128/255.0, 1.0]], dtype=np.float32)
    assert np.allclose(norm.data, expected)
    assert np.min(norm.data) >= 0.0
    assert np.max(norm.data) <= 1.0

def test_normalize_rgb():
    pipeline = VisionPipeline()
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0] = [0, 128, 255]
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 2, 2, 3, "RGB", (0, 255), 100, 1)
    
    norm = pipeline.normalize_frame(vframe)
    assert norm.data.shape == (2, 2, 3)
    assert norm.data.dtype == np.float32
    assert norm.channels == 3
    assert norm.numerical_range == (0.0, 1.0)
    
    expected = np.zeros((2, 2, 3), dtype=np.float32)
    expected[0, 0] = [0.0, 128/255.0, 1.0]
    assert np.allclose(norm.data, expected)

def test_normalize_invalid_input():
    pipeline = VisionPipeline()
    from backend.vision.frame import VisionFrame
    
    # Missing data
    vframe1 = VisionFrame(None, 4, 4, 1, "GRAYSCALE", (0, 255), 100, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.normalize_frame(vframe1)
        
    # Already float (invalid dtype)
    img_float = np.zeros((4, 4), dtype=np.float32)
    vframe2 = VisionFrame(img_float, 4, 4, 1, "GRAYSCALE", (0.0, 1.0), 100, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.normalize_frame(vframe2)


def test_contrast_grayscale_uint8():
    pipeline = VisionPipeline()
    img = np.array([[50, 100], [150, 200]], dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 2, 2, 1, "GRAYSCALE", (0, 255), 100, 1)
    
    stretched = pipeline.contrast_normalize(vframe)
    assert stretched.data.dtype == np.uint8
    assert stretched.numerical_range == (0, 255)
    
    # 50->0, 200->255. Span=150.
    # 100 -> (100-50)/150 = 1/3 -> 255/3 = 85
    # 150 -> (150-50)/150 = 2/3 -> 170
    expected = np.array([[0, 85], [170, 255]], dtype=np.uint8)
    assert np.array_equal(stretched.data, expected)

def test_contrast_rgb_uint8_per_channel():
    pipeline = VisionPipeline()
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:, :, 0] = [[50, 60], [90, 100]] # R: 50..100
    img[:, :, 1] = [[100, 120], [180, 200]] # G: 100..200
    img[:, :, 2] = [[25, 35], [65, 75]] # B: 25..75
    
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 2, 2, 3, "RGB", (0, 255), 100, 1)
    
    stretched = pipeline.contrast_normalize(vframe, per_channel=True)
    assert stretched.data.dtype == np.uint8
    
    # R: 50->0, 100->255
    # G: 100->0, 200->255
    # B: 25->0, 75->255
    assert np.min(stretched.data[:, :, 0]) == 0 and np.max(stretched.data[:, :, 0]) == 255
    assert np.min(stretched.data[:, :, 1]) == 0 and np.max(stretched.data[:, :, 1]) == 255
    assert np.min(stretched.data[:, :, 2]) == 0 and np.max(stretched.data[:, :, 2]) == 255

def test_contrast_grayscale_float32():
    pipeline = VisionPipeline()
    img = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 2, 2, 1, "FLOAT32_GRAYSCALE", (0.0, 1.0), 100, 1)
    
    stretched = pipeline.contrast_normalize(vframe)
    assert stretched.data.dtype == np.float32
    assert stretched.numerical_range == (0.0, 1.0)
    
    # 0.2->0.0, 0.8->1.0. 
    # 0.4 -> (0.4-0.2)/0.6 = 1/3
    expected = np.array([[0.0, 1/3], [2/3, 1.0]], dtype=np.float32)
    assert np.allclose(stretched.data, expected)

def test_contrast_rgb_float32():
    pipeline = VisionPipeline()
    img = np.zeros((2, 2, 3), dtype=np.float32)
    img[:, :, 0] = [[0.1, 0.2], [0.3, 0.4]]
    img[:, :, 1] = [[0.5, 0.6], [0.7, 0.8]]
    img[:, :, 2] = [[0.2, 0.3], [0.4, 0.5]]
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 2, 2, 3, "FLOAT32_RGB", (0.0, 1.0), 100, 1)
    
    stretched = pipeline.contrast_normalize(vframe, per_channel=True)
    assert stretched.data.dtype == np.float32
    assert np.allclose(np.min(stretched.data[:, :, 0]), 0.0)
    assert np.allclose(np.max(stretched.data[:, :, 0]), 1.0)
    assert np.allclose(np.min(stretched.data[:, :, 1]), 0.0)
    assert np.allclose(np.max(stretched.data[:, :, 1]), 1.0)

def test_contrast_constant_image():
    pipeline = VisionPipeline()
    img = np.full((2, 2), 128, dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 2, 2, 1, "GRAYSCALE", (0, 255), 100, 1)
    
    stretched = pipeline.contrast_normalize(vframe)
    assert np.array_equal(stretched.data, img) # Preserved exactly without division by zero NaN
    assert stretched.data.dtype == np.uint8

def test_contrast_metadata_preservation():
    pipeline = VisionPipeline()
    img = np.array([[50, 200]], dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 2, 1, 1, "GRAYSCALE", (0, 255), 12345, 99, {"operations": ["resize"]})
    
    stretched = pipeline.contrast_normalize(vframe)
    assert stretched.timestamp == 12345
    assert stretched.seq_num == 99
    assert "contrast_normalize" in stretched.preprocessing_metadata["operations"]
    assert "resize" in stretched.preprocessing_metadata["operations"]
    assert stretched.preprocessing_metadata["contrast"]["method"] == "min_max"

def test_contrast_invalid_input():
    pipeline = VisionPipeline()
    from backend.vision.frame import VisionFrame
    
    # Missing data
    vframe1 = VisionFrame(None, 4, 4, 1, "GRAYSCALE", (0, 255), 100, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.contrast_normalize(vframe1)
        
    # Invalid dtype
    img_int16 = np.zeros((4, 4), dtype=np.int16)
    vframe2 = VisionFrame(img_int16, 4, 4, 1, "GRAYSCALE", (0, 255), 100, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.contrast_normalize(vframe2)
        
    # Invalid range for float32
    img_float_bad = np.zeros((4, 4), dtype=np.float32)
    vframe3 = VisionFrame(img_float_bad, 4, 4, 1, "GRAYSCALE", (0.0, 255.0), 100, 1)
    with pytest.raises(VisionPipelineError):
        pipeline.contrast_normalize(vframe3)

def test_contrast_nan_inf_safety_and_monotonicity():
    pipeline = VisionPipeline()
    img = np.array([[50, 100, 150, 200]], dtype=np.uint8)
    from backend.vision.frame import VisionFrame
    vframe = VisionFrame(img, 4, 1, 1, "GRAYSCALE", (0, 255), 100, 1)
    
    stretched = pipeline.contrast_normalize(vframe)
    data = stretched.data
    
    assert np.all(np.isfinite(data))
    # Check monotonicity
    assert data[0, 0] < data[0, 1] < data[0, 2] < data[0, 3]

