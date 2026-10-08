import numpy as np
from enum import Enum

class ValidationStatus(Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNSUPPORTED = "UNSUPPORTED"
    UNAVAILABLE = "UNAVAILABLE"

class VisionFrame:
    def __init__(self, data: np.ndarray, width: int, height: int, channels: int, pixel_format: str, numerical_range: tuple, timestamp: int, seq_num: int, preprocessing_metadata: dict = None):
        self.data = data
        self.width = width
        self.height = height
        self.channels = channels
        self.pixel_format = pixel_format
        self.numerical_range = numerical_range
        self.timestamp = timestamp
        self.seq_num = seq_num
        self.preprocessing_metadata = preprocessing_metadata or {}
        
class OCRInput:
    def __init__(self, image: np.ndarray, width: int, height: int, channels: int, numerical_range: tuple, timestamp: int, seq_num: int, roi: tuple = None):
        self.image = image
        self.dimensions = (width, height)
        self.channels = channels
        self.numerical_range = numerical_range
        self.timestamp = timestamp
        self.seq_num = seq_num
        self.roi = roi

class DetectionInput:
    def __init__(self, image: np.ndarray, width: int, height: int, channels: int, normalization: tuple, timestamp: int, seq_num: int):
        self.image = image
        self.dimensions = (width, height)
        self.channels = channels
        self.normalization = normalization
        self.timestamp = timestamp
        self.seq_num = seq_num

class SceneAnalysisInput:
    def __init__(self, image: np.ndarray, width: int, height: int, channels: int, timestamp: int, seq_num: int, roi: tuple = None, preprocessing_metadata: dict = None):
        self.image = image
        self.dimensions = (width, height)
        self.channels = channels
        self.timestamp = timestamp
        self.seq_num = seq_num
        self.roi = roi
        self.preprocessing_metadata = preprocessing_metadata or {}
