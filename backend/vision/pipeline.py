import numpy as np
from backend.vision.frame import ValidationStatus, VisionFrame, OCRInput, DetectionInput, SceneAnalysisInput
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../firmware')))
from hal.camera import CameraFrame

class VisionPipelineError(Exception):
    pass

class VisionPipeline:
    def __init__(self):
        pass

    def validate_frame(self, frame: CameraFrame) -> ValidationStatus:
        if frame is None or getattr(frame, 'data', None) is None:
            return ValidationStatus.UNAVAILABLE
        if frame.width <= 0 or frame.height <= 0:
            return ValidationStatus.INVALID
        if len(frame.data) == 0:
            return ValidationStatus.UNAVAILABLE
            
        expected_len = frame.width * frame.height
        if frame.pixel_format == "RGB" and len(frame.data) != expected_len * 3:
            return ValidationStatus.INVALID
        if frame.pixel_format == "GRAYSCALE" and len(frame.data) != expected_len:
            return ValidationStatus.INVALID
        if frame.pixel_format not in ("RGB", "GRAYSCALE"):
            return ValidationStatus.UNSUPPORTED
            
        return ValidationStatus.VALID

    def process(self, frame: CameraFrame) -> VisionFrame:
        status = self.validate_frame(frame)
        if status != ValidationStatus.VALID:
            raise VisionPipelineError(f"Frame validation failed: {status}")

        channels = 3 if frame.pixel_format == "RGB" else 1
        
        # Convert bytes to numpy array
        # Phase 4A: Identity pipeline. We preserve real pixel data without mock operations.
        # Ensure representation is explicit.
        try:
            img_array = np.frombuffer(frame.data, dtype=np.uint8)
            
            if channels == 3:
                img_array = img_array.reshape((frame.height, frame.width, 3))
            else:
                img_array = img_array.reshape((frame.height, frame.width))
        except ValueError as e:
            raise VisionPipelineError(f"Malformed data dimension: {e}")
            
        # For Phase 4A we do not claim actual resize or contrast normalization.
        # We just preserve the NumPy array.
        
        meta = {"original_size": (frame.width, frame.height), "operations": []}

        return VisionFrame(
            data=img_array, # explicit ndarray representation
            width=frame.width,
            height=frame.height,
            channels=channels,
            pixel_format=frame.pixel_format,
            numerical_range=(0, 255), # Document explicit uint8 numerical range
            timestamp=frame.timestamp,
            seq_num=frame.seq_num,
            preprocessing_metadata=meta
        )

    def create_ocr_input(self, vframe: VisionFrame, roi=None) -> OCRInput:
        return OCRInput(vframe.data, vframe.width, vframe.height, vframe.channels, vframe.numerical_range, vframe.timestamp, vframe.seq_num, roi)

    def create_detection_input(self, vframe: VisionFrame) -> DetectionInput:
        return DetectionInput(vframe.data, vframe.width, vframe.height, vframe.channels, vframe.numerical_range, vframe.timestamp, vframe.seq_num)
        
    def create_scene_analysis_input(self, vframe: VisionFrame, roi=None) -> SceneAnalysisInput:
        return SceneAnalysisInput(vframe.data, vframe.width, vframe.height, vframe.channels, vframe.timestamp, vframe.seq_num, roi, vframe.preprocessing_metadata)
