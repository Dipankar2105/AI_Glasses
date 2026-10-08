import numpy as np
from backend.vision.frame import ValidationStatus, VisionFrame, OCRInput, DetectionInput, SceneAnalysisInput, ImageQualityResult
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

    def resize_frame(self, vframe: VisionFrame, target_w: int, target_h: int) -> VisionFrame:
        if target_w <= 0 or target_h <= 0:
            raise VisionPipelineError("Target width and height must be > 0")
        if getattr(vframe, 'data', None) is None or vframe.data.size == 0:
            raise VisionPipelineError("VisionFrame data is empty or missing")
            
        if len(vframe.data.shape) not in (2, 3):
            raise VisionPipelineError("Unsupported dimensionality for resize")
        if vframe.channels not in (1, 3):
            raise VisionPipelineError("Invalid channel count for resize")
        if vframe.data.shape[0] != vframe.height or vframe.data.shape[1] != vframe.width:
            raise VisionPipelineError("Malformed ndarray dimensions mismatch")
            
        if target_w == vframe.width and target_h == vframe.height:
            # Identity resize
            return vframe
            
        h, w = vframe.height, vframe.width
        
        # Nearest-Neighbor interpolation
        row_indices = np.floor(np.arange(target_h) * (h / target_h)).astype(int)
        col_indices = np.floor(np.arange(target_w) * (w / target_w)).astype(int)
        
        resized_data = vframe.data[row_indices[:, None], col_indices]
            
        new_meta = dict(vframe.preprocessing_metadata)
        
        # Copy the operations list to avoid mutating the original
        operations = list(new_meta.get("operations", []))
        operations.append("resize")
        new_meta["operations"] = operations
        
        new_meta["resize"] = {
            "original_size": (w, h),
            "output_size": (target_w, target_h),
            "interpolation": "nearest_neighbor"
        }
        
        return VisionFrame(
            data=resized_data,
            width=target_w,
            height=target_h,
            channels=vframe.channels,
            pixel_format=vframe.pixel_format,
            numerical_range=vframe.numerical_range,
            timestamp=vframe.timestamp,
            seq_num=vframe.seq_num,
            preprocessing_metadata=new_meta
        )

    def convert_format(self, vframe: VisionFrame, target_format: str) -> VisionFrame:
        if target_format not in ("RGB", "GRAYSCALE"):
            raise VisionPipelineError(f"Unsupported target format: {target_format}")
            
        if getattr(vframe, 'data', None) is None or vframe.data.size == 0:
            raise VisionPipelineError("VisionFrame data is empty or missing")
        if len(vframe.data.shape) not in (2, 3):
            raise VisionPipelineError("Unsupported dimensionality for format conversion")
        if vframe.channels not in (1, 3):
            raise VisionPipelineError("Invalid channel count for format conversion")
        if vframe.channels == 3 and vframe.data.shape[-1] != 3:
            raise VisionPipelineError("RGB arrays must have exactly 3 channels in last dimension")

        if vframe.pixel_format == target_format:
            # Identity conversion
            return vframe
            
        new_meta = dict(vframe.preprocessing_metadata)
        operations = list(new_meta.get("operations", []))
        operations.append("convert_format")
        new_meta["operations"] = operations
        new_meta["conversion"] = {
            "input_format": vframe.pixel_format,
            "output_format": target_format
        }

        if target_format == "GRAYSCALE":
            # RGB -> Grayscale
            # Y = 0.299R + 0.587G + 0.114B
            # Using standard rounding before casting to uint8
            r = vframe.data[:, :, 0].astype(np.float32)
            g = vframe.data[:, :, 1].astype(np.float32)
            b = vframe.data[:, :, 2].astype(np.float32)
            gray = np.round(0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
            
            return VisionFrame(
                data=gray,
                width=vframe.width,
                height=vframe.height,
                channels=1,
                pixel_format="GRAYSCALE",
                numerical_range=vframe.numerical_range,
                timestamp=vframe.timestamp,
                seq_num=vframe.seq_num,
                preprocessing_metadata=new_meta
            )
            
        elif target_format == "RGB":
            # Grayscale -> RGB
            rgb = np.stack((vframe.data,)*3, axis=-1)
            
            return VisionFrame(
                data=rgb,
                width=vframe.width,
                height=vframe.height,
                channels=3,
                pixel_format="RGB",
                numerical_range=vframe.numerical_range,
                timestamp=vframe.timestamp,
                seq_num=vframe.seq_num,
                preprocessing_metadata=new_meta
            )

    def normalize_frame(self, vframe: VisionFrame) -> VisionFrame:
        if getattr(vframe, 'data', None) is None or vframe.data.size == 0:
            raise VisionPipelineError("VisionFrame data is empty or missing")
        if vframe.data.dtype != np.uint8:
            raise VisionPipelineError(f"Normalization requires uint8 data, got {vframe.data.dtype}")
        
        normalized_data = (vframe.data.astype(np.float32) / 255.0)
        
        new_meta = dict(vframe.preprocessing_metadata)
        operations = list(new_meta.get("operations", []))
        operations.append("normalize")
        new_meta["operations"] = operations
        
        return VisionFrame(
            data=normalized_data,
            width=vframe.width,
            height=vframe.height,
            channels=vframe.channels,
            pixel_format=vframe.pixel_format,
            numerical_range=(0.0, 1.0),
            timestamp=vframe.timestamp,
            seq_num=vframe.seq_num,
            preprocessing_metadata=new_meta
        )

    def contrast_normalize(self, vframe: VisionFrame, per_channel: bool = True) -> VisionFrame:
        if getattr(vframe, 'data', None) is None or vframe.data.size == 0:
            raise VisionPipelineError("VisionFrame data is empty or missing")
            
        is_uint8 = vframe.data.dtype == np.uint8
        is_float32 = vframe.data.dtype == np.float32
        
        if not (is_uint8 or is_float32):
            raise VisionPipelineError(f"Contrast normalization requires uint8 or float32 data, got {vframe.data.dtype}")
            
        if is_uint8 and vframe.numerical_range != (0, 255):
            raise VisionPipelineError("uint8 data must have numerical range (0, 255)")
        if is_float32 and vframe.numerical_range != (0.0, 1.0):
            raise VisionPipelineError("float32 data must have numerical range (0.0, 1.0)")
            
        # Perform computation in float32 for precision
        data_float = vframe.data.astype(np.float32)
        
        if vframe.channels == 3 and per_channel:
            # Per-channel contrast stretch
            for c in range(3):
                c_min = np.min(data_float[..., c])
                c_max = np.max(data_float[..., c])
                if c_max > c_min:
                    data_float[..., c] = (data_float[..., c] - c_min) / (c_max - c_min)
                else:
                    if is_uint8:
                        data_float[..., c] = data_float[..., c] / 255.0
        else:
            # Global contrast stretch
            c_min = np.min(data_float)
            c_max = np.max(data_float)
            if c_max > c_min:
                data_float = (data_float - c_min) / (c_max - c_min)
            else:
                if is_uint8:
                    data_float = data_float / 255.0
                    
        # Numerical safety explicitly clipping to range
        data_float = np.clip(data_float, 0.0, 1.0)
        if not np.all(np.isfinite(data_float)):
            raise VisionPipelineError("Contrast normalization produced non-finite values")
            
        # Map back to original dtype
        if is_uint8:
            final_data = np.round(data_float * 255.0).astype(np.uint8)
        else:
            final_data = data_float
            
        new_meta = dict(vframe.preprocessing_metadata)
        operations = list(new_meta.get("operations", []))
        operations.append("contrast_normalize")
        new_meta["operations"] = operations
        new_meta["contrast"] = {
            "method": "min_max",
            "per_channel": per_channel and vframe.channels == 3
        }
        
        return VisionFrame(
            data=final_data,
            width=vframe.width,
            height=vframe.height,
            channels=vframe.channels,
            pixel_format=vframe.pixel_format,
            numerical_range=vframe.numerical_range,
            timestamp=vframe.timestamp,
            seq_num=vframe.seq_num,
            preprocessing_metadata=new_meta
        )

    def analyze_quality(self, vframe: VisionFrame) -> ImageQualityResult:
        if getattr(vframe, 'data', None) is None or vframe.data.size == 0:
            raise VisionPipelineError("VisionFrame data is empty or missing")
        if len(vframe.data.shape) not in (2, 3):
            raise VisionPipelineError("Unsupported dimensionality for quality analysis")
        if vframe.channels not in (1, 3):
            raise VisionPipelineError("Invalid channel count for quality analysis")
            
        is_uint8 = vframe.data.dtype == np.uint8
        is_float32 = vframe.data.dtype == np.float32
        
        if not (is_uint8 or is_float32):
            raise VisionPipelineError(f"Quality analysis requires uint8 or float32 data, got {vframe.data.dtype}")
            
        if is_uint8 and vframe.numerical_range != (0, 255):
            raise VisionPipelineError("uint8 data must have numerical range (0, 255)")
        if is_float32 and vframe.numerical_range != (0.0, 1.0):
            raise VisionPipelineError("float32 data must have numerical range (0.0, 1.0)")
            
        # Copy to ensure read-only behavior on original
        original_data = vframe.data
        data = original_data.astype(np.float32)
        
        # Convert to luminance if RGB
        if vframe.channels == 3:
            # Y = 0.299R + 0.587G + 0.114B
            r = data[..., 0]
            g = data[..., 1]
            b = data[..., 2]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            lum = data
            
        # A. Brightness (normalized [0, 1])
        mean_val = np.mean(lum)
        brightness = mean_val / 255.0 if is_uint8 else float(mean_val)
        
        # B. Contrast (normalized stddev)
        std_val = np.std(lum)
        contrast = std_val / 255.0 if is_uint8 else float(std_val)
        
        # C. Sharpness (Variance of Laplacian)
        # D. Noise estimate (Mean absolute deviation of Laplacian)
        # Fast 2D Laplacian using slicing
        lap = np.zeros_like(lum, dtype=np.float32)
        if lum.shape[0] >= 3 and lum.shape[1] >= 3:
            lap[1:-1, 1:-1] = (
                lum[:-2, 1:-1] + lum[2:, 1:-1] +
                lum[1:-1, :-2] + lum[1:-1, 2:] -
                4.0 * lum[1:-1, 1:-1]
            )
        
        sharpness = float(np.var(lap))
        if is_uint8:
            sharpness /= (255.0 * 255.0)
            
        noise_estimate = float(np.std(lap))
        if is_uint8:
            noise_estimate /= 255.0
            
        # E. Dynamic range
        max_val = np.max(lum)
        min_val = np.min(lum)
        dr = float(max_val - min_val)
        dynamic_range = dr / 255.0 if is_uint8 else dr
        
        # Heuristic Quality Score
        quality_score = float(np.clip(contrast * 2.0 + sharpness * 10.0 - noise_estimate, 0.0, 1.0))
        
        return ImageQualityResult(
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            noise_estimate=noise_estimate,
            dynamic_range=dynamic_range,
            width=vframe.width,
            height=vframe.height,
            channels=vframe.channels,
            timestamp=vframe.timestamp,
            seq_num=vframe.seq_num,
            quality_score=quality_score
        )

    def create_ocr_input(self, vframe: VisionFrame, roi=None) -> OCRInput:
        return OCRInput(vframe.data, vframe.width, vframe.height, vframe.channels, vframe.numerical_range, vframe.timestamp, vframe.seq_num, roi)

    def create_detection_input(self, vframe: VisionFrame) -> DetectionInput:
        return DetectionInput(vframe.data, vframe.width, vframe.height, vframe.channels, vframe.numerical_range, vframe.timestamp, vframe.seq_num)
        
    def create_scene_analysis_input(self, vframe: VisionFrame, roi=None) -> SceneAnalysisInput:
        return SceneAnalysisInput(vframe.data, vframe.width, vframe.height, vframe.channels, vframe.timestamp, vframe.seq_num, roi, vframe.preprocessing_metadata)
