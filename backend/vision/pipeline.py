import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../firmware')))
from hal.camera import CameraFrame
from .frame import VisionFrame, ValidationStatus, OCRInput, DetectionInput, SceneAnalysisInput

class VisionPipelineError(Exception):
    pass

class VisionPipeline:
    def __init__(self):
        pass

    def validate_frame(self, frame: CameraFrame) -> ValidationStatus:
        if frame is None or frame.data is None or len(frame.data) == 0:
            return ValidationStatus.UNAVAILABLE
        if frame.width == 0 or frame.height == 0:
            return ValidationStatus.INVALID
        if frame.width < 0 or frame.height < 0:
            return ValidationStatus.INVALID
        expected_len = frame.width * frame.height
        if frame.pixel_format == "RGB" and len(frame.data) != expected_len * 3:
            return ValidationStatus.INVALID
        if frame.pixel_format == "GRAYSCALE" and len(frame.data) != expected_len:
            return ValidationStatus.INVALID
        if frame.pixel_format not in ("RGB", "GRAYSCALE"):
            return ValidationStatus.UNSUPPORTED
        return ValidationStatus.VALID

    def rgb_to_grayscale(self, data: bytes, width: int, height: int) -> bytes:
        gray = bytearray(width * height)
        for i in range(width * height):
            r = data[i*3]
            g = data[i*3+1]
            b = data[i*3+2]
            gray[i] = int(0.299*r + 0.587*g + 0.114*b)
        return bytes(gray)

    def extract_roi(self, data: bytes, width: int, height: int, channels: int, roi_x: int, roi_y: int, roi_w: int, roi_h: int) -> tuple[bytes, int, int]:
        if roi_x < 0 or roi_y < 0 or roi_w <= 0 or roi_h <= 0 or roi_x + roi_w > width or roi_y + roi_h > height:
            raise VisionPipelineError("ROI out of bounds")
        out = bytearray(roi_w * roi_h * channels)
        for y in range(roi_h):
            src_y = roi_y + y
            src_idx = (src_y * width + roi_x) * channels
            dst_idx = (y * roi_w) * channels
            row_len = roi_w * channels
            out[dst_idx:dst_idx+row_len] = data[src_idx:src_idx+row_len]
        return bytes(out), roi_w, roi_h

    def resize(self, data: bytes, width: int, height: int, channels: int, target_w: int, target_h: int) -> tuple[bytes, int, int]:
        # Simple nearest neighbor
        out = bytearray(target_w * target_h * channels)
        x_ratio = width / float(target_w)
        y_ratio = height / float(target_h)
        for i in range(target_h):
            for j in range(target_w):
                px = int(j * x_ratio)
                py = int(i * y_ratio)
                src_idx = (py * width + px) * channels
                dst_idx = (i * target_w + j) * channels
                out[dst_idx:dst_idx+channels] = data[src_idx:src_idx+channels]
        return bytes(out), target_w, target_h

    def normalize(self, data: bytes) -> list[float]:
        return [float(b) / 255.0 for b in data]

    def contrast_normalize(self, norm_data: list[float]) -> list[float]:
        if not norm_data: return norm_data
        min_v = min(norm_data)
        max_v = max(norm_data)
        if max_v - min_v < 1e-6:
            return [0.5 for _ in norm_data] # uniform
        span = max_v - min_v
        return [(v - min_v) / span for v in norm_data]

    def evaluate_quality(self, data: list[float]) -> dict:
        if not data:
            return {"brightness": 0.0, "contrast": 0.0, "is_uniform": True, "very_dark": True, "very_bright": False, "unusable": True}
        min_v = min(data)
        max_v = max(data)
        avg = sum(data) / len(data)
        contrast = max_v - min_v
        is_uniform = contrast < 0.01
        very_dark = avg < 0.1
        very_bright = avg > 0.9
        unusable = is_uniform or very_dark or very_bright
        return {
            "brightness": avg,
            "contrast": contrast,
            "is_uniform": is_uniform,
            "very_dark": very_dark,
            "very_bright": very_bright,
            "unusable": unusable
        }

    def process(self, frame: CameraFrame, target_w=None, target_h=None, roi=None, apply_contrast_norm=False) -> VisionFrame:
        status = self.validate_frame(frame)
        if status != ValidationStatus.VALID:
            raise VisionPipelineError(f"Frame validation failed: {status}")

        channels = 3 if frame.pixel_format == "RGB" else 1
        data = frame.data
        w, h = frame.width, frame.height
        
        meta = {"original_size": (w, h), "operations": []}

        if frame.pixel_format == "RGB":
            data = self.rgb_to_grayscale(data, w, h)
            channels = 1
            meta["operations"].append("grayscale")

        if roi:
            data, w, h = self.extract_roi(data, w, h, channels, *roi)
            meta["operations"].append("roi")

        if target_w and target_h:
            data, w, h = self.resize(data, w, h, channels, target_w, target_h)
            meta["operations"].append("resize")

        norm_data = self.normalize(data)
        meta["operations"].append("normalize")

        if apply_contrast_norm:
            norm_data = self.contrast_normalize(norm_data)
            meta["operations"].append("contrast_normalize")

        quality = self.evaluate_quality(norm_data)
        meta["quality"] = quality

        return VisionFrame(
            data=norm_data,
            width=w,
            height=h,
            channels=channels,
            pixel_format="FLOAT32_GRAYSCALE",
            numerical_range=(0.0, 1.0),
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
