import time
from typing import Optional, List
import numpy as np
from PIL import Image

from backend.ai.contracts import OCREngine, OCRResult, OCRTextRegion, BoundingBox
from backend.vision.frame import OCRInput
from backend.ai.errors import AIVisionError, AIVisionErrorStatus

class RapidOCREngine(OCREngine):
    """
    RapidOCR implementation using onnxruntime for lightweight CPU-based scene text and document OCR.
    Loads DBNet detection + SVTR/CRNN recognition ONNX models once upon instantiation and reuses them.
    """
    def __init__(self, **kwargs):
        super().__init__()
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR(**kwargs)
        except ImportError as e:
            raise AIVisionError(
                AIVisionErrorStatus.ENGINE_FAILURE,
                f"rapidocr_onnxruntime is not installed: {e}"
            )
        except Exception as e:
            raise AIVisionError(
                AIVisionErrorStatus.ENGINE_FAILURE,
                f"Failed to initialize RapidOCR engine: {e}"
            )

    def process(self, input_data: OCRInput) -> OCRResult:
        if input_data is None or input_data.image is None:
            raise AIVisionError(
                AIVisionErrorStatus.INVALID_INPUT,
                "Input data or image array is None"
            )

        arr = input_data.image
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            raise AIVisionError(
                AIVisionErrorStatus.INVALID_INPUT,
                "Invalid or empty numpy array provided for OCRInput"
            )

        if not hasattr(input_data, 'dimensions') or input_data.dimensions[0] <= 0 or input_data.dimensions[1] <= 0:
            raise AIVisionError(
                AIVisionErrorStatus.INVALID_INPUT,
                f"Invalid dimensions on OCRInput: {getattr(input_data, 'dimensions', None)}"
            )

        # Convert grayscale (H, W) or (H, W, 1) to RGB (H, W, 3) if needed
        if arr.ndim == 2:
            arr_rgb = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr_rgb = np.concatenate([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            arr_rgb = arr
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr_rgb = arr[:, :, :3]
        else:
            raise AIVisionError(
                AIVisionErrorStatus.INVALID_INPUT,
                f"Unsupported image array shape: {arr.shape}"
            )

        img_h, img_w = arr_rgb.shape[:2]
        t0 = time.perf_counter()

        try:
            raw_result, elapse = self._engine(arr_rgb)
        except Exception as e:
            raise AIVisionError(
                AIVisionErrorStatus.ENGINE_FAILURE,
                f"RapidOCR inference error: {e}"
            )

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        if not raw_result:
            return OCRResult(
                regions=[],
                full_text="",
                metadata={
                    "engine": "RapidOCR",
                    "latency_ms": latency_ms,
                    "raw_elapse": elapse or []
                }
            )

        regions: List[OCRTextRegion] = []
        text_pieces: List[str] = []

        for idx, item in enumerate(raw_result):
            # item format: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], text, confidence_str]
            box_pts = item[0]
            text = str(item[1]).strip()
            try:
                conf = float(item[2])
            except (ValueError, TypeError):
                conf = 0.0

            # Clamp confidence to [0.0, 1.0]
            conf = max(0.0, min(1.0, conf))

            # Compute bounding box from 4 polygon points
            xs = [pt[0] for pt in box_pts]
            ys = [pt[1] for pt in box_pts]

            min_x = max(0.0, float(min(xs)))
            max_x = min(float(img_w), float(max(xs)))
            min_y = max(0.0, float(min(ys)))
            max_y = min(float(img_h), float(max(ys)))

            # Normalized [0.0, 1.0]
            norm_x = min_x / img_w
            norm_y = min_y / img_h
            norm_w = max(0.0, (max_x - min_x) / img_w)
            norm_h = max(0.0, (max_y - min_y) / img_h)

            # Ensure bounds fit in [0.0, 1.0]
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            norm_w = max(0.0, min(1.0 - norm_x, norm_w))
            norm_h = max(0.0, min(1.0 - norm_y, norm_h))

            bbox = BoundingBox(x=norm_x, y=norm_y, w=norm_w, h=norm_h)
            region = OCRTextRegion(
                text=text,
                confidence=conf,
                box=bbox,
                order=idx
            )
            regions.append(region)
            if text:
                text_pieces.append(text)

        full_text = " ".join(text_pieces)

        return OCRResult(
            regions=regions,
            full_text=full_text,
            metadata={
                "engine": "RapidOCR",
                "latency_ms": latency_ms,
                "raw_elapse": elapse or [],
                "regions_count": len(regions)
            }
        )
