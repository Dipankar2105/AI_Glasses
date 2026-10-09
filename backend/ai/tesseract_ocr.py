import numpy as np
from PIL import Image
import pytesseract
from typing import List

from backend.ai.contracts import (
    OCREngine, 
    OCRResult, 
    OCRTextRegion, 
    BoundingBox
)
from backend.vision.frame import OCRInput
from backend.ai.errors import AIVisionError, AIVisionErrorStatus

class TesseractOCREngine(OCREngine):
    def __init__(self):
        super().__init__()
        import os
        cmd = os.environ.get('TESSERACT_CMD')
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

    def process(self, input_data: OCRInput) -> OCRResult:
        if input_data.image is None or input_data.dimensions[0] == 0 or input_data.dimensions[1] == 0:
            raise AIVisionError(AIVisionErrorStatus.INVALID_INPUT, "Dimensions are zero or image is missing")
            
        data = input_data.image
        
        # Verify the dtype and range (must be uint8)
        if data.dtype != np.uint8:
            raise AIVisionError(AIVisionErrorStatus.INVALID_INPUT, f"TesseractOCREngine requires uint8 input, got {data.dtype}")
            
        if input_data.numerical_range != (0, 255):
            raise AIVisionError(AIVisionErrorStatus.INVALID_INPUT, "TesseractOCREngine requires numerical range (0, 255)")
            
        if input_data.channels not in (1, 3):
            raise AIVisionError(AIVisionErrorStatus.INVALID_INPUT, f"Unsupported channels: {input_data.channels}")

        # Convert to PIL Image
        try:
            if input_data.channels == 1:
                img = Image.fromarray(data, mode='L')
            else:
                img = Image.fromarray(data, mode='RGB')
        except Exception as e:
            raise AIVisionError(AIVisionErrorStatus.ENGINE_FAILURE, f"Failed to convert array to PIL Image: {e}")

        # Invoke Tesseract
        try:
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError:
            raise AIVisionError(
                AIVisionErrorStatus.ENGINE_FAILURE, 
                "Tesseract executable not found. Please install tesseract-ocr and ensure it is in your PATH."
            )
        except Exception as e:
            raise AIVisionError(AIVisionErrorStatus.ENGINE_FAILURE, f"Tesseract invocation failed: {e}")

        regions: List[OCRTextRegion] = []
        full_text_parts = []
        width, height = input_data.dimensions
        
        n_boxes = len(ocr_data['level'])
        for i in range(n_boxes):
            text = str(ocr_data['text'][i]).strip()
            conf_str = str(ocr_data['conf'][i])
            
            # Skip empty text and invalid confidences (tesseract sometimes gives '-1')
            if not text or conf_str == '-1':
                continue
                
            try:
                conf = float(conf_str)
            except ValueError:
                continue
                
            # Normalize confidence from [0, 100] to [0.0, 1.0]
            conf_normalized = max(0.0, min(1.0, conf / 100.0))
            
            # Extract coordinates and clamp to image boundaries
            x = max(0, ocr_data['left'][i])
            y = max(0, ocr_data['top'][i])
            w = max(0, ocr_data['width'][i])
            h = max(0, ocr_data['height'][i])
            
            # Convert to normalized bounds [0.0, 1.0]
            nx = float(x) / width
            ny = float(y) / height
            nw = float(w) / width
            nh = float(h) / height
            
            # Ensure safe bounds
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            nw = max(0.0, min(1.0 - nx, nw))
            nh = max(0.0, min(1.0 - ny, nh))
            
            box = BoundingBox(nx, ny, nw, nh)
            region = OCRTextRegion(text, conf_normalized, box, order=i)
            regions.append(region)
            full_text_parts.append(text)
            
        full_text = " ".join(full_text_parts)
        
        metadata = {
            "engine": "pytesseract",
            "timestamp": input_data.timestamp,
            "seq_num": input_data.seq_num
        }
        
        return OCRResult(regions=regions, full_text=full_text, metadata=metadata)
