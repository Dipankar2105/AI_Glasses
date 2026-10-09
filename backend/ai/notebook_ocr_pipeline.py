import os
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

from backend.vision.frame import OCRInput
from backend.ai.contracts import OCRResult, OCRTextRegion, BoundingBox, validate_confidence
from backend.ai.rapid_ocr import RapidOCREngine

class ImageQualityReport:
    """Quality metrics for notebook image inputs."""
    def __init__(
        self,
        is_valid: bool,
        is_blurry: bool,
        blur_score: float,
        contrast_score: float,
        warnings: List[str] = None
    ):
        self.is_valid = is_valid
        self.is_blurry = is_blurry
        self.blur_score = round(float(blur_score), 2)
        self.contrast_score = round(float(contrast_score), 2)
        self.warnings = warnings or []

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "is_blurry": self.is_blurry,
            "blur_score": self.blur_score,
            "contrast_score": self.contrast_score,
            "warnings": self.warnings
        }


class NotebookOCRResult:
    """Structured output for end-to-end notebook OCR processing."""
    def __init__(
        self,
        full_text: str,
        paragraphs: List[str],
        lines: List[str],
        confidence: float,
        regions: List[OCRTextRegion],
        quality_report: ImageQualityReport,
        num_lines: int,
        num_paragraphs: int,
        unreadable_count: int,
        formula_count: int,
        latency_ms: float,
        processing_status: str,
        metadata: Optional[dict] = None
    ):
        self.full_text = full_text
        self.paragraphs = paragraphs
        self.lines = lines
        self.confidence = validate_confidence(confidence)
        self.regions = regions
        self.quality_report = quality_report
        self.num_lines = num_lines
        self.num_paragraphs = num_paragraphs
        self.unreadable_count = unreadable_count
        self.formula_count = formula_count
        self.latency_ms = round(float(latency_ms), 2)
        self.processing_status = processing_status
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "full_text": self.full_text,
            "paragraphs": self.paragraphs,
            "lines": self.lines,
            "confidence": self.confidence,
            "quality_report": self.quality_report.to_dict(),
            "num_lines": self.num_lines,
            "num_paragraphs": self.num_paragraphs,
            "unreadable_count": self.unreadable_count,
            "formula_count": self.formula_count,
            "latency_ms": self.latency_ms,
            "processing_status": self.processing_status,
            "metadata": self.metadata
        }


class NotebookOCRPipeline:
    """
    Experimental end-to-end OCR pipeline specifically tuned for handwritten notebook pages.
    Includes:
      1. Image quality assessment (blur detection via Laplacian variance, contrast evaluation).
      2. Gentle stroke-preserving contrast enhancement (CLAHE on L-channel).
      3. Deep-learning text detection + recognition via RapidOCR ONNX.
      4. Multi-column aware layout clustering and reading-order reconstruction.
      5. Paragraph and heading segmentation.
      6. Quality-aware confidence filtering and unreadable region tagging.
    """
    def __init__(
        self,
        recognizer_engine=None,
        blur_threshold: float = 40.0,
        min_confidence: float = 0.30,
        enable_clahe: bool = True
    ):
        self.recognizer = recognizer_engine if recognizer_engine is not None else RapidOCREngine()
        self.blur_threshold = blur_threshold
        self.min_confidence = min_confidence
        self.enable_clahe = enable_clahe

    def assess_quality(self, image: np.ndarray) -> ImageQualityReport:
        """Evaluates image validity, blurriness, and contrast."""
        warnings = []
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return ImageQualityReport(
                is_valid=False,
                is_blurry=True,
                blur_score=0.0,
                contrast_score=0.0,
                warnings=["Invalid or empty image array"]
            )

        if len(image.shape) == 2:
            gray = image
        elif len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            return ImageQualityReport(
                is_valid=False,
                is_blurry=True,
                blur_score=0.0,
                contrast_score=0.0,
                warnings=[f"Unsupported image dimensions: {image.shape}"]
            )

        # Blur estimation: Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = laplacian_var < self.blur_threshold
        if is_blurry:
            warnings.append(f"Image is potentially blurry (Laplacian var {laplacian_var:.1f} < {self.blur_threshold:.1f})")

        # Contrast estimation: standard deviation of luminance
        contrast_std = float(np.std(gray))
        if contrast_std < 25.0:
            warnings.append(f"Low contrast detected (std {contrast_std:.1f} < 25.0)")

        return ImageQualityReport(
            is_valid=True,
            is_blurry=is_blurry,
            blur_score=laplacian_var,
            contrast_score=contrast_std,
            warnings=warnings
        )

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Gentle contrast enhancement using CLAHE in LAB color space."""
        if not self.enable_clahe or image is None or image.size == 0:
            return image

        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            enhanced_lab = cv2.merge((cl, a_channel, b_channel))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        elif len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            return clahe.apply(image)
        return image

    def _detect_columns(self, boxes: List[dict], img_w: float = 1.0) -> List[List[dict]]:
        """
        Partitions boxes into columns if a distinct multi-column layout is detected.
        Uses horizontal center distribution gap analysis.
        """
        if len(boxes) < 4:
            return [boxes]

        # Analyze horizontal centers
        x_centers = sorted([b['xc'] for b in boxes])
        # Find maximum horizontal gap in middle 60% of page
        valid_gaps = []
        for i in range(len(x_centers) - 1):
            x1, x2 = x_centers[i], x_centers[i+1]
            gap = x2 - x1
            mid = (x1 + x2) / 2.0
            if 0.25 <= mid <= 0.75 and gap > 0.15:
                valid_gaps.append((gap, mid))

        if valid_gaps:
            valid_gaps.sort(key=lambda x: x[0], reverse=True)
            split_x = valid_gaps[0][1]
            col_left = [b for b in boxes if b['xc'] < split_x]
            col_right = [b for b in boxes if b['xc'] >= split_x]
            if len(col_left) >= 2 and len(col_right) >= 2:
                return [col_left, col_right]

        return [boxes]

    def reconstruct_layout(
        self,
        regions: List[OCRTextRegion],
        line_overlap_ratio: float = 0.55,
        para_gap_ratio: float = 1.6
    ) -> Tuple[str, List[str], List[str], int, int]:
        """
        Reconstructs reading order, lines, and paragraphs from detected OCR regions.
        Handles:
          - Line clustering by vertical overlap.
          - Multi-column separation.
          - Paragraph grouping.
          - Low-confidence tagging.
          - Math formula heuristic flagging.
        """
        if not regions:
            return "", [], [], 0, 0

        raw_boxes = []
        unreadable_count = 0
        formula_count = 0

        for r in regions:
            text = r.text.strip()
            if not text:
                continue

            # Check low confidence
            if r.confidence < self.min_confidence:
                unreadable_count += 1
                text = f"[Unreadable: {text}]"

            # Check formula / math heuristic
            math_symbols = {'=', '+', '-', '*', '/', '^', '\\sqrt', '\\int', '\\Sigma', '<', '>', '\u00b1'}
            symbol_hits = sum(1 for ch in text if ch in math_symbols)
            if symbol_hits >= 2 and any(ch.isdigit() for ch in text):
                formula_count += 1
                text = f"[Formula: {text}]"

            xmin = r.box.x
            ymin = r.box.y
            xmax = r.box.x + r.box.width
            ymax = r.box.y + r.box.height
            xc = xmin + r.box.width / 2.0
            yc = ymin + r.box.height / 2.0
            h = max(r.box.height, 0.001)

            raw_boxes.append({
                'text': text,
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax,
                'xc': xc,
                'yc': yc,
                'h': h,
                'conf': r.confidence
            })

        if not raw_boxes:
            return "", [], [], unreadable_count, formula_count

        columns = self._detect_columns(raw_boxes)
        all_ordered_lines = []
        all_paragraphs = []

        for col_boxes in columns:
            col_boxes.sort(key=lambda b: b['yc'])
            
            # Cluster boxes into lines
            lines_in_col = []
            curr_line = [col_boxes[0]]
            curr_yc = col_boxes[0]['yc']
            curr_h = col_boxes[0]['h']

            for b in col_boxes[1:]:
                if abs(b['yc'] - curr_yc) < (curr_h * line_overlap_ratio):
                    curr_line.append(b)
                    curr_yc = np.mean([x['yc'] for x in curr_line])
                    curr_h = np.mean([x['h'] for x in curr_line])
                else:
                    curr_line.sort(key=lambda x: x['xmin'])
                    line_str = " ".join([x['text'] for x in curr_line])
                    line_ymin = min(x['ymin'] for x in curr_line)
                    line_ymax = max(x['ymax'] for x in curr_line)
                    lines_in_col.append({
                        'text': line_str,
                        'ymin': line_ymin,
                        'ymax': line_ymax,
                        'h': line_ymax - line_ymin
                    })
                    curr_line = [b]
                    curr_yc = b['yc']
                    curr_h = b['h']

            if curr_line:
                curr_line.sort(key=lambda x: x['xmin'])
                line_str = " ".join([x['text'] for x in curr_line])
                line_ymin = min(x['ymin'] for x in curr_line)
                line_ymax = max(x['ymax'] for x in curr_line)
                lines_in_col.append({
                    'text': line_str,
                    'ymin': line_ymin,
                    'ymax': line_ymax,
                    'h': line_ymax - line_ymin
                })

            for l in lines_in_col:
                all_ordered_lines.append(l['text'])

            # Group lines into paragraphs based on vertical spacing
            if lines_in_col:
                median_line_h = np.median([l['h'] for l in lines_in_col])
                para_threshold = max(median_line_h * para_gap_ratio, 0.02)
                
                curr_para = [lines_in_col[0]['text']]
                prev_ymax = lines_in_col[0]['ymax']

                for l in lines_in_col[1:]:
                    vertical_gap = l['ymin'] - prev_ymax
                    if vertical_gap > para_threshold:
                        all_paragraphs.append("\n".join(curr_para))
                        curr_para = [l['text']]
                    else:
                        curr_para.append(l['text'])
                    prev_ymax = l['ymax']

                if curr_para:
                    all_paragraphs.append("\n".join(curr_para))

        full_text = "\n\n".join(all_paragraphs)
        return full_text, all_paragraphs, all_ordered_lines, unreadable_count, formula_count

    def process(self, ocr_input: OCRInput) -> NotebookOCRResult:
        """Runs the full notebook OCR pipeline synchronously."""
        t0 = time.perf_counter()

        if ocr_input is None or ocr_input.image is None:
            quality = ImageQualityReport(
                is_valid=False,
                is_blurry=True,
                blur_score=0.0,
                contrast_score=0.0,
                warnings=["Null OCRInput or empty image buffer"]
            )
            return NotebookOCRResult(
                full_text="",
                paragraphs=[],
                lines=[],
                confidence=0.0,
                regions=[],
                quality_report=quality,
                num_lines=0,
                num_paragraphs=0,
                unreadable_count=0,
                formula_count=0,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                processing_status="ERROR_INVALID_INPUT"
            )

        # 1. Quality Assessment
        quality = self.assess_quality(ocr_input.image)
        if not quality.is_valid:
            return NotebookOCRResult(
                full_text="",
                paragraphs=[],
                lines=[],
                confidence=0.0,
                regions=[],
                quality_report=quality,
                num_lines=0,
                num_paragraphs=0,
                unreadable_count=0,
                formula_count=0,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                processing_status="ERROR_INVALID_IMAGE"
            )

        # 2. Preprocessing
        enhanced_image = self.preprocess_image(ocr_input.image)
        w, h = ocr_input.dimensions if hasattr(ocr_input, "dimensions") else (ocr_input.image.shape[1], ocr_input.image.shape[0])
        enhanced_input = OCRInput(
            image=enhanced_image,
            width=w,
            height=h,
            channels=ocr_input.channels if hasattr(ocr_input, "channels") else 3,
            numerical_range=ocr_input.numerical_range if hasattr(ocr_input, "numerical_range") else (0, 255),
            timestamp=ocr_input.timestamp if hasattr(ocr_input, "timestamp") else 0,
            seq_num=ocr_input.seq_num if hasattr(ocr_input, "seq_num") else 0
        )

        # 3. Recognition
        raw_ocr_result: OCRResult = self.recognizer.process(enhanced_input)

        # 4. Text & Layout Reconstruction
        full_text, paragraphs, lines, unreadable_cnt, formula_cnt = self.reconstruct_layout(
            raw_ocr_result.regions
        )

        avg_conf = 0.0
        if raw_ocr_result.regions:
            avg_conf = float(np.mean([r.confidence for r in raw_ocr_result.regions]))
            avg_conf = max(0.0, min(1.0, avg_conf))

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        status = "SUCCESS" if lines else "EMPTY_OUTPUT"

        return NotebookOCRResult(
            full_text=full_text,
            paragraphs=paragraphs,
            lines=lines,
            confidence=avg_conf,
            regions=raw_ocr_result.regions,
            quality_report=quality,
            num_lines=len(lines),
            num_paragraphs=len(paragraphs),
            unreadable_count=unreadable_cnt,
            formula_count=formula_cnt,
            latency_ms=latency_ms,
            processing_status=status,
            metadata={
                "engine": self.recognizer.__class__.__name__,
                "input_dimensions": [w, h],
                "clahe_applied": self.enable_clahe
            }
        )
