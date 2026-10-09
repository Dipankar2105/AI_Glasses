import os
import sys
import json
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.vision.frame import OCRInput
from backend.ai.contracts import OCRTextRegion, BoundingBox, OCRResult
from backend.ai.notebook_ocr_pipeline import NotebookOCRPipeline, ImageQualityReport, NotebookOCRResult

class MockRecognizer:
    def __init__(self, mock_regions=None):
        self.mock_regions = mock_regions or []

    def process(self, ocr_input: OCRInput) -> OCRResult:
        full_text = " ".join([r.text for r in self.mock_regions])
        return OCRResult(regions=self.mock_regions, full_text=full_text)


def test_quality_assessment_valid_and_invalid():
    pipeline = NotebookOCRPipeline(recognizer_engine=MockRecognizer(), blur_threshold=40.0)

    # Valid sharp synthetic image
    sharp_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    report_sharp = pipeline.assess_quality(sharp_img)
    assert report_sharp.is_valid is True
    assert report_sharp.blur_score > 0

    # Smooth/blurry uniform image
    blurry_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    report_blurry = pipeline.assess_quality(blurry_img)
    assert report_blurry.is_valid is True
    assert report_blurry.is_blurry is True
    assert report_blurry.blur_score < 40.0

    # Invalid empty input
    report_invalid = pipeline.assess_quality(np.zeros((0, 0, 3), dtype=np.uint8))
    assert report_invalid.is_valid is False


def test_layout_reconstruction_and_ordering():
    pipeline = NotebookOCRPipeline(recognizer_engine=MockRecognizer())

    # Create 3 synthetic text regions: 2 on line 1, 1 on line 2
    r1 = OCRTextRegion("Hello", 0.95, BoundingBox(0.1, 0.1, 0.1, 0.05), order=1)
    r2 = OCRTextRegion("World", 0.90, BoundingBox(0.25, 0.105, 0.1, 0.05), order=2)
    r3 = OCRTextRegion("Second line", 0.85, BoundingBox(0.1, 0.25, 0.2, 0.05), order=3)

    full_text, paragraphs, lines, unreadable, formulas = pipeline.reconstruct_layout([r2, r3, r1])

    assert len(lines) == 2
    assert lines[0] == "Hello World"
    assert lines[1] == "Second line"
    assert unreadable == 0
    assert formulas == 0


def test_low_confidence_and_formula_tagging():
    pipeline = NotebookOCRPipeline(recognizer_engine=MockRecognizer(), min_confidence=0.40)

    r_low = OCRTextRegion("scribble", 0.20, BoundingBox(0.1, 0.1, 0.1, 0.05))
    r_math = OCRTextRegion("x = 2 + y / 4", 0.85, BoundingBox(0.1, 0.2, 0.2, 0.05))

    full_text, paragraphs, lines, unreadable, formulas = pipeline.reconstruct_layout([r_low, r_math])

    assert unreadable == 1
    assert formulas == 1
    assert "[Unreadable: scribble]" in full_text
    assert "[Formula: x = 2 + y / 4]" in full_text


def test_end_to_end_pipeline_processing():
    r1 = OCRTextRegion("Notebook Title", 0.95, BoundingBox(0.1, 0.05, 0.3, 0.04))
    r2 = OCRTextRegion("Section 1 content", 0.90, BoundingBox(0.1, 0.15, 0.4, 0.04))
    mock_engine = MockRecognizer([r1, r2])

    pipeline = NotebookOCRPipeline(recognizer_engine=mock_engine, enable_clahe=True)

    dummy_img = np.full((300, 300, 3), 200, dtype=np.uint8)
    ocr_input = OCRInput(
        image=dummy_img,
        width=300,
        height=300,
        channels=3,
        numerical_range=(0, 255),
        timestamp=1000,
        seq_num=1
    )

    result = pipeline.process(ocr_input)

    assert isinstance(result, NotebookOCRResult)
    assert result.processing_status == "SUCCESS"
    assert result.num_lines == 2
    assert "Notebook Title" in result.full_text
    assert result.latency_ms > 0.0
    assert result.quality_report.is_valid is True


def test_invalid_input_handling():
    pipeline = NotebookOCRPipeline(recognizer_engine=MockRecognizer())
    res = pipeline.process(None)
    assert res.processing_status == "ERROR_INVALID_INPUT"
    assert res.full_text == ""


def test_phase4c19_result_schema():
    result_path = "tests/results/phase4c19-notebook-ocr-pipeline.json"
    if not os.path.exists(result_path):
        pytest.skip(f"Result file {result_path} not found")

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "benchmark_name" in data
    assert "pipeline_architecture" in data
    assert "dataset_summaries" in data
    assert "per_sample_evaluations" in data

    summaries = data["dataset_summaries"]
    for split in ["full_dataset", "dev_split", "heldout_split"]:
        assert split in summaries
        assert "tesseract_baseline" in summaries[split]
        assert "rapidocr_baseline" in summaries[split]
        assert "notebook_ocr_pipeline" in summaries[split]

