import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ai.notebook_ocr_pipeline import NotebookOCRPipeline, ImageQualityReport, NotebookOCRResult
from backend.ai.contracts import OCRTextRegion, BoundingBox, OCRResult
from backend.vision.frame import OCRInput
from backend.ai.ocr_benchmark import calculate_cer, calculate_wer, normalize_text


class DummyMockRecognizer:
    def __init__(self, text_items):
        self.text_items = text_items

    def process(self, inp: OCRInput) -> OCRResult:
        regions = []
        for idx, (t, conf, x, y, w, h) in enumerate(self.text_items):
            regions.append(OCRTextRegion(
                text=t,
                confidence=conf,
                box=BoundingBox(x=x, y=y, w=w, h=h),
                order=idx
            ))
        full = " ".join([r.text for r in regions])
        return OCRResult(regions=regions, full_text=full)


def test_candidate_pipeline_cjk_filtering():
    # Mock recognizer returning English words interleaved with hallucinated CJK characters
    mock = DummyMockRecognizer([
        ("Introduction", 0.95, 0.1, 0.1, 0.2, 0.05),
        ("to", 0.90, 0.32, 0.1, 0.05, 0.05),
        ("个", 0.30, 0.38, 0.1, 0.02, 0.05),  # Hallucinated CJK radical
        ("Data", 0.92, 0.42, 0.1, 0.1, 0.05),
        ("Structures", 0.91, 0.54, 0.1, 0.2, 0.05),
        ("］", 0.25, 0.75, 0.1, 0.02, 0.05),  # Fullwidth bracket
    ])

    pipe = NotebookOCRPipeline(recognizer_engine=mock, filter_cjk=True, tag_anomalies=False)
    dummy_img = np.full((100, 100, 3), 200, dtype=np.uint8)
    inp = OCRInput(image=dummy_img, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)
    res = pipe.process(inp)

    assert "Introduction to Data Structures" in res.full_text
    assert "个" not in res.full_text
    assert "］" not in res.full_text


def test_candidate_pipeline_no_tag_pollution():
    # When tag_anomalies=False, unreadable or formula text is NOT wrapped in literal string tags
    mock = DummyMockRecognizer([
        ("low_conf_word", 0.20, 0.1, 0.1, 0.2, 0.05),
        ("a = 2 + b / 3", 0.85, 0.1, 0.2, 0.3, 0.05)
    ])

    pipe = NotebookOCRPipeline(recognizer_engine=mock, tag_anomalies=False, min_confidence=0.30)
    dummy_img = np.full((100, 100, 3), 200, dtype=np.uint8)
    inp = OCRInput(image=dummy_img, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)
    res = pipe.process(inp)

    # Anomaly counts tracked in structured metadata
    assert res.unreadable_count == 1
    assert res.formula_count == 1
    # Literal tag strings NOT injected into text
    assert "[Unreadable:" not in res.full_text
    assert "[Formula:" not in res.full_text
    assert "low_conf_word" in res.full_text
    assert "a = 2 + b / 3" in res.full_text


def test_candidate_benchmark_artifact_integrity():
    artifact_path = "tests/results/phase4c20-candidate-notebook-ocr.json"
    assert os.path.exists(artifact_path), f"Candidate artifact {artifact_path} does not exist"

    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["provenance_summary"]["sample_count"] == 30
    assert data["provenance_summary"]["dev_samples"] == 15
    assert data["provenance_summary"]["heldout_samples"] == 15
    assert len(data["per_sample_evaluations"]) == 30

    dev_summary = data["dataset_summaries"]["dev_split"]
    heldout_summary = data["dataset_summaries"]["heldout_split"]
    full_summary = data["dataset_summaries"]["full_dataset"]

    # Verify arithmetic: macro CER must match mean of per-sample CERs
    dev_cers = [s["cer"] for s in data["per_sample_evaluations"] if s["split"] == "Dev"]
    assert pytest.approx(dev_summary["macro_cer"], rel=1e-3) == float(np.mean(dev_cers))

    # Verify latency improvement: mean latency < 5000ms
    assert full_summary["mean_latency_ms"] < 5000.0
