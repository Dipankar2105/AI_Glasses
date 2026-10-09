import os, sys, json
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.ai.contracts import OCRTextRegion, BoundingBox
from tests.scripts.evaluate_local_htr import cluster_and_order_regions

def test_line_clustering_ordering():
    # Construct 3 lines of mock text boxes out of order
    # Line 1 (y=0.10): "First", "Line"
    # Line 2 (y=0.25): "Second", "Sentence"
    # Line 3 (y=0.40): "Third", "Entry"
    r1 = OCRTextRegion(text="Line", confidence=0.9, box=BoundingBox(0.30, 0.10, 0.15, 0.05))
    r2 = OCRTextRegion(text="First", confidence=0.95, box=BoundingBox(0.10, 0.10, 0.15, 0.05))
    r3 = OCRTextRegion(text="Sentence", confidence=0.88, box=BoundingBox(0.35, 0.25, 0.20, 0.05))
    r4 = OCRTextRegion(text="Second", confidence=0.92, box=BoundingBox(0.10, 0.25, 0.20, 0.05))
    r5 = OCRTextRegion(text="Entry", confidence=0.85, box=BoundingBox(0.28, 0.40, 0.15, 0.05))
    r6 = OCRTextRegion(text="Third", confidence=0.91, box=BoundingBox(0.10, 0.40, 0.15, 0.05))

    # Pass in shuffled order
    shuffled = [r5, r2, r4, r1, r6, r3]
    ordered_text = cluster_and_order_regions(shuffled)
    expected = "First Line\nSecond Sentence\nThird Entry"
    assert ordered_text == expected, f"Expected:\n{expected}\nGot:\n{ordered_text}"

def test_local_htr_benchmark_artifact_integrity():
    results_path = os.path.join(os.path.dirname(__file__), "../../tests/results/phase4c17-local-htr-evaluation.json")
    assert os.path.exists(results_path), f"Missing {results_path}"

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert "dataset_summaries" in data
    assert "per_sample_evaluations" in data
    evals = data["per_sample_evaluations"]
    assert len(evals) == 30

    for e in evals:
        assert "tesseract_fullpage" in e
        assert "rapidocr_fullpage" in e
        assert "rapidocr_line_clustered" in e
        assert e["rapidocr_line_clustered"]["cer"] >= 0
        assert e["rapidocr_line_clustered"]["mean_latency_ms"] > 0
