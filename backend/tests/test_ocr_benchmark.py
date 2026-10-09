import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.ai.ocr_benchmark import (
    calculate_cer,
    calculate_wer,
    normalize_text,
    levenshtein_distance,
    generate_multi_type_benchmark,
    evaluate_engine_on_benchmark,
    BenchmarkCategory,
    SampleProvenance
)
from backend.ai.mock_engines import MockOCREngine
from backend.ai.tesseract_ocr import TesseractOCREngine
import pytesseract


def test_normalize_text():
    assert normalize_text("  Hello   World \n \t ") == "hello world"
    assert normalize_text("Hello\r\nWorld", case_sensitive=True) == "Hello World"
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_levenshtein_distance():
    assert levenshtein_distance([], []) == 0
    assert levenshtein_distance(["a"], []) == 1
    assert levenshtein_distance([], ["b"]) == 1
    assert levenshtein_distance(list("kitten"), list("sitting")) == 3


def test_calculate_cer():
    # Exact match
    assert calculate_cer("HELLO", "hello") == 0.0
    assert calculate_cer("HELLO", "HELLO", case_sensitive=True) == 0.0
    
    # Substitutions / Insertions / Deletions
    assert np.isclose(calculate_cer("cat", "hat"), 1.0 / 3.0)
    assert np.isclose(calculate_cer("cat", "cats"), 1.0 / 3.0)
    assert np.isclose(calculate_cer("cat", "ca"), 1.0 / 3.0)
    
    # Empty reference vs prediction
    assert calculate_cer("", "") == 0.0
    assert calculate_cer("", "text") == 1.0
    assert calculate_cer("text", "") == 1.0


def test_calculate_wer():
    # Exact match
    assert calculate_wer("hello world", "Hello World") == 0.0
    
    # 1 word substitution out of 4 words -> 0.25
    assert np.isclose(calculate_wer("the quick brown fox", "the fast brown fox"), 0.25)
    
    # Empty cases
    assert calculate_wer("", "") == 0.0
    assert calculate_wer("", "word") == 1.0
    assert calculate_wer("word", "") == 1.0


def test_benchmark_dataset_structure():
    samples = generate_multi_type_benchmark()
    assert len(samples) >= 15
    
    categories_present = {s.category for s in samples}
    assert BenchmarkCategory.PRINTED_TEXT in categories_present
    assert BenchmarkCategory.HANDWRITTEN_PAPER in categories_present
    assert BenchmarkCategory.DIGITAL_BOARD in categories_present
    assert BenchmarkCategory.CLASSROOM_BOARD in categories_present
    assert BenchmarkCategory.MATHEMATICS in categories_present
    
    for s in samples:
        assert len(s.expected_text) > 0
        assert s.image_array.dtype == np.uint8
        assert s.image_array.ndim in (2, 3)
        assert s.dimensions[0] > 0 and s.dimensions[1] > 0
        assert s.channels in (1, 3)
        assert isinstance(s.provenance, SampleProvenance)


def test_evaluate_engine_with_mock():
    mock_engine = MockOCREngine()
    samples = generate_multi_type_benchmark()
    summary = evaluate_engine_on_benchmark(mock_engine, samples[:3])
    
    assert summary["total_samples"] == 3
    assert "mean_cer" in summary
    assert "mean_wer" in summary
    assert "category_breakdown" in summary
    assert len(summary["evaluations"]) == 3


def has_tesseract():
    try:
        _ = TesseractOCREngine()
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not has_tesseract(), reason="Tesseract executable not found")
def test_evaluate_engine_with_tesseract():
    tess_engine = TesseractOCREngine()
    samples = generate_multi_type_benchmark()
    
    # Test on a small subset of benchmark fixtures
    subset = [s for s in samples if s.sample_id in ("A1_printed_simple", "D2_blackboard_chalk")]
    summary = evaluate_engine_on_benchmark(tess_engine, subset)
    
    assert summary["total_samples"] == 2
    assert summary["mean_cer"] < 0.20
    assert summary["evaluations"][0].sample.sample_id == "A1_printed_simple"
    assert summary["evaluations"][1].sample.sample_id == "D2_blackboard_chalk"
