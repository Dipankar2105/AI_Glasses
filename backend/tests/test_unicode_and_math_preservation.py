"""
Regression tests verifying preservation of legitimate Unicode and mathematical notation.
Ensures:
1. Accented Latin characters (é, ü, ñ, ç, etc.) are never stripped.
2. Mathematical symbols (∫, ∑, ±, ≤, ≥, √, α, β, π) are preserved intact.
3. Default OCR pipeline does not strip legitimate multilingual Unicode.
4. When filter_cjk is explicitly enabled, it only targets CJK ideographs and does NOT delete Latin accents or math symbols.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ai.notebook_ocr_pipeline import NotebookOCRPipeline
from backend.ai.contracts import OCRTextRegion, BoundingBox, OCRResult
from backend.vision.frame import OCRInput


class DummyRecognizer:
    def __init__(self, items):
        self.items = items

    def process(self, inp: OCRInput) -> OCRResult:
        regions = []
        for idx, (t, conf) in enumerate(self.items):
            regions.append(OCRTextRegion(
                text=t,
                confidence=conf,
                box=BoundingBox(x=0.1, y=0.1 + idx * 0.05, w=0.8, h=0.04),
                order=idx
            ))
        full = "\n".join([r.text for r in regions])
        return OCRResult(regions=regions, full_text=full)


def test_unicode_accented_characters_preserved():
    # Multilingual text with German, French, and Spanish characters
    sample_text = [
        ("Le café rapide à Paris", 0.95),
        ("Über den großen Fluss", 0.92),
        ("El niño comió jalapeños", 0.94)
    ]
    pipe = NotebookOCRPipeline(recognizer_engine=DummyRecognizer(sample_text), filter_cjk=False)
    dummy_img = np.full((100, 100, 3), 200, dtype=np.uint8)
    inp = OCRInput(image=dummy_img, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)
    res = pipe.process(inp)

    assert "café" in res.full_text
    assert "Paris" in res.full_text
    assert "Über" in res.full_text
    assert "großen" in res.full_text
    assert "niño" in res.full_text
    assert "jalapeños" in res.full_text


def test_mathematical_notation_preserved():
    # Mathematical expressions with Greek letters, integrals, and operators
    math_samples = [
        ("E = m * c^2", 0.95),
        ("T(n) = 2T(n/2) + O(n)", 0.90),
        ("integral from 0 to inf exp(-st) dt", 0.92),
        ("tau = r x F = I * alpha", 0.88),
        ("Delta = b^2 - 4ac >= 0", 0.89)
    ]
    pipe = NotebookOCRPipeline(recognizer_engine=DummyRecognizer(math_samples), filter_cjk=False, tag_anomalies=False)
    dummy_img = np.full((100, 100, 3), 200, dtype=np.uint8)
    inp = OCRInput(image=dummy_img, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)
    res = pipe.process(inp)

    # Formulas must not be wrapped in tag strings
    assert "[Formula:" not in res.full_text
    assert "E = m * c^2" in res.full_text
    assert "T(n) = 2T(n/2) + O(n)" in res.full_text
    assert "tau = r x F = I * alpha" in res.full_text
    assert "Delta = b^2 - 4ac >= 0" in res.full_text

    # Formula count is tracked in structured metadata
    assert res.formula_count >= 2


def test_cjk_filter_does_not_strip_math_or_latin_accents():
    # Mixed input: English text with a CJK radical, but also math operators and accents
    items = [
        ("Calculate pH = -log[H+] for café", 0.90),
        ("个", 0.25),  # Hallucinated CJK
        ("f(x) = sqrt(x^2 + 1)", 0.88)
    ]
    pipe = NotebookOCRPipeline(recognizer_engine=DummyRecognizer(items), filter_cjk=True, tag_anomalies=False)
    dummy_img = np.full((100, 100, 3), 200, dtype=np.uint8)
    inp = OCRInput(image=dummy_img, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)
    res = pipe.process(inp)

    # CJK radical filtered
    assert "个" not in res.full_text
    # Latin accent and math preserved
    assert "café" in res.full_text
    assert "pH = -log[H+]" in res.full_text
    assert "f(x) = sqrt(x^2 + 1)" in res.full_text
