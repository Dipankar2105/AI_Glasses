import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Dict, Tuple, Any, Optional
from enum import Enum

from backend.vision.frame import OCRInput
from backend.ai.contracts import OCREngine, OCRResult


class BenchmarkCategory(Enum):
    PRINTED_TEXT = "A. Printed Text"
    HANDWRITTEN_PAPER = "B. Handwritten Text (Paper)"
    DIGITAL_BOARD = "C. Digital Boards & Slides"
    CLASSROOM_BOARD = "D. Classroom Boards (Whiteboard/Blackboard)"
    MATHEMATICS = "E. Mathematics & Classroom Notes"


class SampleProvenance(Enum):
    SYNTHETIC_PRINTED = "Synthetic Printed"
    SYNTHETIC_HANDWRITING = "Synthetic Handwriting Style"
    SYNTHETIC_BOARD = "Synthetic Board Render"
    REAL_SAMPLE = "Real Sample"


class BenchmarkSample:
    def __init__(
        self,
        sample_id: str,
        category: BenchmarkCategory,
        provenance: SampleProvenance,
        expected_text: str,
        image_array: np.ndarray,
        description: str = ""
    ):
        self.sample_id = sample_id
        self.category = category
        self.provenance = provenance
        self.expected_text = expected_text
        self.image_array = image_array
        self.description = description

    @property
    def dimensions(self) -> Tuple[int, int]:
        h, w = self.image_array.shape[:2]
        return (w, h)

    @property
    def channels(self) -> int:
        if len(self.image_array.shape) == 2:
            return 1
        return self.image_array.shape[2]


class SampleEvaluation:
    def __init__(
        self,
        sample: BenchmarkSample,
        recognized_text: str,
        cer: float,
        wer: float,
        exact_match: bool,
        confidence: float,
        latency_ms: float,
        regions_count: int
    ):
        self.sample = sample
        self.recognized_text = recognized_text
        self.cer = cer
        self.wer = wer
        self.exact_match = exact_match
        self.confidence = confidence
        self.latency_ms = latency_ms
        self.regions_count = regions_count


def normalize_text(text: str, case_sensitive: bool = False) -> str:
    """Normalize whitespace, newlines, and optionally case."""
    if not text:
        return ""
    # Replace newlines and tabs with space
    cleaned = re.sub(r'[\r\n\t]+', ' ', text)
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not case_sensitive:
        cleaned = cleaned.lower()
    return cleaned


def levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> int:
    """Standard dynamic programming Levenshtein distance."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
        
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )
    return dp[n][m]


def calculate_cer(reference: str, hypothesis: str, case_sensitive: bool = False) -> float:
    """Calculate Character Error Rate (CER)."""
    norm_ref = normalize_text(reference, case_sensitive=case_sensitive)
    norm_hyp = normalize_text(hypothesis, case_sensitive=case_sensitive)
    
    if len(norm_ref) == 0:
        return 0.0 if len(norm_hyp) == 0 else 1.0
        
    dist = levenshtein_distance(list(norm_ref), list(norm_hyp))
    return float(dist) / float(len(norm_ref))


def calculate_wer(reference: str, hypothesis: str, case_sensitive: bool = False) -> float:
    """Calculate Word Error Rate (WER)."""
    norm_ref = normalize_text(reference, case_sensitive=case_sensitive)
    norm_hyp = normalize_text(hypothesis, case_sensitive=case_sensitive)
    
    ref_words = norm_ref.split() if norm_ref else []
    hyp_words = norm_hyp.split() if norm_hyp else []
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
        
    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist) / float(len(ref_words))


def generate_multi_type_benchmark() -> List[BenchmarkSample]:
    """Generates the fixed 17-sample multi-type OCR benchmark dataset."""
    samples: List[BenchmarkSample] = []
    
    # ---------------------------------------------------------
    # CATEGORY A: PRINTED TEXT
    # ---------------------------------------------------------
    # A1. Simple Uppercase
    im_a1 = Image.new('L', (300, 60), color=255)
    d = ImageDraw.Draw(im_a1)
    d.text((20, 20), "HELLO WORLD", fill=0)
    samples.append(BenchmarkSample(
        "A1_printed_simple",
        BenchmarkCategory.PRINTED_TEXT,
        SampleProvenance.SYNTHETIC_PRINTED,
        "HELLO WORLD",
        np.array(im_a1),
        "Clean high-contrast printed uppercase"
    ))
    
    # A2. Mixed Case & Year
    im_a2 = Image.new('RGB', (400, 60), color=(255, 255, 255))
    d = ImageDraw.Draw(im_a2)
    d.text((20, 20), "Optical Character Recognition 2026", fill=(0, 0, 0))
    samples.append(BenchmarkSample(
        "A2_printed_mixed_case",
        BenchmarkCategory.PRINTED_TEXT,
        SampleProvenance.SYNTHETIC_PRINTED,
        "Optical Character Recognition 2026",
        np.array(im_a2),
        "Clean mixed-case sentence with numbers"
    ))

    # A3. Multiline Document Paragraph
    im_a3 = Image.new('L', (380, 100), color=255)
    d = ImageDraw.Draw(im_a3)
    lines_a3 = ["AI Glasses Vision Pipeline", "Real Time Host Processing", "Frame Sequence 101"]
    for i, line in enumerate(lines_a3):
        d.text((15, 15 + i * 25), line, fill=0)
    samples.append(BenchmarkSample(
        "A3_printed_multiline",
        BenchmarkCategory.PRINTED_TEXT,
        SampleProvenance.SYNTHETIC_PRINTED,
        "AI Glasses Vision Pipeline Real Time Host Processing Frame Sequence 101",
        np.array(im_a3),
        "Multi-line structured printed text"
    ))

    # A4. Low Contrast Printed
    im_a4 = Image.new('RGB', (350, 60), color=(200, 200, 200))
    d = ImageDraw.Draw(im_a4)
    d.text((20, 20), "Low Contrast Document", fill=(140, 140, 140))
    samples.append(BenchmarkSample(
        "A4_printed_low_contrast",
        BenchmarkCategory.PRINTED_TEXT,
        SampleProvenance.SYNTHETIC_PRINTED,
        "Low Contrast Document",
        np.array(im_a4),
        "Low contrast gray-on-gray printed text"
    ))

    # ---------------------------------------------------------
    # CATEGORY B: HANDWRITTEN TEXT ON PAPER
    # (Labelled strictly as Synthetic Handwriting Style)
    # ---------------------------------------------------------
    # B1. Simulated Italic/Cursive Hand Note
    im_b1 = Image.new('RGB', (420, 70), color=(250, 248, 240)) # Paper tint
    d = ImageDraw.Draw(im_b1)
    d.text((20, 25), "Welcome to AI Glasses class", fill=(20, 30, 80)) # Blue ink
    samples.append(BenchmarkSample(
        "B1_handwritten_note",
        BenchmarkCategory.HANDWRITTEN_PAPER,
        SampleProvenance.SYNTHETIC_HANDWRITING,
        "Welcome to AI Glasses class",
        np.array(im_b1),
        "Simulated casual hand note with paper tint"
    ))

    # B2. Casual Numbered Notes
    im_b2 = Image.new('L', (350, 60), color=250)
    d = ImageDraw.Draw(im_b2)
    d.text((20, 20), "Room 302 at 4:30 PM", fill=30)
    samples.append(BenchmarkSample(
        "B2_handwritten_numbers",
        BenchmarkCategory.HANDWRITTEN_PAPER,
        SampleProvenance.SYNTHETIC_HANDWRITING,
        "Room 302 at 4:30 PM",
        np.array(im_b2),
        "Simulated handwritten room and time notes"
    ))

    # B3. Uneven Hand Note with Mild Blur
    im_b3 = Image.new('L', (380, 60), color=255)
    d = ImageDraw.Draw(im_b3)
    d.text((15, 20), "Meeting on Friday morning", fill=0)
    im_b3 = im_b3.filter(ImageFilter.GaussianBlur(radius=0.6))
    samples.append(BenchmarkSample(
        "B3_handwritten_blurred",
        BenchmarkCategory.HANDWRITTEN_PAPER,
        SampleProvenance.SYNTHETIC_HANDWRITING,
        "Meeting on Friday morning",
        np.array(im_b3),
        "Simulated hand note with mild optical blur"
    ))

    # ---------------------------------------------------------
    # CATEGORY C: DIGITAL BOARDS & PROJECTED SLIDES
    # ---------------------------------------------------------
    # C1. Slide Presentation (Inverted Dark Background)
    im_c1 = Image.new('RGB', (420, 100), color=(20, 30, 50)) # Navy slide background
    d = ImageDraw.Draw(im_c1)
    d.text((20, 15), "EDGE AI ARCHITECTURE", fill=(240, 240, 255))
    d.text((20, 45), "- Low Latency Processing", fill=(200, 220, 240))
    d.text((20, 70), "- Host Side OCR Engine", fill=(200, 220, 240))
    samples.append(BenchmarkSample(
        "C1_digital_slide_dark",
        BenchmarkCategory.DIGITAL_BOARD,
        SampleProvenance.SYNTHETIC_BOARD,
        "EDGE AI ARCHITECTURE Low Latency Processing Host Side OCR Engine",
        np.array(im_c1),
        "Dark theme projected presentation slide"
    ))

    # C2. Slide with Localized Glare / Brightness Gradient
    im_c2 = Image.new('RGB', (400, 70), color=(240, 245, 255))
    grad = np.tile(np.linspace(30, 0, 400, dtype=np.uint8), (70, 1))
    arr_c2 = np.array(im_c2)
    arr_c2[..., 0] = np.clip(arr_c2[..., 0].astype(int) + grad, 0, 255).astype(np.uint8)
    im_c2_glare = Image.fromarray(arr_c2)
    d = ImageDraw.Draw(im_c2_glare)
    d.text((25, 25), "PROJECTOR DISPLAY TEST", fill=(40, 50, 70))
    samples.append(BenchmarkSample(
        "C2_digital_slide_glare",
        BenchmarkCategory.DIGITAL_BOARD,
        SampleProvenance.SYNTHETIC_BOARD,
        "PROJECTOR DISPLAY TEST",
        np.array(im_c2_glare),
        "Projected slide with simulated lighting gradient/glare"
    ))

    # ---------------------------------------------------------
    # CATEGORY D: CLASSROOM BOARDS (WHITEBOARD / BLACKBOARD)
    # ---------------------------------------------------------
    # D1. Whiteboard Marker (Blue on Whiteboard)
    im_d1 = Image.new('RGB', (420, 70), color=(245, 245, 242)) # Whiteboard gloss
    d = ImageDraw.Draw(im_d1)
    d.text((20, 25), "Assignment Due Next Tuesday", fill=(10, 50, 180)) # Dry erase blue
    samples.append(BenchmarkSample(
        "D1_whiteboard_marker",
        BenchmarkCategory.CLASSROOM_BOARD,
        SampleProvenance.SYNTHETIC_BOARD,
        "Assignment Due Next Tuesday",
        np.array(im_d1),
        "Dry-erase blue marker on whiteboard"
    ))

    # D2. Blackboard Chalk (White on Dark Slate)
    im_d2 = Image.new('RGB', (420, 70), color=(40, 48, 45)) # Dark chalkboard
    d = ImageDraw.Draw(im_d2)
    d.text((20, 25), "Physics 101: Newton Laws", fill=(230, 235, 230)) # Chalk white
    samples.append(BenchmarkSample(
        "D2_blackboard_chalk",
        BenchmarkCategory.CLASSROOM_BOARD,
        SampleProvenance.SYNTHETIC_BOARD,
        "Physics 101: Newton Laws",
        np.array(im_d2),
        "White chalk text on dark classroom chalkboard"
    ))

    # D3. Board with Perspective Distortion
    im_d3 = Image.new('L', (400, 70), color=245)
    d = ImageDraw.Draw(im_d3)
    d.text((30, 25), "CLASSROOM LECTURE NOTES", fill=20)
    # Perspective shear
    im_d3_skew = im_d3.transform(
        (400, 70),
        Image.AFFINE,
        (1.0, 0.15, -10, 0.0, 1.0, 0),
        resample=Image.BICUBIC,
        fillcolor=245
    )
    samples.append(BenchmarkSample(
        "D3_board_perspective",
        BenchmarkCategory.CLASSROOM_BOARD,
        SampleProvenance.SYNTHETIC_BOARD,
        "CLASSROOM LECTURE NOTES",
        np.array(im_d3_skew),
        "Classroom whiteboard with angled camera perspective"
    ))

    # ---------------------------------------------------------
    # CATEGORY E: MATHEMATICS & CLASSROOM NOTES
    # ---------------------------------------------------------
    # E1. Basic Arithmetic
    im_e1 = Image.new('L', (300, 60), color=255)
    d = ImageDraw.Draw(im_e1)
    d.text((20, 20), "12 + 34 = 46", fill=0)
    samples.append(BenchmarkSample(
        "E1_math_arithmetic",
        BenchmarkCategory.MATHEMATICS,
        SampleProvenance.SYNTHETIC_PRINTED,
        "12 + 34 = 46",
        np.array(im_e1),
        "Basic arithmetic equation"
    ))

    # E2. Physics Formula
    im_e2 = Image.new('RGB', (350, 60), color=(255, 255, 255))
    d = ImageDraw.Draw(im_e2)
    d.text((20, 20), "E = mc^2 and F = ma", fill=(0, 0, 0))
    samples.append(BenchmarkSample(
        "E2_math_formula",
        BenchmarkCategory.MATHEMATICS,
        SampleProvenance.SYNTHETIC_PRINTED,
        "E = mc^2 and F = ma",
        np.array(im_e2),
        "Elementary physics equations with exponents"
    ))

    # E3. Linear Equation
    im_e3 = Image.new('L', (350, 60), color=255)
    d = ImageDraw.Draw(im_e3)
    d.text((20, 20), "dy/dx = 2x + 5", fill=0)
    samples.append(BenchmarkSample(
        "E3_math_calculus",
        BenchmarkCategory.MATHEMATICS,
        SampleProvenance.SYNTHETIC_PRINTED,
        "dy/dx = 2x + 5",
        np.array(im_e3),
        "Simple derivative equation"
    ))

    return samples


def evaluate_engine_on_benchmark(
    engine: OCREngine,
    samples: Optional[List[BenchmarkSample]] = None,
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """Runs OCR evaluation across all benchmark samples and aggregates metrics by category."""
    import time
    
    if samples is None:
        samples = generate_multi_type_benchmark()
        
    evaluations: List[SampleEvaluation] = []
    
    for s in samples:
        w, h = s.dimensions
        c = s.channels
        numerical_range = (0, 255)
        
        inp = OCRInput(
            image=s.image_array,
            width=w,
            height=h,
            channels=c,
            numerical_range=numerical_range,
            timestamp=1000,
            seq_num=1
        )
        
        t0 = time.perf_counter()
        res = engine.process(inp)
        t1 = time.perf_counter()
        latency = (t1 - t0) * 1000.0
        
        pred_text = res.full_text
        cer = calculate_cer(s.expected_text, pred_text, case_sensitive=case_sensitive)
        wer = calculate_wer(s.expected_text, pred_text, case_sensitive=case_sensitive)
        
        norm_ref = normalize_text(s.expected_text, case_sensitive=case_sensitive)
        norm_pred = normalize_text(pred_text, case_sensitive=case_sensitive)
        exact_match = (norm_ref == norm_pred)
        
        avg_conf = 0.0
        if res.regions:
            avg_conf = float(np.mean([r.confidence for r in res.regions]))
            
        evals = SampleEvaluation(
            sample=s,
            recognized_text=pred_text,
            cer=cer,
            wer=wer,
            exact_match=exact_match,
            confidence=avg_conf,
            latency_ms=latency,
            regions_count=len(res.regions)
        )
        evaluations.append(evals)
        
    # Aggregate by Category
    cat_summary: Dict[str, Dict[str, Any]] = {}
    for cat in BenchmarkCategory:
        cat_evals = [e for e in evaluations if e.sample.category == cat]
        if not cat_evals:
            continue
        cat_summary[cat.value] = {
            "sample_count": len(cat_evals),
            "mean_cer": float(np.mean([e.cer for e in cat_evals])),
            "mean_wer": float(np.mean([e.wer for e in cat_evals])),
            "exact_match_rate": float(np.mean([1.0 if e.exact_match else 0.0 for e in cat_evals])),
            "mean_confidence": float(np.mean([e.confidence for e in cat_evals])),
            "mean_latency_ms": float(np.mean([e.latency_ms for e in cat_evals]))
        }
        
    total_summary = {
        "total_samples": len(evaluations),
        "mean_cer": float(np.mean([e.cer for e in evaluations])),
        "mean_wer": float(np.mean([e.wer for e in evaluations])),
        "exact_match_rate": float(np.mean([1.0 if e.exact_match else 0.0 for e in evaluations])),
        "mean_latency_ms": float(np.mean([e.latency_ms for e in evaluations])),
        "category_breakdown": cat_summary,
        "evaluations": evaluations
    }
    
    return total_summary
