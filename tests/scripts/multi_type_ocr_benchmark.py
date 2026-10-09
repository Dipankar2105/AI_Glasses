import sys, os, time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.ai.tesseract_ocr import TesseractOCREngine
from backend.ai.ocr_benchmark import (
    generate_multi_type_benchmark,
    evaluate_engine_on_benchmark,
    calculate_cer,
    calculate_wer,
    normalize_text,
    BenchmarkCategory
)
from backend.vision.frame import OCRInput
from backend.ai.errors import AIVisionError


def run_benchmark():
    print("================================================================================")
    print("AI GLASSES — MULTI-TYPE OCR ROBUSTNESS & HANDWRITING BENCHMARK (PHASE 4C.4)")
    print("================================================================================")
    
    engine = TesseractOCREngine()
    samples = generate_multi_type_benchmark()
    
    print(f"Total Test Fixtures: {len(samples)}")
    print("Evaluating Tesseract 5.4.0 on Multi-Type Benchmark...\n")
    
    results = evaluate_engine_on_benchmark(engine, samples, case_sensitive=False)
    
    # 1. Print Detailed Sample-by-Sample Results
    print(f"{'Sample ID':<25} | {'Category':<22} | {'Exp':<30} | {'Recog':<30} | {'CER':<5} | {'WER':<5} | {'EM'}")
    print("-" * 135)
    for ev in results["evaluations"]:
        s = ev.sample
        exp_disp = (s.expected_text[:27] + "...") if len(s.expected_text) > 27 else s.expected_text
        rec_disp = (ev.recognized_text[:27] + "...") if len(ev.recognized_text) > 27 else ev.recognized_text
        em_str = "PASS" if ev.exact_match else "FAIL"
        print(f"{s.sample_id:<25} | {s.category.name:<22} | {exp_disp:<30} | {rec_disp:<30} | {ev.cer:.2f}  | {ev.wer:.2f}  | {em_str}")

    # 2. Print Category Summary Breakdown
    print("\n" + "=" * 80)
    print("CATEGORY PERFORMANCE BREAKDOWN")
    print("=" * 80)
    print(f"{'Category':<45} | {'Samples':<7} | {'Mean CER':<9} | {'Mean WER':<9} | {'Exact Match'}")
    print("-" * 80)
    for cat_name, cat_data in results["category_breakdown"].items():
        print(f"{cat_name:<45} | {cat_data['sample_count']:<7} | {cat_data['mean_cer']:<9.2f} | {cat_data['mean_wer']:<9.2f} | {cat_data['exact_match_rate']*100:.1f}%")

    print("-" * 80)
    print(f"{'OVERALL AVERAGE':<45} | {results['total_samples']:<7} | {results['mean_cer']:<9.2f} | {results['mean_wer']:<9.2f} | {results['exact_match_rate']*100:.1f}%")
    print(f"Mean Inference Latency: {results['mean_latency_ms']:.2f} ms\n")

    # 3. Controlled Board Preprocessing Experiments
    print("=" * 80)
    print("BOARD PREPROCESSING & PERSPECTIVE CORRECTION EXPERIMENTS")
    print("=" * 80)
    
    # Board sample with perspective skew
    board_sample = next(s for s in samples if s.sample_id == "D3_board_perspective")
    ref_text = board_sample.expected_text
    
    # A. Raw skewed image
    raw_img = board_sample.image_array
    
    # B. Contrast normalized
    from backend.vision.pipeline import VisionPipeline
    pipeline = VisionPipeline()
    vframe_raw = pipeline.create_ocr_input(
        pipeline.process_frame(raw_img) if hasattr(pipeline, 'process_frame') else None
    ) if False else None
    
    # Let's test distinct preprocessing variations:
    # 1. Raw Skewed
    # 2. Local Thresholding (Binarized)
    # 3. Perspective Rectification (Unskewed via Inverse Affine)
    im_pil = Image.fromarray(raw_img)
    im_rectified = im_pil.transform(
        (400, 70),
        Image.AFFINE,
        (1.0, -0.15, 10, 0.0, 1.0, 0),
        resample=Image.BICUBIC,
        fillcolor=245
    )
    arr_rectified = np.array(im_rectified)
    
    # 4. Perspective Rectification + Thresholding
    arr_rectified_thresh = np.where(arr_rectified < 128, 0, 255).astype(np.uint8)
    
    board_experiments = {
        "1. Raw Perspective Skew": raw_img,
        "2. OCR-Local Thresholding": np.where(raw_img < 128, 0, 255).astype(np.uint8),
        "3. Perspective Rectified": arr_rectified,
        "4. Rectified + Thresholded": arr_rectified_thresh
    }
    
    for exp_name, arr in board_experiments.items():
        h, w = arr.shape[:2]
        c = 1 if len(arr.shape) == 2 else arr.shape[2]
        inp = OCRInput(arr, w, h, c, (0, 255), 1000, 1)
        res = engine.process(inp)
        cer = calculate_cer(ref_text, res.full_text)
        wer = calculate_wer(ref_text, res.full_text)
        print(f"{exp_name:<30} | Recog: '{res.full_text}' | CER: {cer:.2f} | WER: {wer:.2f}")


if __name__ == "__main__":
    run_benchmark()
