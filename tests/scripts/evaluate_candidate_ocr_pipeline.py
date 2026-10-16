import os
import sys
import time
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.ai.tesseract_ocr import TesseractOCREngine
from backend.ai.rapid_ocr import RapidOCREngine
from backend.ai.notebook_ocr_pipeline import NotebookOCRPipeline
from backend.vision.frame import OCRInput
from backend.ai.ocr_benchmark import calculate_cer, calculate_wer, normalize_text

DATASET_DIR = "data/external/notebooks"
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
OUTPUT_JSON = "tests/results/phase4c20-candidate-notebook-ocr.json"

DEV_SAMPLE_IDS = [f"nb_{i:02d}" for i in range(1, 16)]
HELDOUT_SAMPLE_IDS = [f"nb_{i:02d}" for i in range(16, 31)]


def run_candidate_evaluation():
    print("================================================================================")
    print("PHASE 4C.20 — CANDIDATE NOTEBOOK OCR PIPELINE FINAL EVALUATION")
    print("================================================================================")

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 30, f"Expected 30 records, got {len(records)}"
    print(f"Loaded {len(records)} verified notebook records (15 Dev, 15 Held-Out).")

    # 1. Preload images into memory
    print("\nPre-loading 30 genuine notebook page scans into RAM...")
    loaded_inputs = {}
    for r in records:
        s_id = r["id"]
        img_path = os.path.join(DATASET_DIR, r["file_name"])
        with Image.open(img_path) as pil_img:
            pil_rgb = pil_img.convert('RGB')
            w, h = pil_rgb.size
            arr = np.array(pil_rgb, dtype=np.uint8)
        loaded_inputs[s_id] = {
            "input": OCRInput(
                image=arr,
                width=w,
                height=h,
                channels=3,
                numerical_range=(0, 255),
                timestamp=1000,
                seq_num=int(s_id.split('_')[-1])
            ),
            "dimensions": [w, h],
            "metadata": r
        }

    # 2. Cold-Start Initialization
    print("\nMeasuring Cold-Start Initialization...")
    t0 = time.perf_counter()
    candidate_pipeline = NotebookOCRPipeline(
        recognizer_engine=None,  # instantiate fresh RapidOCREngine
        enable_clahe=True,
        filter_cjk=True,
        tag_anomalies=False
    )
    cold_start_candidate_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Candidate Pipeline Cold Start: {cold_start_candidate_ms:.2f} ms")

    # 3. Warm-up
    print("\nWarming up candidate pipeline on synthetic buffer...")
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_inp = OCRInput(image=dummy, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)
    candidate_pipeline.process(dummy_inp)

    # 4. Evaluation across all 30 pages
    print("\nExecuting candidate pipeline evaluation across all 30 notebook pages...")
    per_sample_results = []

    dev_cers, dev_wers, dev_lats = [], [], []
    heldout_cers, heldout_wers, heldout_lats = [], [], []
    all_cers, all_wers, all_lats = [], [], []

    dev_ref_chars, dev_err_chars = 0, 0
    dev_ref_words, dev_err_words = 0, 0
    heldout_ref_chars, heldout_err_chars = 0, 0
    heldout_ref_words, heldout_err_words = 0, 0

    from backend.ai.ocr_benchmark import levenshtein_distance

    for idx, r in enumerate(records):
        s_id = r["id"]
        split = "Dev" if s_id in DEV_SAMPLE_IDS else "HeldOut"
        ref_text = r["transcription"]
        norm_ref = normalize_text(ref_text)
        inp = loaded_inputs[s_id]["input"]

        t0 = time.perf_counter()
        res = candidate_pipeline.process(inp)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        pred_text = res.full_text
        norm_pred = normalize_text(pred_text)
        cer = calculate_cer(ref_text, pred_text)
        wer = calculate_wer(ref_text, pred_text)
        exact_match = (norm_ref == norm_pred) and (len(norm_ref) > 0)

        # Micro metrics calculation
        ref_c_len = len(norm_ref)
        dist_c = levenshtein_distance(list(norm_ref), list(norm_pred))
        ref_w_list = norm_ref.split()
        hyp_w_list = norm_pred.split()
        ref_w_len = len(ref_w_list)
        dist_w = levenshtein_distance(ref_w_list, hyp_w_list)

        all_cers.append(cer)
        all_wers.append(wer)
        all_lats.append(lat_ms)

        if split == "Dev":
            dev_cers.append(cer)
            dev_wers.append(wer)
            dev_lats.append(lat_ms)
            dev_ref_chars += ref_c_len
            dev_err_chars += dist_c
            dev_ref_words += ref_w_len
            dev_err_words += dist_w
        else:
            heldout_cers.append(cer)
            heldout_wers.append(wer)
            heldout_lats.append(lat_ms)
            heldout_ref_chars += ref_c_len
            heldout_err_chars += dist_c
            heldout_ref_words += ref_w_len
            heldout_err_words += dist_w

        # Error categorization
        error_categories = []
        if cer > 1.0:
            error_categories.append("insertion_word_density_exceeds_reference")
        if "integral" in ref_text.lower() or "sum" in ref_text.lower() or "exp(" in ref_text.lower() or "dy/dx" in ref_text:
            error_categories.append("mathematical_notation_substitution")
        if res.quality_report.is_blurry:
            error_categories.append("image_blur_warning")
        if res.unreadable_count > 0:
            error_categories.append("low_confidence_handwriting_regions")

        per_sample_results.append({
            "sample_id": s_id,
            "split": split,
            "category": r["category"],
            "subject": r["subject"],
            "topic": r["topic"],
            "file_name": r["file_name"],
            "dimensions": loaded_inputs[s_id]["dimensions"],
            "reference_text": ref_text,
            "prediction": pred_text,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact_match": exact_match,
            "latency_ms": round(lat_ms, 2),
            "num_lines": res.num_lines,
            "num_paragraphs": res.num_paragraphs,
            "quality_report": res.quality_report.to_dict(),
            "unreadable_regions": res.unreadable_count,
            "formula_regions": res.formula_count,
            "error_categories": error_categories
        })

        print(f"  [{idx+1:02d}/30] {s_id} ({split}): CER={cer*100:6.2f}%, WER={wer*100:6.2f}%, Lat={lat_ms:7.1f}ms | Lines={res.num_lines}")

    # Aggregates
    summary = {
        "benchmark_name": "Phase 4C.20 Candidate Notebook OCR Pipeline Final Evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance_summary": {
            "dataset_name": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
            "sample_count": 30,
            "dev_samples": 15,
            "heldout_samples": 15,
            "license": "CC BY 4.0 / Open Access Research Dataset"
        },
        "system_environment": {
            "pipeline": "NotebookOCRPipeline",
            "recognizer": "RapidOCREngine (PP-OCRv3/v4 ONNX)",
            "clahe_enabled": True,
            "filter_cjk": True,
            "tag_anomalies": False,
            "cold_start_ms": round(cold_start_candidate_ms, 2)
        },
        "dataset_summaries": {
            "dev_split": {
                "sample_count": 15,
                "macro_cer": round(float(np.mean(dev_cers)), 4),
                "macro_wer": round(float(np.mean(dev_wers)), 4),
                "micro_cer": round(float(dev_err_chars / max(dev_ref_chars, 1)), 4),
                "micro_wer": round(float(dev_err_words / max(dev_ref_words, 1)), 4),
                "exact_match_count": sum(1 for s in per_sample_results if s["split"] == "Dev" and s["exact_match"]),
                "exact_match_rate": round(sum(1 for s in per_sample_results if s["split"] == "Dev" and s["exact_match"]) / 15.0, 4),
                "mean_latency_ms": round(float(np.mean(dev_lats)), 2),
                "median_latency_ms": round(float(np.median(dev_lats)), 2)
            },
            "heldout_split": {
                "sample_count": 15,
                "macro_cer": round(float(np.mean(heldout_cers)), 4),
                "macro_wer": round(float(np.mean(heldout_wers)), 4),
                "micro_cer": round(float(heldout_err_chars / max(heldout_ref_chars, 1)), 4),
                "micro_wer": round(float(heldout_err_words / max(heldout_ref_words, 1)), 4),
                "exact_match_count": sum(1 for s in per_sample_results if s["split"] == "HeldOut" and s["exact_match"]),
                "exact_match_rate": round(sum(1 for s in per_sample_results if s["split"] == "HeldOut" and s["exact_match"]) / 15.0, 4),
                "mean_latency_ms": round(float(np.mean(heldout_lats)), 2),
                "median_latency_ms": round(float(np.median(heldout_lats)), 2)
            },
            "full_dataset": {
                "sample_count": 30,
                "macro_cer": round(float(np.mean(all_cers)), 4),
                "macro_wer": round(float(np.mean(all_wers)), 4),
                "micro_cer": round(float((dev_err_chars + heldout_err_chars) / max(dev_ref_chars + heldout_ref_chars, 1)), 4),
                "micro_wer": round(float((dev_err_words + heldout_err_words) / max(dev_ref_words + heldout_ref_words, 1)), 4),
                "exact_match_count": sum(1 for s in per_sample_results if s["exact_match"]),
                "exact_match_rate": round(sum(1 for s in per_sample_results if s["exact_match"]) / 30.0, 4),
                "mean_latency_ms": round(float(np.mean(all_lats)), 2),
                "median_latency_ms": round(float(np.median(all_lats)), 2)
            }
        },
        "per_sample_evaluations": per_sample_results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(" CANDIDATE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Dev Split (15 pages):     Macro CER = {summary['dataset_summaries']['dev_split']['macro_cer']*100:.2f}%, Mean Latency = {summary['dataset_summaries']['dev_split']['mean_latency_ms']:.1f}ms")
    print(f"Held-Out Split (15 pages): Macro CER = {summary['dataset_summaries']['heldout_split']['macro_cer']*100:.2f}%, Mean Latency = {summary['dataset_summaries']['heldout_split']['mean_latency_ms']:.1f}ms")
    print(f"Full Dataset (30 pages):   Macro CER = {summary['dataset_summaries']['full_dataset']['macro_cer']*100:.2f}%, Mean Latency = {summary['dataset_summaries']['full_dataset']['mean_latency_ms']:.1f}ms")
    print(f"\nArtifact successfully saved to: {OUTPUT_JSON}")
    return summary


if __name__ == "__main__":
    run_candidate_evaluation()
