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
OUTPUT_JSON = "tests/results/phase4c19-notebook-ocr-pipeline.json"

DEV_SAMPLE_IDS = [f"nb_{i:02d}" for i in range(1, 16)]
HELDOUT_SAMPLE_IDS = [f"nb_{i:02d}" for i in range(16, 31)]

def run_notebook_pipeline_evaluation():
    print("================================================================================")
    print("PHASE 4C.19 — NOTEBOOK OCR PIPELINE EVALUATION")
    print("================================================================================")

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 30, f"Expected 30 records, got {len(records)}"

    print(f"Loaded {len(records)} verified notebook records (15 Dev, 15 Held-Out).")

    # 1. Preload images
    print("\nPre-loading images into memory...")
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

    # 2. Initialize Engines
    print("\nInitializing OCR Engines and Experimental Pipeline...")
    tess_engine = TesseractOCREngine()
    rapid_engine = RapidOCREngine()
    notebook_pipeline = NotebookOCRPipeline(recognizer_engine=rapid_engine, enable_clahe=True)

    # 3. Warm-up
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_inp = OCRInput(image=dummy, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)
    tess_engine.process(dummy_inp)
    notebook_pipeline.process(dummy_inp)

    # 4. Run Evaluation
    print("\nEvaluating all 30 pages across 3 pipelines...")
    per_sample_results = []

    for r in records:
        s_id = r["id"]
        split = "Dev" if s_id in DEV_SAMPLE_IDS else "HeldOut"
        ref_text = r["transcription"]
        norm_ref = normalize_text(ref_text)
        inp = loaded_inputs[s_id]["input"]

        # 1. Baseline Tesseract
        t0 = time.perf_counter()
        tess_res = tess_engine.process(inp)
        tess_lat = (time.perf_counter() - t0) * 1000.0
        tess_pred = tess_res.full_text
        tess_cer = calculate_cer(ref_text, tess_pred)
        tess_wer = calculate_wer(ref_text, tess_pred)
        tess_exact = (norm_ref == normalize_text(tess_pred)) and (len(norm_ref) > 0)

        # 2. Baseline RapidOCR Full-Page
        t0 = time.perf_counter()
        rapid_res = rapid_engine.process(inp)
        rapid_lat = (time.perf_counter() - t0) * 1000.0
        rapid_pred = rapid_res.full_text
        rapid_cer = calculate_cer(ref_text, rapid_pred)
        rapid_wer = calculate_wer(ref_text, rapid_pred)
        rapid_exact = (norm_ref == normalize_text(rapid_pred)) and (len(norm_ref) > 0)

        # 3. Experimental NotebookOCRPipeline
        t0 = time.perf_counter()
        pipe_res = notebook_pipeline.process(inp)
        pipe_lat = (time.perf_counter() - t0) * 1000.0
        pipe_pred = pipe_res.full_text
        pipe_cer = calculate_cer(ref_text, pipe_pred)
        pipe_wer = calculate_wer(ref_text, pipe_pred)
        pipe_exact = (norm_ref == normalize_text(pipe_pred)) and (len(norm_ref) > 0)

        per_sample_results.append({
            "sample_id": s_id,
            "split": split,
            "category": r["category"],
            "subject": r["subject"],
            "topic": r["topic"],
            "file_name": r["file_name"],
            "dimensions": loaded_inputs[s_id]["dimensions"],
            "quality_report": pipe_res.quality_report.to_dict(),
            "detected_lines": pipe_res.num_lines,
            "detected_paragraphs": pipe_res.num_paragraphs,
            "unreadable_regions": pipe_res.unreadable_count,
            "formula_regions": pipe_res.formula_count,
            "reference_text": ref_text,
            "tesseract_baseline": {
                "prediction": tess_pred,
                "cer": round(tess_cer, 4),
                "wer": round(tess_wer, 4),
                "exact_match": tess_exact,
                "latency_ms": round(tess_lat, 2)
            },
            "rapidocr_baseline": {
                "prediction": rapid_pred,
                "cer": round(rapid_cer, 4),
                "wer": round(rapid_wer, 4),
                "exact_match": rapid_exact,
                "latency_ms": round(rapid_lat, 2)
            },
            "notebook_ocr_pipeline": {
                "prediction": pipe_pred,
                "cer": round(pipe_cer, 4),
                "wer": round(pipe_wer, 4),
                "exact_match": pipe_exact,
                "latency_ms": round(pipe_lat, 2),
                "processing_status": pipe_res.processing_status
            }
        })
        print(f"  Processed {s_id} ({split}) | Tess CER: {tess_cer*100:.1f}% | Rapid CER: {rapid_cer*100:.1f}% | Pipe CER: {pipe_cer*100:.1f}%")

    # 5. Summarize metrics
    def aggregate_pipeline(items, pipe_key):
        cers = [x[pipe_key]["cer"] for x in items]
        wers = [x[pipe_key]["wer"] for x in items]
        exacts = [x[pipe_key]["exact_match"] for x in items]
        lats = [x[pipe_key]["latency_ms"] for x in items]

        total_ref_chars = sum([max(len(normalize_text(x["reference_text"])), 1) for x in items])
        total_char_edits = sum([int(round(x[pipe_key]["cer"] * max(len(normalize_text(x["reference_text"])), 1))) for x in items])
        total_ref_words = sum([max(len(normalize_text(x["reference_text"]).split()), 1) for x in items])
        total_word_edits = sum([int(round(x[pipe_key]["wer"] * max(len(normalize_text(x["reference_text"]).split()), 1))) for x in items])

        empty_cnt = sum(1 for x in items if len(normalize_text(x[pipe_key]["prediction"])) == 0)

        return {
            "sample_count": len(items),
            "macro_cer": round(float(np.mean(cers)), 4),
            "macro_wer": round(float(np.mean(wers)), 4),
            "micro_cer": round(float(total_char_edits / total_ref_chars), 4),
            "micro_wer": round(float(total_word_edits / total_ref_words), 4),
            "exact_match_count": int(sum(exacts)),
            "exact_match_rate": round(float(np.mean(exacts)), 4),
            "empty_output_count": empty_cnt,
            "empty_output_rate": round(float(empty_cnt / len(items)), 4),
            "mean_latency_ms": round(float(np.mean(lats)), 2),
            "median_latency_ms": round(float(np.median(lats)), 2)
        }

    dev_items = [x for x in per_sample_results if x["split"] == "Dev"]
    held_items = [x for x in per_sample_results if x["split"] == "HeldOut"]

    summaries = {}
    for s_key, s_data in [("full_dataset", per_sample_results), ("dev_split", dev_items), ("heldout_split", held_items)]:
        summaries[s_key] = {
            "tesseract_baseline": aggregate_pipeline(s_data, "tesseract_baseline"),
            "rapidocr_baseline": aggregate_pipeline(s_data, "rapidocr_baseline"),
            "notebook_ocr_pipeline": aggregate_pipeline(s_data, "notebook_ocr_pipeline")
        }

    output_data = {
        "benchmark_name": "Phase 4C.19 Notebook OCR Pipeline Evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_architecture": {
            "preprocessor": "CLAHE on L-channel (clipLimit=1.5, tileGridSize=8x8)",
            "detector_recognizer": "RapidOCR ONNX PP-OCRv4 (DBNet + SVTR/CRNN)",
            "layout_segmenter": "Multi-column center gap detection + horizontal line clustering",
            "quality_verifier": "Laplacian variance blur check + std luminance contrast check",
            "text_reconstruction": "Paragraph gap thresholding (1.6x line height) + low-confidence tagging"
        },
        "dataset_summaries": summaries,
        "per_sample_evaluations": per_sample_results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print(f"\nWrote results to {OUTPUT_JSON}")
    print("\n================================================================================")
    print("PHASE 4C.19 BENCHMARK SUMMARY TABLE:")
    print("================================================================================")
    print(f"{'Split / Metric':<20} | {'Tesseract Baseline':<20} | {'RapidOCR Baseline':<20} | {'Notebook OCR Pipeline':<22}")
    print("-" * 92)
    for split_key, split_label in [("dev_split", "Dev Split (15)"), ("heldout_split", "Held-Out Split (15)"), ("full_dataset", "Full Dataset (30)")]:
        t = summaries[split_key]["tesseract_baseline"]
        r = summaries[split_key]["rapidocr_baseline"]
        p = summaries[split_key]["notebook_ocr_pipeline"]
        print(f"{split_label:<20} | CER: {t['macro_cer']*100:>6.2f}% (WER:{t['macro_wer']*100:>6.2f}%) | CER: {r['macro_cer']*100:>6.2f}% (WER:{r['macro_wer']*100:>6.2f}%) | CER: {p['macro_cer']*100:>6.2f}% (WER:{p['macro_wer']*100:>6.2f}%)")
        print(f"{'  Latency (mean / med)':<20} | {t['mean_latency_ms']:>6.1f}ms / {t['median_latency_ms']:>6.1f}ms    | {r['mean_latency_ms']:>6.1f}ms / {r['median_latency_ms']:>6.1f}ms    | {p['mean_latency_ms']:>6.1f}ms / {p['median_latency_ms']:>6.1f}ms")
        print("-" * 92)
    print("================================================================================\n")
    return output_data

if __name__ == '__main__':
    run_notebook_pipeline_evaluation()
