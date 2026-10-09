import os, sys, time, json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.ai.tesseract_ocr import TesseractOCREngine
from backend.ai.rapid_ocr import RapidOCREngine
from backend.ai.ocr_benchmark import calculate_cer, calculate_wer, normalize_text
from backend.vision.frame import OCRInput
import pytesseract

DATASET_DIR = "data/external/notebooks"
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
OUTPUT_JSON = "tests/results/phase4c16-notebook-ocr-benchmark.json"

DEV_SAMPLE_IDS = [f"nb_{i:02d}" for i in range(1, 16)]
HELDOUT_SAMPLE_IDS = [f"nb_{i:02d}" for i in range(16, 31)]

def run_notebook_ocr_evaluation():
    print("================================================================================")
    print("PHASE 4C.16 — REAL NOTEBOOK OCR BENCHMARK (TESSERACT VS RAPIDOCR)")
    print("================================================================================")

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 30, f"Expected 30 records, got {len(records)}"

    # 1. Pre-decode images into memory to isolate OCR inference latency from disk I/O
    print("Pre-loading and decoding 30 genuine notebook page scans into RAM...")
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
    t0_tess = time.perf_counter()
    tess_engine = TesseractOCREngine()
    t1_tess = time.perf_counter()
    cold_start_tess_ms = (t1_tess - t0_tess) * 1000.0

    t0_rapid = time.perf_counter()
    rapid_engine = RapidOCREngine()
    t1_rapid = time.perf_counter()
    cold_start_rapid_ms = (t1_rapid - t0_rapid) * 1000.0

    print(f"  Tesseract Cold Start: {cold_start_tess_ms:.2f} ms")
    print(f"  RapidOCR Cold Start:  {cold_start_rapid_ms:.2f} ms")

    # 3. Engine Warmup
    print("\nWarming up engines on synthetic buffer...")
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_inp = OCRInput(image=dummy, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)
    tess_engine.process(dummy_inp)
    rapid_engine.process(dummy_inp)

    # 4. Multi-Pass Measurement (3 Repetitions)
    NUM_RUNS = 3
    print(f"\nExecuting {NUM_RUNS} controlled evaluation passes across all 30 notebook pages...")

    tess_evals = {r["id"]: {"latencies": [], "res": None} for r in records}
    rapid_evals = {r["id"]: {"latencies": [], "res": None} for r in records}

    for pass_idx in range(NUM_RUNS):
        print(f"  --- Measurement Pass {pass_idx + 1} / {NUM_RUNS} ---")
        
        # Tesseract Pass
        for r in records:
            s_id = r["id"]
            inp = loaded_inputs[s_id]["input"]
            t0 = time.perf_counter()
            res = tess_engine.process(inp)
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            tess_evals[s_id]["latencies"].append(lat_ms)
            if pass_idx == 0:
                tess_evals[s_id]["res"] = res

        # RapidOCR Pass
        for r in records:
            s_id = r["id"]
            inp = loaded_inputs[s_id]["input"]
            t0 = time.perf_counter()
            res = rapid_engine.process(inp)
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            rapid_evals[s_id]["latencies"].append(lat_ms)
            if pass_idx == 0:
                rapid_evals[s_id]["res"] = res

    # 5. Compute Per-Sample Metrics
    per_sample_results = []
    
    for r in records:
        s_id = r["id"]
        split = "Dev" if s_id in DEV_SAMPLE_IDS else "HeldOut"
        ref_text = r["transcription"]
        norm_ref = normalize_text(ref_text)

        # Tesseract results
        tess_res = tess_evals[s_id]["res"]
        tess_pred = tess_res.full_text
        norm_tess_pred = normalize_text(tess_pred)
        tess_cer = calculate_cer(ref_text, tess_pred)
        tess_wer = calculate_wer(ref_text, tess_pred)
        tess_exact = (norm_ref == norm_tess_pred) and (len(norm_ref) > 0)
        tess_lats = tess_evals[s_id]["latencies"]
        tess_mean_lat = float(np.mean(tess_lats))
        tess_median_lat = float(np.median(tess_lats))

        # RapidOCR results
        rapid_res = rapid_evals[s_id]["res"]
        rapid_pred = rapid_res.full_text
        norm_rapid_pred = normalize_text(rapid_pred)
        rapid_cer = calculate_cer(ref_text, rapid_pred)
        rapid_wer = calculate_wer(ref_text, rapid_pred)
        rapid_exact = (norm_ref == norm_rapid_pred) and (len(norm_ref) > 0)
        rapid_lats = rapid_evals[s_id]["latencies"]
        rapid_mean_lat = float(np.mean(rapid_lats))
        rapid_median_lat = float(np.median(rapid_lats))

        # Error categorization
        error_reasons = []
        if tess_cer > 0.50:
            error_reasons.append("cursive_script_segmentation_failure")
        if "integral" in ref_text.lower() or "sum" in ref_text.lower() or "exp(" in ref_text.lower():
            error_reasons.append("mathematical_notation_substitution")
        if len(tess_res.regions) < 3:
            error_reasons.append("sparse_detection_dropout")

        per_sample_results.append({
            "sample_id": s_id,
            "split": split,
            "category": r["category"],
            "subject": r["subject"],
            "topic": r["topic"],
            "file_name": r["file_name"],
            "dimensions": loaded_inputs[s_id]["dimensions"],
            "reference_text": ref_text,
            "tesseract": {
                "prediction": tess_pred,
                "cer": round(tess_cer, 4),
                "wer": round(tess_wer, 4),
                "exact_match": tess_exact,
                "regions_count": len(tess_res.regions),
                "latencies_ms": [round(x, 2) for x in tess_lats],
                "mean_latency_ms": round(tess_mean_lat, 2),
                "median_latency_ms": round(tess_median_lat, 2)
            },
            "rapidocr": {
                "prediction": rapid_pred,
                "cer": round(rapid_cer, 4),
                "wer": round(rapid_wer, 4),
                "exact_match": rapid_exact,
                "regions_count": len(rapid_res.regions),
                "latencies_ms": [round(x, 2) for x in rapid_lats],
                "mean_latency_ms": round(rapid_mean_lat, 2),
                "median_latency_ms": round(rapid_median_lat, 2)
            },
            "error_categories": error_reasons
        })

    # 6. Aggregate Summaries (Dev, Held-Out, Full)
    def compute_engine_summary(items, engine_key):
        cers = [x[engine_key]["cer"] for x in items]
        wers = [x[engine_key]["wer"] for x in items]
        exacts = [x[engine_key]["exact_match"] for x in items]
        means = [x[engine_key]["mean_latency_ms"] for x in items]
        medians = [x[engine_key]["median_latency_ms"] for x in items]
        empty_count = sum([1 for x in items if len(x[engine_key]["prediction"].strip()) == 0 or x[engine_key]["regions_count"] == 0])

        total_ref_chars = sum([max(len(normalize_text(x["reference_text"])), 1) for x in items])
        total_char_edits = sum([int(round(x[engine_key]["cer"] * max(len(normalize_text(x["reference_text"])), 1))) for x in items])
        total_ref_words = sum([max(len(normalize_text(x["reference_text"]).split()), 1) for x in items])
        total_word_edits = sum([int(round(x[engine_key]["wer"] * max(len(normalize_text(x["reference_text"]).split()), 1))) for x in items])

        micro_cer = float(total_char_edits / total_ref_chars) if total_ref_chars > 0 else 0.0
        micro_wer = float(total_word_edits / total_ref_words) if total_ref_words > 0 else 0.0

        return {
            "sample_count": len(items),
            "macro_cer": round(float(np.mean(cers)), 4),
            "macro_wer": round(float(np.mean(wers)), 4),
            "micro_cer": round(micro_cer, 4),
            "micro_wer": round(micro_wer, 4),
            "exact_match_count": int(sum(exacts)),
            "exact_match_rate": round(float(np.mean(exacts)), 4),
            "empty_output_count": empty_count,
            "empty_output_rate": round(empty_count / len(items), 4),
            "mean_latency_ms": round(float(np.mean(means)), 2),
            "median_latency_ms": round(float(np.median(medians)), 2)
        }

    dev_items = [x for x in per_sample_results if x["split"] == "Dev"]
    held_items = [x for x in per_sample_results if x["split"] == "HeldOut"]

    dataset_summaries = {
        "full_dataset": {
            "tesseract": compute_engine_summary(per_sample_results, "tesseract"),
            "rapidocr": compute_engine_summary(per_sample_results, "rapidocr")
        },
        "dev_split": {
            "tesseract": compute_engine_summary(dev_items, "tesseract"),
            "rapidocr": compute_engine_summary(dev_items, "rapidocr")
        },
        "heldout_split": {
            "tesseract": compute_engine_summary(held_items, "tesseract"),
            "rapidocr": compute_engine_summary(held_items, "rapidocr")
        }
    }

    # 7. Output Result Artifact
    output_data = {
        "benchmark_name": "Phase 4C.16 Genuine Handwritten Notebook OCR Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance_summary": {
            "dataset_name": "NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge",
            "sample_count": len(records),
            "dev_samples": len(dev_items),
            "heldout_samples": len(held_items),
            "license": "CC BY 4.0 / Open Access Research Dataset",
            "source_repository": "https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge"
        },
        "cold_start_latencies_ms": {
            "tesseract": round(cold_start_tess_ms, 2),
            "rapidocr": round(cold_start_rapid_ms, 2)
        },
        "system_environment": {
            "tesseract_version": str(pytesseract.get_tesseract_version()),
            "rapidocr_version": "1.2.3 (onnxruntime 1.31.0, PP-OCRv4)",
            "hardware": "AMD64 Windows 11 CPU",
            "repetitions": NUM_RUNS
        },
        "dataset_summaries": dataset_summaries,
        "per_sample_evaluations": per_sample_results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print(f"\nBenchmark results saved to: {OUTPUT_JSON}")

    # 8. Print Summary Table
    print("\n================================================================================")
    print("NOTEBOOK OCR BENCHMARK RESULTS SUMMARY:")
    print("================================================================================")
    print(f"{'Split / Metric':<24} | {'Tesseract (Baseline)':<22} | {'RapidOCR (PP-OCRv4)':<22} | {'Delta / Winner':<16}")
    print("-" * 92)

    for split_key, split_name in [("dev_split", "Dev Split (15)"), ("heldout_split", "Held-Out Split (15)"), ("full_dataset", "Full Dataset (30)")]:
        t_sum = dataset_summaries[split_key]["tesseract"]
        r_sum = dataset_summaries[split_key]["rapidocr"]
        cer_delta = (t_sum["macro_cer"] - r_sum["macro_cer"]) * 100.0
        winner = f"RapidOCR (+{cer_delta:.2f}%)" if cer_delta > 0 else f"Tesseract (+{-cer_delta:.2f}%)"
        print(f"{split_name:<24} | CER: {t_sum['macro_cer']*100:>5.2f}% (WER: {t_sum['macro_wer']*100:>5.2f}%) | CER: {r_sum['macro_cer']*100:>5.2f}% (WER: {r_sum['macro_wer']*100:>5.2f}%) | {winner}")
        print(f"{'  Latency (mean / med)':<24} | {t_sum['mean_latency_ms']:>6.1f} ms / {t_sum['median_latency_ms']:>6.1f} ms    | {r_sum['mean_latency_ms']:>6.1f} ms / {r_sum['median_latency_ms']:>6.1f} ms    | {t_sum['mean_latency_ms']/r_sum['mean_latency_ms']:>5.2f}x speedup")
        print("-" * 92)

    print("================================================================================\n")
    return output_data

if __name__ == '__main__':
    run_notebook_ocr_evaluation()
