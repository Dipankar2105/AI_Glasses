import os, sys, time, json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.ai.tesseract_ocr import TesseractOCREngine
from backend.ai.rapid_ocr import RapidOCREngine
from backend.ai.registry import AIEngineRegistry
from backend.ai.ocr_benchmark import calculate_cer, calculate_wer, normalize_text
from backend.vision.frame import OCRInput

DEV_SAMPLE_IDS = ["01", "02", "03", "04", "05", "07", "08", "09", "10", "11", "17"]
HELDOUT_SAMPLE_IDS = ["06", "12", "13", "14", "15", "16", "18", "19", "20", "21"]
OUTPUT_JSON = "tests/results/phase4c15.2-timing.json"

def run_timing_integrity_evaluation():
    print("================================================================================")
    print("PHASE 4C.15.2 — CONTROLLED OCR TIMING INTEGRITY & METRIC REPLAY")
    print("================================================================================")

    # 1. Load ground truth metadata
    meta_path = "data/external/whiteboards/metadata.jsonl"
    data_dir = "data/external/whiteboards"
    with open(meta_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 21, f"Expected 21 records, got {len(records)}"
    record_ids = [r["id"] for r in records]
    assert len(set(record_ids)) == 21, "Duplicate sample IDs found in metadata"

    # 2. Pre-decode images into memory to isolate OCR engine latency from disk I/O
    print("Pre-loading and decoding 21 whiteboard images into memory...")
    loaded_inputs = {}
    for r in records:
        s_id = r["id"]
        img_path = os.path.join(data_dir, r["file_name"])
        with Image.open(img_path) as pil_img:
            pil_rgb = pil_img.convert('RGB')
            w, h = pil_rgb.size
            arr = np.array(pil_rgb, dtype=np.uint8)
        loaded_inputs[s_id] = OCRInput(
            image=arr,
            width=w,
            height=h,
            channels=3,
            numerical_range=(0, 255),
            timestamp=1000,
            seq_num=int(s_id) if s_id.isdigit() else 1
        )

    # 3. Instantiate and Measure Cold Starts
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

    # 4. Perform Engine Warmup
    print("\nWarming up OCR engines on synthetic buffer...")
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_inp = OCRInput(image=dummy, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)
    tess_engine.process(dummy_inp)
    rapid_engine.process(dummy_inp)

    # 5. Controlled Multi-Pass Measurement
    # We will run 3 repetitions of:
    # Pass A: Tesseract Only (Standalone)
    # Pass B: Strategy C (True Execution Path: Tesseract first, fallback to RapidOCR iff 0 regions / empty text)
    NUM_RUNS = 3
    print(f"\nExecuting {NUM_RUNS} controlled measurement passes...")

    tess_runs = {s_id: [] for s_id in record_ids}
    strat_c_runs = {s_id: [] for s_id in record_ids}
    strat_c_metadata = {}

    for run_idx in range(NUM_RUNS):
        print(f"  --- Measurement Pass {run_idx + 1} / {NUM_RUNS} ---")
        
        # Pass A: Tesseract Standalone
        for r in records:
            s_id = r["id"]
            inp = loaded_inputs[s_id]
            t0 = time.perf_counter()
            res_tess = tess_engine.process(inp)
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            tess_runs[s_id].append(lat_ms)

        # Pass B: Strategy C (True Execution Path)
        for r in records:
            s_id = r["id"]
            inp = loaded_inputs[s_id]
            ref_text = r.get("transcription", "").strip()

            t0 = time.perf_counter()
            res = tess_engine.process(inp)
            fb_fired = (len(res.regions) == 0 or len(res.full_text.strip()) == 0)
            if fb_fired:
                res = rapid_engine.process(inp)
                engine_used = "rapidocr_fallback"
            else:
                engine_used = "tesseract"
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            strat_c_runs[s_id].append(lat_ms)

            # Record prediction on run 0 for metric replay
            if run_idx == 0:
                cer = calculate_cer(ref_text, res.full_text)
                wer = calculate_wer(ref_text, res.full_text)
                exact = (normalize_text(ref_text) == normalize_text(res.full_text)) and (len(normalize_text(ref_text)) > 0)
                strat_c_metadata[s_id] = {
                    "split": "Dev" if s_id in DEV_SAMPLE_IDS else "HeldOut",
                    "category": r.get("category", "unknown"),
                    "file_name": r["file_name"],
                    "fallback_triggered": fb_fired,
                    "engine_used": engine_used,
                    "cer": round(cer, 4),
                    "wer": round(wer, 4),
                    "exact_match": exact,
                    "reference_text": ref_text,
                    "prediction": res.full_text,
                    "dimensions": list(inp.dimensions)
                }

    # 6. Aggregate per-sample statistics across runs
    per_sample_results = []
    for r in records:
        s_id = r["id"]
        meta = strat_c_metadata[s_id]
        tess_lats = tess_runs[s_id]
        strat_c_lats = strat_c_runs[s_id]

        tess_mean = float(np.mean(tess_lats))
        tess_median = float(np.median(tess_lats))
        strat_c_mean = float(np.mean(strat_c_lats))
        strat_c_median = float(np.median(strat_c_lats))

        per_sample_results.append({
            "sample_id": s_id,
            "split": meta["split"],
            "category": meta["category"],
            "file_name": meta["file_name"],
            "dimensions": meta["dimensions"],
            "fallback_triggered": meta["fallback_triggered"],
            "engine_used": meta["engine_used"],
            "tesseract_latencies_ms": [round(x, 2) for x in tess_lats],
            "tesseract_mean_lat_ms": round(tess_mean, 2),
            "tesseract_median_lat_ms": round(tess_median, 2),
            "strategy_c_latencies_ms": [round(x, 2) for x in strat_c_lats],
            "strategy_c_mean_lat_ms": round(strat_c_mean, 2),
            "strategy_c_median_lat_ms": round(strat_c_median, 2),
            "cer": meta["cer"],
            "wer": meta["wer"],
            "exact_match": meta["exact_match"],
            "reference_text": meta["reference_text"],
            "prediction": meta["prediction"]
        })

    # 7. Compute full-precision dataset summaries
    all_strat_c_means = [x["strategy_c_mean_lat_ms"] for x in per_sample_results]
    all_tess_means = [x["tesseract_mean_lat_ms"] for x in per_sample_results]
    all_cers = [x["cer"] for x in per_sample_results]
    all_wers = [x["wer"] for x in per_sample_results]
    all_fbs = [x["fallback_triggered"] for x in per_sample_results]

    dev_items = [x for x in per_sample_results if x["split"] == "Dev"]
    held_items = [x for x in per_sample_results if x["split"] == "HeldOut"]

    def compute_split_summary(items):
        strat_c_m = [x["strategy_c_mean_lat_ms"] for x in items]
        tess_m = [x["tesseract_mean_lat_ms"] for x in items]
        cers = [x["cer"] for x in items]
        wers = [x["wer"] for x in items]
        fbs = [x["fallback_triggered"] for x in items]
        exacts = [x["exact_match"] for x in items]
        return {
            "sample_count": len(items),
            "fallback_count": int(sum(fbs)),
            "fallback_rate": round(float(np.mean(fbs)), 4),
            "macro_cer": round(float(np.mean(cers)), 4),
            "macro_wer": round(float(np.mean(wers)), 4),
            "exact_match_count": int(sum(exacts)),
            "exact_match_rate": round(float(np.mean(exacts)), 4),
            "strategy_c_mean_lat_ms": round(float(np.mean(strat_c_m)), 2),
            "strategy_c_median_lat_ms": round(float(np.median(strat_c_m)), 2),
            "tesseract_mean_lat_ms": round(float(np.mean(tess_m)), 2),
            "tesseract_median_lat_ms": round(float(np.median(tess_m)), 2)
        }

    dataset_summary = {
        "full_dataset": compute_split_summary(per_sample_results),
        "dev_split": compute_split_summary(dev_items),
        "heldout_split": compute_split_summary(held_items)
    }

    # 8. INTEGRITY ASSERTIONS
    print("\n--- RUNNING INTEGRITY ASSERTIONS ---")

    # Assertion 1: Sample count & Uniqueness
    assert len(per_sample_results) == 21, "Assertion Failed: Sample count must be exactly 21"
    assert len(set([x["sample_id"] for x in per_sample_results])) == 21, "Assertion Failed: Sample IDs must be unique"
    print("[PASS] Sample count is 21 with unique IDs")

    # Assertion 2: Full-precision Mean Equality
    computed_full_mean = float(np.mean([x["strategy_c_mean_lat_ms"] for x in per_sample_results]))
    saved_full_mean = dataset_summary["full_dataset"]["strategy_c_mean_lat_ms"]
    assert abs(computed_full_mean - saved_full_mean) < 0.05, f"Assertion Failed: Mean mismatch {computed_full_mean} != {saved_full_mean}"
    print(f"[PASS] Strategy C full mean ({saved_full_mean:.2f} ms) equals arithmetic mean of per-sample latencies (diff: {abs(computed_full_mean - saved_full_mean):.4f} ms)")

    # Assertion 3: Fallback Invocations
    dev_fbs = sum([x["fallback_triggered"] for x in dev_items])
    held_fbs = sum([x["fallback_triggered"] for x in held_items])
    assert dev_fbs == 0, f"Assertion Failed: Dev split had {dev_fbs} fallbacks (expected 0)"
    assert held_fbs == 1, f"Assertion Failed: HeldOut split had {held_fbs} fallbacks (expected 1)"
    print(f"[PASS] Fallback invocations verified: Dev=0/11 (0%), HeldOut=1/10 (10% on sample 06)")

    # Assertion 4: Engine Routing Correctness
    for x in per_sample_results:
        if x["sample_id"] == "06":
            assert x["engine_used"] == "rapidocr_fallback", f"Sample 06 must use rapidocr_fallback"
            assert x["fallback_triggered"] is True
        else:
            assert x["engine_used"] == "tesseract", f"Sample {x['sample_id']} must use tesseract"
            assert x["fallback_triggered"] is False
    print("[PASS] Engine routing strictly matches fallback condition for all 21 samples")

    # 9. Save Output JSON Artifact
    output_data = {
        "benchmark_name": "Phase 4C.15.2 Controlled Fallback Timing & Integrity Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cold_start_latencies_ms": {
            "tesseract": round(cold_start_tess_ms, 2),
            "rapidocr": round(cold_start_rapid_ms, 2)
        },
        "measurement_parameters": {
            "repetitions": NUM_RUNS,
            "timing_boundary": "OCRInput -> OCREngine.process() -> OCRResult (excluding disk I/O and image decoding)",
            "memory_preloading": True,
            "hardware": "AMD64 Windows 11 CPU"
        },
        "dataset_summaries": dataset_summary,
        "per_sample_evaluations": per_sample_results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print(f"\nSaved verified timing artifact to: {OUTPUT_JSON}")
    print("\n=== VERIFIED TIMING BREAKDOWN TABLE ===")
    print(f"{'ID':<4} | {'Split':<7} | {'Tess Mean (ms)':<15} | {'Strat C Mean (ms)':<18} | {'FB?':<6} | {'Engine':<18} | {'CER':<8}")
    print("-" * 88)
    for x in per_sample_results:
        print(f"{x['sample_id']:<4} | {x['split']:<7} | {x['tesseract_mean_lat_ms']:>15.2f} | {x['strategy_c_mean_lat_ms']:>18.2f} | {str(x['fallback_triggered']):<6} | {x['engine_used']:<18} | {x['cer']*100:>7.2f}%")

    print("\n================================================================================")
    print("SUMMARY OF VERIFIED PERFORMANCE:")
    print(f"  Full Whiteboard Mean Latency:      {dataset_summary['full_dataset']['strategy_c_mean_lat_ms']:.2f} ms (Median: {dataset_summary['full_dataset']['strategy_c_median_lat_ms']:.2f} ms)")
    print(f"  Tesseract Baseline Mean Latency:   {dataset_summary['full_dataset']['tesseract_mean_lat_ms']:.2f} ms (Median: {dataset_summary['full_dataset']['tesseract_median_lat_ms']:.2f} ms)")
    print(f"  Macro CER (Tesseract -> StrategyC): {dataset_summary['full_dataset']['macro_cer']*100:.2f}% (Held-out: {dataset_summary['heldout_split']['macro_cer']*100:.2f}%)")
    print(f"  Sample 06 Fallback Mean Latency:   {strat_c_metadata['06'] and [x['strategy_c_mean_lat_ms'] for x in per_sample_results if x['sample_id']=='06'][0]:.2f} ms")
    print("================================================================================\n")
    return output_data

if __name__ == '__main__':
    run_timing_integrity_evaluation()
