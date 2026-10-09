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
OUTPUT_JSON = "tests/results/phase4c15.1-fallback-timing.json"

def run_isolated_timing_verification():
    print("================================================================================")
    print("PHASE 4C.15.1 — ISOLATED FALLBACK TIMING & REGISTRY VERIFICATION")
    print("================================================================================")

    # 1. Verify Engine Registry instantiation
    registry = AIEngineRegistry()
    tess_engine = TesseractOCREngine()
    rapid_engine = RapidOCREngine()
    registry.register_ocr("tesseract", tess_engine, set_active=True)
    registry.register_ocr("rapidocr", rapid_engine, set_active=False)

    assert registry.active_ocr == "tesseract"
    assert registry.get_ocr() == tess_engine

    with open("data/external/whiteboards/metadata.jsonl", 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Warmup both engines
    warmup_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    warmup_inp = OCRInput(image=warmup_arr, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)
    tess_engine.process(warmup_inp)
    rapid_engine.process(warmup_inp)

    per_sample_results = []

    for r in records:
        s_id = r["id"]
        split = "Dev" if s_id in DEV_SAMPLE_IDS else "HeldOut"
        img_path = os.path.join("data/external/whiteboards", r["file_name"])
        ref_text = r.get("transcription", "").strip()

        with Image.open(img_path) as pil_img:
            pil_rgb = pil_img.convert('RGB')
            w, h = pil_rgb.size
            arr = np.array(pil_rgb, dtype=np.uint8)

        inp = OCRInput(
            image=arr,
            width=w,
            height=h,
            channels=3,
            numerical_range=(0, 255),
            timestamp=1000,
            seq_num=int(s_id) if s_id.isdigit() else 1
        )

        # 1. Measure Tesseract-only in isolation
        t0 = time.perf_counter()
        tess_res = tess_engine.process(inp)
        t1 = time.perf_counter()
        tess_lat = (t1 - t0) * 1000.0

        # 2. Measure RapidOCR-only in isolation
        t0 = time.perf_counter()
        rapid_res = rapid_engine.process(inp)
        t1 = time.perf_counter()
        rapid_lat = (t1 - t0) * 1000.0

        # 3. Measure Strategy C (Zero Detection Fallback) end-to-end
        t0 = time.perf_counter()
        res_tess_c = tess_engine.process(inp)
        fallback_fired = (len(res_tess_c.regions) == 0 or len(res_tess_c.full_text.strip()) == 0)
        if fallback_fired:
            res_c = rapid_engine.process(inp)
            engine_used = "rapidocr_fallback"
        else:
            res_c = res_tess_c
            engine_used = "tesseract"
        t1 = time.perf_counter()
        strat_c_lat = (t1 - t0) * 1000.0

        # Calculated combined fallback latency (Tess + Rapid)
        combined_fallback_lat = tess_lat + rapid_lat

        cer_tess = calculate_cer(ref_text, tess_res.full_text)
        wer_tess = calculate_wer(ref_text, tess_res.full_text)
        cer_rapid = calculate_cer(ref_text, rapid_res.full_text)
        wer_rapid = calculate_wer(ref_text, rapid_res.full_text)
        cer_strat_c = calculate_cer(ref_text, res_c.full_text)
        wer_strat_c = calculate_wer(ref_text, res_c.full_text)

        per_sample_results.append({
            "sample_id": s_id,
            "split": split,
            "file_name": r["file_name"],
            "dimensions": [w, h],
            "tesseract_only_lat_ms": round(tess_lat, 2),
            "rapidocr_only_lat_ms": round(rapid_lat, 2),
            "fallback_fired": fallback_fired,
            "engine_used": engine_used,
            "combined_fallback_lat_ms": round(combined_fallback_lat, 2),
            "strategy_c_lat_ms": round(strat_c_lat, 2),
            "tesseract_cer": round(cer_tess, 4),
            "tesseract_wer": round(wer_tess, 4),
            "rapidocr_cer": round(cer_rapid, 4),
            "rapidocr_wer": round(wer_rapid, 4),
            "strategy_c_cer": round(cer_strat_c, 4),
            "strategy_c_wer": round(wer_strat_c, 4),
            "reference_text": ref_text,
            "strategy_c_prediction": res_c.full_text
        })

    # Summary statistics
    tess_lats = [x["tesseract_only_lat_ms"] for x in per_sample_results]
    rapid_lats = [x["rapidocr_only_lat_ms"] for x in per_sample_results]
    strat_c_lats = [x["strategy_c_lat_ms"] for x in per_sample_results]
    fallbacks = [x["fallback_fired"] for x in per_sample_results]

    dev_samples = [x for x in per_sample_results if x["split"] == "Dev"]
    held_samples = [x for x in per_sample_results if x["split"] == "HeldOut"]

    summary = {
        "benchmark_name": "Phase 4C.15.1 Isolated Fallback Timing Verification",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "registry_active_ocr": registry.active_ocr,
        "sample_count": len(per_sample_results),
        "full_dataset_summary": {
            "tesseract_mean_lat_ms": round(float(np.mean(tess_lats)), 2),
            "tesseract_median_lat_ms": round(float(np.median(tess_lats)), 2),
            "rapidocr_mean_lat_ms": round(float(np.mean(rapid_lats)), 2),
            "rapidocr_median_lat_ms": round(float(np.median(rapid_lats)), 2),
            "strategy_c_mean_lat_ms": round(float(np.mean(strat_c_lats)), 2),
            "strategy_c_median_lat_ms": round(float(np.median(strat_c_lats)), 2),
            "fallback_rate": round(float(np.mean(fallbacks)), 4),
            "fallback_count": int(sum(fallbacks)),
            "tesseract_macro_cer": round(float(np.mean([x["tesseract_cer"] for x in per_sample_results])), 4),
            "tesseract_macro_wer": round(float(np.mean([x["tesseract_wer"] for x in per_sample_results])), 4),
            "strategy_c_macro_cer": round(float(np.mean([x["strategy_c_cer"] for x in per_sample_results])), 4),
            "strategy_c_macro_wer": round(float(np.mean([x["strategy_c_wer"] for x in per_sample_results])), 4)
        },
        "dev_split_summary": {
            "sample_count": len(dev_samples),
            "fallback_count": int(sum([x["fallback_fired"] for x in dev_samples])),
            "strategy_c_mean_lat_ms": round(float(np.mean([x["strategy_c_lat_ms"] for x in dev_samples])), 2),
            "strategy_c_median_lat_ms": round(float(np.median([x["strategy_c_lat_ms"] for x in dev_samples])), 2),
            "tesseract_macro_cer": round(float(np.mean([x["tesseract_cer"] for x in dev_samples])), 4),
            "strategy_c_macro_cer": round(float(np.mean([x["strategy_c_cer"] for x in dev_samples])), 4)
        },
        "heldout_split_summary": {
            "sample_count": len(held_samples),
            "fallback_count": int(sum([x["fallback_fired"] for x in held_samples])),
            "strategy_c_mean_lat_ms": round(float(np.mean([x["strategy_c_lat_ms"] for x in held_samples])), 2),
            "strategy_c_median_lat_ms": round(float(np.median([x["strategy_c_lat_ms"] for x in held_samples])), 2),
            "tesseract_macro_cer": round(float(np.mean([x["tesseract_cer"] for x in held_samples])), 4),
            "strategy_c_macro_cer": round(float(np.mean([x["strategy_c_cer"] for x in held_samples])), 4)
        },
        "per_sample_evaluations": per_sample_results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nResults written to: {OUTPUT_JSON}")
    print("\n=== PER-SAMPLE TIMING BREAKDOWN ===")
    print(f"{'ID':<4} | {'Split':<7} | {'Tess (ms)':<10} | {'Rapid (ms)':<10} | {'FB?':<6} | {'Combined (ms)':<14} | {'Strat C (ms)':<12} | {'Tess CER':<10} | {'Strat C CER':<12}")
    print("-" * 105)
    for x in per_sample_results:
        print(f"{x['sample_id']:<4} | {x['split']:<7} | {x['tesseract_only_lat_ms']:>10.2f} | {x['rapidocr_only_lat_ms']:>10.2f} | {str(x['fallback_fired']):<6} | {x['combined_fallback_lat_ms']:>14.2f} | {x['strategy_c_lat_ms']:>12.2f} | {x['tesseract_cer']*100:>9.2f}% | {x['strategy_c_cer']*100:>11.2f}%")

    print("================================================================================")
    print(f"Full Dataset Summary:")
    print(f"  Tesseract Mean Latency:  {summary['full_dataset_summary']['tesseract_mean_lat_ms']} ms")
    print(f"  RapidOCR Mean Latency:   {summary['full_dataset_summary']['rapidocr_mean_lat_ms']} ms")
    print(f"  Strategy C Mean Latency: {summary['full_dataset_summary']['strategy_c_mean_lat_ms']} ms (Median: {summary['full_dataset_summary']['strategy_c_median_lat_ms']} ms)")
    print(f"  Fallback Invocations:    {summary['full_dataset_summary']['fallback_count']} / {summary['sample_count']} ({summary['full_dataset_summary']['fallback_rate']*100:.2f}%)")
    print(f"  CER Change:              {summary['full_dataset_summary']['tesseract_macro_cer']*100:.2f}% -> {summary['full_dataset_summary']['strategy_c_macro_cer']*100:.2f}%")
    print("================================================================================")
    return summary

if __name__ == '__main__':
    run_isolated_timing_verification()
