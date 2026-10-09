import sys, os, time, json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.ai.tesseract_ocr import TesseractOCREngine
from backend.ai.rapid_ocr import RapidOCREngine
from backend.ai.ocr_benchmark import (
    calculate_cer,
    calculate_wer,
    normalize_text
)
from backend.vision.frame import OCRInput

DEV_SAMPLE_IDS = ["01", "02", "03", "04", "05", "07", "08", "09", "10", "11", "17"]
HELDOUT_SAMPLE_IDS = ["06", "12", "13", "14", "15", "16", "18", "19", "20", "21"]
OUTPUT_JSON = "tests/results/phase4c15-engine-selection.json"

def run_selection_experiment():
    print("================================================================================")
    print("PHASE 4C.15 — OCR ENGINE-SELECTION FEASIBILITY EXPERIMENT")
    print("================================================================================")
    
    with open("data/external/whiteboards/metadata.jsonl", 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    dev_records = [r for r in records if r["id"] in DEV_SAMPLE_IDS]
    held_records = [r for r in records if r["id"] in HELDOUT_SAMPLE_IDS]
    
    tess_engine = TesseractOCREngine()
    rapid_engine = RapidOCREngine()
    
    # Pre-execute engines on Dev set to collect base data
    print("Evaluating Dev Split (11 samples)...")
    
    def eval_strategy(strategy_name, records, rule_fn):
        evals = []
        for r in records:
            sample_id = r["id"]
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
                seq_num=int(sample_id) if sample_id.isdigit() else 1
            )
            
            t0 = time.perf_counter()
            res, engine_used, fallback_triggered = rule_fn(tess_engine, rapid_engine, inp)
            t1 = time.perf_counter()
            
            lat_ms = (t1 - t0) * 1000.0
            pred_text = res.full_text
            
            cer = calculate_cer(ref_text, pred_text)
            wer = calculate_wer(ref_text, pred_text)
            exact = (normalize_text(ref_text) == normalize_text(pred_text)) and (len(normalize_text(ref_text)) > 0)
            
            evals.append({
                "sample_id": sample_id,
                "engine_used": engine_used,
                "fallback_triggered": fallback_triggered,
                "cer": round(cer, 4),
                "wer": round(wer, 4),
                "exact_match": exact,
                "latency_ms": round(lat_ms, 2),
                "prediction": pred_text
            })
            
        cers = [e["cer"] for e in evals]
        wers = [e["wer"] for e in evals]
        lats = [e["latency_ms"] for e in evals]
        exacts = [e["exact_match"] for e in evals]
        
        return {
            "strategy_name": strategy_name,
            "sample_count": len(evals),
            "macro_cer": round(float(np.mean(cers)), 4),
            "macro_wer": round(float(np.mean(wers)), 4),
            "mean_latency_ms": round(float(np.mean(lats)), 2),
            "median_latency_ms": round(float(np.median(lats)), 2),
            "exact_match_rate": round(float(np.mean(exacts)), 4),
            "evaluations": evals
        }
        
    # Strategy 1: Tesseract Only
    def rule_tess_only(tess, rapid, inp):
        res = tess.process(inp)
        return res, "tesseract", False

    # Strategy 2: RapidOCR Only
    def rule_rapid_only(tess, rapid, inp):
        res = rapid.process(inp)
        return res, "rapidocr", False

    # Strategy 3: Zero-Detection Fallback (Tess -> Rapid on empty text)
    def rule_zero_detection_fallback(tess, rapid, inp):
        res_tess = tess.process(inp)
        if len(res_tess.regions) == 0 or len(res_tess.full_text.strip()) == 0:
            res_rapid = rapid.process(inp)
            return res_rapid, "rapidocr_fallback", True
        return res_tess, "tesseract", False

    # Strategy 4: Low-Confidence Fallback (Tess -> Rapid if Tess avg word conf < 0.35)
    def rule_low_confidence_fallback(tess, rapid, inp):
        res_tess = tess.process(inp)
        avg_conf = float(np.mean([r.confidence for r in res_tess.regions])) if res_tess.regions else 0.0
        if avg_conf < 0.35 or len(res_tess.full_text.strip()) == 0:
            res_rapid = rapid.process(inp)
            return res_rapid, "rapidocr_fallback", True
        return res_tess, "tesseract", False

    strategies = [
        ("Tesseract_Only", rule_tess_only),
        ("RapidOCR_Only", rule_rapid_only),
        ("Zero_Detection_Fallback", rule_zero_detection_fallback),
        ("Low_Confidence_Fallback", rule_low_confidence_fallback)
    ]
    
    results = {
        "benchmark_name": "Phase 4C.15 OCR Engine Selection Feasibility",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dev_split_evaluation": {},
        "heldout_split_evaluation": {},
        "full_dataset_evaluation": {}
    }
    
    print("\n--- DEVELOPMENT SPLIT (11 samples) ---")
    for s_name, s_fn in strategies:
        res = eval_strategy(s_name, dev_records, s_fn)
        results["dev_split_evaluation"][s_name] = res
        print(f"Strategy: {s_name:<26} | CER: {res['macro_cer']*100:>6.2f}% | WER: {res['macro_wer']*100:>6.2f}% | Latency: {res['mean_latency_ms']:>8.1f}ms")
        
    print("\n--- HELDOUT SPLIT (10 samples) ---")
    for s_name, s_fn in strategies:
        res = eval_strategy(s_name, held_records, s_fn)
        results["heldout_split_evaluation"][s_name] = res
        print(f"Strategy: {s_name:<26} | CER: {res['macro_cer']*100:>6.2f}% | WER: {res['macro_wer']*100:>6.2f}% | Latency: {res['mean_latency_ms']:>8.1f}ms")

    print("\n--- FULL DATASET (21 samples) ---")
    for s_name, s_fn in strategies:
        res = eval_strategy(s_name, records, s_fn)
        results["full_dataset_evaluation"][s_name] = res
        print(f"Strategy: {s_name:<26} | CER: {res['macro_cer']*100:>6.2f}% | WER: {res['macro_wer']*100:>6.2f}% | Latency: {res['mean_latency_ms']:>8.1f}ms")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults written to: {OUTPUT_JSON}")
    print("================================================================================\n")
    return results

if __name__ == '__main__':
    run_selection_experiment()
