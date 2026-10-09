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
import pytesseract

DEV_SAMPLE_IDS = ["01", "02", "03", "04", "05", "07", "08", "09", "10", "11", "17"]
HELDOUT_SAMPLE_IDS = ["06", "12", "13", "14", "15", "16", "18", "19", "20", "21"]

OUTPUT_JSON = "tests/results/phase4c14-rapidocr-comparison.json"

def evaluate_dataset_with_engine(engine, dataset_name, meta_path, data_dir, sample_filter=None):
    with open(meta_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    if sample_filter:
        records = [r for r in records if r["id"] in sample_filter]
        
    evaluations = []
    
    total_ref_chars = 0
    total_char_edits = 0
    total_ref_words = 0
    total_word_edits = 0
    
    for r in records:
        sample_id = r["id"]
        img_path = os.path.join(data_dir, r["file_name"])
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
        res = engine.process(inp)
        t1 = time.perf_counter()
        
        pred_text = res.full_text
        latency_ms = (t1 - t0) * 1000.0
        
        norm_ref = normalize_text(ref_text)
        norm_pred = normalize_text(pred_text)
        
        cer = calculate_cer(ref_text, pred_text)
        wer = calculate_wer(ref_text, pred_text)
        exact_match = (norm_ref == norm_pred) and (len(norm_ref) > 0)
        
        # Calculate raw edit distance counts for micro-averaging
        ref_char_len = max(len(norm_ref), 1)
        char_dist = int(round(cer * ref_char_len))
        ref_word_len = max(len(norm_ref.split()), 1)
        word_dist = int(round(wer * ref_word_len))
        
        total_ref_chars += ref_char_len
        total_char_edits += char_dist
        total_ref_words += ref_word_len
        total_word_edits += word_dist
        
        evaluations.append({
            "sample_id": sample_id,
            "file_name": r.get("file_name"),
            "category": r.get("category", "unknown"),
            "reference_text": ref_text,
            "prediction": pred_text,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact_match": exact_match,
            "latency_ms": round(latency_ms, 2),
            "dimensions": [w, h],
            "regions_count": len(res.regions)
        })
        
    cers = [e["cer"] for e in evaluations]
    wers = [e["wer"] for e in evaluations]
    lats = [e["latency_ms"] for e in evaluations]
    exacts = [e["exact_match"] for e in evaluations]
    
    macro_cer = float(np.mean(cers)) if cers else 0.0
    macro_wer = float(np.mean(wers)) if wers else 0.0
    micro_cer = float(total_char_edits / total_ref_chars) if total_ref_chars > 0 else 0.0
    micro_wer = float(total_word_edits / total_ref_words) if total_ref_words > 0 else 0.0
    
    return {
        "dataset_name": dataset_name,
        "sample_count": len(evaluations),
        "macro_cer": round(macro_cer, 4),
        "macro_wer": round(macro_wer, 4),
        "micro_cer": round(micro_cer, 4),
        "micro_wer": round(micro_wer, 4),
        "exact_match_count": int(sum(exacts)),
        "exact_match_rate": round(float(np.mean(exacts)), 4) if exacts else 0.0,
        "mean_latency_ms": round(float(np.mean(lats)), 2) if lats else 0.0,
        "median_latency_ms": round(float(np.median(lats)), 2) if lats else 0.0,
        "sample_evaluations": evaluations
    }


def run_comprehensive_comparison():
    print("================================================================================")
    print("PHASE 4C.14 — CONTROLLED TESSERACT VS RAPIDOCR COMPARISON BENCHMARK")
    print("================================================================================")
    
    # 1. Instantiate Engines & Measure Cold-Start Latency
    print("Initializing Tesseract OCR Engine...")
    t0_tess = time.perf_counter()
    tess_engine = TesseractOCREngine()
    t1_tess = time.perf_counter()
    tess_cold_ms = (t1_tess - t0_tess) * 1000.0
    print(f"Tesseract initialized in {tess_cold_ms:.2f} ms")
    
    print("\nInitializing RapidOCR ONNX Engine (loading DBNet + SVTR ONNX weights)...")
    t0_rapid = time.perf_counter()
    rapid_engine = RapidOCREngine()
    t1_rapid = time.perf_counter()
    rapid_cold_ms = (t1_rapid - t0_rapid) * 1000.0
    print(f"RapidOCR initialized in {rapid_cold_ms:.2f} ms")
    
    datasets = [
        ("Whiteboard_Dev_11", "data/external/whiteboards/metadata.jsonl", "data/external/whiteboards", DEV_SAMPLE_IDS),
        ("Whiteboard_Heldout_10", "data/external/whiteboards/metadata.jsonl", "data/external/whiteboards", HELDOUT_SAMPLE_IDS),
        ("Whiteboard_Full_21", "data/external/whiteboards/metadata.jsonl", "data/external/whiteboards", None),
        ("Printed_Scene_15", "data/external/printed_scene/metadata.jsonl", "data/external/printed_scene", None),
        ("Handwriting_12", "data/external/handwriting/metadata.jsonl", "data/external/handwriting", None)
    ]
    
    comparison_summary = {
        "benchmark_name": "Phase 4C.14 Tesseract vs RapidOCR Comparison",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tesseract_version": str(pytesseract.get_tesseract_version()),
        "rapidocr_version": "1.2.3 (onnxruntime 1.31.0)",
        "cold_start_latencies_ms": {
            "tesseract": round(tess_cold_ms, 2),
            "rapidocr": round(rapid_cold_ms, 2)
        },
        "datasets": {}
    }
    
    for name, meta_path, data_dir, sample_filter in datasets:
        print(f"\n--------------------------------------------------------------------------------")
        print(f"EVALUATING DATASET: {name}")
        print(f"--------------------------------------------------------------------------------")
        
        print(f"Running Tesseract on {name}...")
        tess_res = evaluate_dataset_with_engine(tess_engine, name, meta_path, data_dir, sample_filter)
        print(f"  Tesseract -> Macro CER: {tess_res['macro_cer']*100:.2f}% | Macro WER: {tess_res['macro_wer']*100:.2f}% | Latency: {tess_res['mean_latency_ms']:.1f}ms")
        
        print(f"Running RapidOCR on {name}...")
        rapid_res = evaluate_dataset_with_engine(rapid_engine, name, meta_path, data_dir, sample_filter)
        print(f"  RapidOCR  -> Macro CER: {rapid_res['macro_cer']*100:.2f}% | Macro WER: {rapid_res['macro_wer']*100:.2f}% | Latency: {rapid_res['mean_latency_ms']:.1f}ms")
        
        cer_delta = rapid_res['macro_cer'] - tess_res['macro_cer']
        speedup = tess_res['mean_latency_ms'] / rapid_res['mean_latency_ms'] if rapid_res['mean_latency_ms'] > 0 else 1.0
        
        comparison_summary["datasets"][name] = {
            "tesseract": tess_res,
            "rapidocr": rapid_res,
            "cer_improvement_absolute": round(-cer_delta, 4),
            "speedup_ratio": round(speedup, 2)
        }
        
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, indent=4)
        
    print("\n================================================================================")
    print("COMPARISON SUMMARY ACROSS ALL DOMAINS:")
    print("================================================================================")
    print(f"{'Dataset':<24} | {'Tesseract CER':<14} | {'RapidOCR CER':<14} | {'CER Gain':<10} | {'Tess Lat':<10} | {'Rapid Lat':<10} | {'Speedup':<8}")
    print("-" * 100)
    for name, data in comparison_summary["datasets"].items():
        t = data["tesseract"]
        r = data["rapidocr"]
        gain = (t["macro_cer"] - r["macro_cer"]) * 100.0
        print(f"{name:<24} | {t['macro_cer']*100:>12.2f}% | {r['macro_cer']*100:>12.2f}% | {gain:>8.2f}% | {t['mean_latency_ms']:>8.1f}ms | {r['mean_latency_ms']:>8.1f}ms | {data['speedup_ratio']:>6.2f}x")
    print("================================================================================")
    print(f"Results written to: {OUTPUT_JSON}\n")
    return comparison_summary

if __name__ == '__main__':
    run_comprehensive_comparison()
