import sys, os, time, json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.ai.tesseract_ocr import TesseractOCREngine
from backend.ai.ocr_benchmark import (
    calculate_cer,
    calculate_wer,
    normalize_text
)
from backend.vision.frame import OCRInput
import pytesseract
import shutil

# Ensure pytesseract points to valid executable
if not shutil.which("tesseract"):
    default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_path):
        pytesseract.pytesseract.tesseract_cmd = default_path

DATASET_DIR = "data/external/whiteboards"
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
OUTPUT_JSON = "tests/results/phase4c13-controlled-whiteboard-benchmark.json"

# Fixed deterministic splits
DEV_SAMPLE_IDS = ["01", "02", "03", "04", "05", "07", "08", "09", "10", "11", "17"]
HELDOUT_SAMPLE_IDS = ["06", "12", "13", "14", "15", "16", "18", "19", "20", "21"]

def preprocess_image(pil_img, mode):
    w, h = pil_img.size
    if "resize_2048" in mode:
        max_dim = 2048
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h
            
    if "gray_clahe" in mode:
        gray = pil_img.convert('L')
        # CLAHE proxy using PIL autocontrast
        from PIL import ImageOps
        gray = ImageOps.autocontrast(gray, cutoff=2)
        arr = np.array(gray, dtype=np.uint8)
        return arr, w, h, 1
    else:
        arr = np.array(pil_img.convert('RGB'), dtype=np.uint8)
        return arr, w, h, 3

def run_controlled_benchmark():
    print("================================================================================")
    print("PHASE 4C.13 — CONTROLLED WHITEBOARD OCR REPEATABILITY BENCHMARK")
    print("================================================================================")
    
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        all_records = [json.loads(line) for line in f if line.strip()]
        
    dev_records = [r for r in all_records if r["id"] in DEV_SAMPLE_IDS]
    heldout_records = [r for r in all_records if r["id"] in HELDOUT_SAMPLE_IDS]
    
    print(f"Dev samples count:     {len(dev_records)} (IDs: {DEV_SAMPLE_IDS})")
    print(f"Held-out count:        {len(heldout_records)} (IDs: {HELDOUT_SAMPLE_IDS})")
    
    configurations = [
        {"id": "native_psm3", "label": "Native 4K, PSM 3 (Auto)", "psm": "--psm 3", "mode": "native"},
        {"id": "native_psm6", "label": "Native 4K, PSM 6 (Single Block)", "psm": "--psm 6", "mode": "native"},
        {"id": "resize2048_psm3", "label": "Max Dim 2048, PSM 3 (Auto)", "psm": "--psm 3", "mode": "resize_2048"},
        {"id": "resize2048_gray_clahe", "label": "Max Dim 2048, Grayscale CLAHE, PSM 6", "psm": "--psm 6", "mode": "resize_2048_gray_clahe"}
    ]
    
    results = {
        "benchmark_name": "Phase 4C.13 Controlled Whiteboard Benchmark",
        "tesseract_version": str(pytesseract.get_tesseract_version()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dev_sample_ids": DEV_SAMPLE_IDS,
        "heldout_sample_ids": HELDOUT_SAMPLE_IDS,
        "configurations": []
    }
    
    for cfg in configurations:
        cfg_id = cfg["id"]
        cfg_label = cfg["label"]
        psm_str = cfg["psm"]
        mode = cfg["mode"]
        
        print(f"\nEvaluating configuration: {cfg_label} (3 repetitions on 11 dev samples)...")
        engine = TesseractOCREngine(config=psm_str)
        
        repetition_runs = [] # 3 runs
        
        for run_idx in range(3):
            run_type = "cold_start" if run_idx == 0 else f"warm_run_{run_idx}"
            t_run_start = time.perf_counter()
            sample_evals = []
            
            for r in dev_records:
                sample_id = r["id"]
                img_path = os.path.join(DATASET_DIR, r["file_name"])
                ref_text = r.get("transcription", "")
                
                with Image.open(img_path) as pil_img:
                    arr, w, h, c = preprocess_image(pil_img, mode)
                    
                inp = OCRInput(
                    image=arr,
                    width=w,
                    height=h,
                    channels=c,
                    numerical_range=(0, 255),
                    timestamp=1000,
                    seq_num=int(sample_id)
                )
                
                t0 = time.perf_counter()
                res = engine.process(inp)
                t1 = time.perf_counter()
                
                pred_text = res.full_text
                lat_ms = (t1 - t0) * 1000.0
                
                cer = calculate_cer(ref_text, pred_text)
                wer = calculate_wer(ref_text, pred_text)
                exact = (normalize_text(ref_text) == normalize_text(pred_text)) and (len(normalize_text(ref_text)) > 0)
                
                sample_evals.append({
                    "sample_id": sample_id,
                    "cer": round(cer, 4),
                    "wer": round(wer, 4),
                    "exact_match": exact,
                    "latency_ms": round(lat_ms, 2),
                    "dimensions": [w, h],
                    "prediction": pred_text
                })
                
            t_run_end = time.perf_counter()
            run_total_ms = (t_run_end - t_run_start) * 1000.0
            
            run_cers = [e["cer"] for e in sample_evals]
            run_wers = [e["wer"] for e in sample_evals]
            run_lats = [e["latency_ms"] for e in sample_evals]
            
            repetition_runs.append({
                "run_index": run_idx,
                "run_type": run_type,
                "mean_cer": round(float(np.mean(run_cers)), 4),
                "mean_wer": round(float(np.mean(run_wers)), 4),
                "mean_latency_ms": round(float(np.mean(run_lats)), 2),
                "median_latency_ms": round(float(np.median(run_lats)), 2),
                "total_run_time_ms": round(run_total_ms, 2),
                "sample_evaluations": sample_evals
            })
            print(f"  Run {run_idx+1} ({run_type:<10}): Mean Latency = {np.mean(run_lats):.2f} ms | Mean CER = {np.mean(run_cers)*100:.2f}% | Mean WER = {np.mean(run_wers)*100:.2f}%")
            
        all_mean_lats = [r["mean_latency_ms"] for r in repetition_runs]
        warm_mean_lats = [r["mean_latency_ms"] for r in repetition_runs[1:]] # warm runs
        
        cfg_summary = {
            "config_id": cfg_id,
            "label": cfg_label,
            "psm": psm_str,
            "preprocessing_mode": mode,
            "mean_cer": repetition_runs[0]["mean_cer"],
            "mean_wer": repetition_runs[0]["mean_wer"],
            "cold_start_latency_ms": repetition_runs[0]["mean_latency_ms"],
            "warm_mean_latency_ms": round(float(np.mean(warm_mean_lats)), 2),
            "overall_median_latency_ms": round(float(np.median([e["latency_ms"] for r in repetition_runs for e in r["sample_evaluations"]])), 2),
            "repetition_runs": repetition_runs
        }
        results["configurations"].append(cfg_summary)
        
    # Evaluate chosen candidate (native_psm6) once on Held-out split
    print("\nEvaluating Candidate (native_psm6) once on Held-Out Split (10 samples)...")
    cand_engine = TesseractOCREngine(config="--psm 6")
    heldout_evals = []
    for r in heldout_records:
        sample_id = r["id"]
        img_path = os.path.join(DATASET_DIR, r["file_name"])
        ref_text = r.get("transcription", "")
        with Image.open(img_path) as pil_img:
            arr, w, h, c = preprocess_image(pil_img, "native")
        inp = OCRInput(
            image=arr,
            width=w,
            height=h,
            channels=c,
            numerical_range=(0, 255),
            timestamp=1000,
            seq_num=int(sample_id)
        )
        t0 = time.perf_counter()
        res = cand_engine.process(inp)
        t1 = time.perf_counter()
        pred_text = res.full_text
        lat_ms = (t1 - t0) * 1000.0
        cer = calculate_cer(ref_text, pred_text)
        wer = calculate_wer(ref_text, pred_text)
        heldout_evals.append({
            "sample_id": sample_id,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "latency_ms": round(lat_ms, 2),
            "prediction": pred_text
        })
        
    heldout_cers = [e["cer"] for e in heldout_evals]
    heldout_wers = [e["wer"] for e in heldout_evals]
    heldout_lats = [e["latency_ms"] for e in heldout_evals]
    
    results["heldout_candidate_evaluation"] = {
        "sample_count": len(heldout_evals),
        "mean_cer": round(float(np.mean(heldout_cers)), 4),
        "mean_wer": round(float(np.mean(heldout_wers)), 4),
        "mean_latency_ms": round(float(np.mean(heldout_lats)), 2),
        "median_latency_ms": round(float(np.median(heldout_lats)), 2),
        "evaluations": heldout_evals
    }
    
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nHeldout Evaluation: Mean CER = {np.mean(heldout_cers)*100:.2f}% | Mean Latency = {np.mean(heldout_lats):.2f} ms")
    print(f"Results written to: {OUTPUT_JSON}")
    print("================================================================================\n")
    return results

if __name__ == '__main__':
    run_controlled_benchmark()
