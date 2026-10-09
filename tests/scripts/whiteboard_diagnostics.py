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
from backend.ai.errors import AIVisionError


# Deterministic Split
DEV_SAMPLE_IDS = ["01", "02", "03", "04", "05", "07", "08", "09", "10", "11", "17"]
HELDOUT_SAMPLE_IDS = ["06", "12", "13", "14", "15", "16", "18", "19", "20", "21"]


def preprocess_image(pil_img: Image.Image, config_type: str) -> np.ndarray:
    """Preprocess image according to diagnostic configuration."""
    w, h = pil_img.size
    
    if config_type == "baseline":
        return np.array(pil_img.convert('RGB'), dtype=np.uint8)
        
    elif config_type == "psm11":
        return np.array(pil_img.convert('RGB'), dtype=np.uint8)
        
    elif config_type == "psm6":
        return np.array(pil_img.convert('RGB'), dtype=np.uint8)
        
    elif config_type == "resize_2048":
        max_dim = 2048
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return np.array(pil_img.convert('RGB'), dtype=np.uint8)
        
    elif config_type == "resize_2048_psm11":
        max_dim = 2048
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return np.array(pil_img.convert('RGB'), dtype=np.uint8)
        
    elif config_type == "gray_contrast_resize_psm11":
        # 1. Resize
        max_dim = 2048
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        # 2. Grayscale
        gray = pil_img.convert('L')
        arr = np.array(gray, dtype=np.float32)
        # 3. Contrast stretch
        c_min, c_max = np.min(arr), np.max(arr)
        if c_max > c_min:
            arr = (arr - c_min) / (c_max - c_min) * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)
        
    return np.array(pil_img.convert('RGB'), dtype=np.uint8)


def evaluate_configuration(records, dataset_dir, config_name, psm_config=""):
    engine = TesseractOCREngine(config=psm_config)
    evals = []
    
    for r in records:
        img_path = os.path.join(dataset_dir, r["file_name"])
        ref_text = r["transcription"]
        pil_img = Image.open(img_path)
        
        arr = preprocess_image(pil_img, config_name)
        h, w = arr.shape[:2]
        c = 1 if arr.ndim == 2 else arr.shape[2]
        
        inp = OCRInput(arr, w, h, c, (0, 255), 1000, 1)
        t0 = time.perf_counter()
        try:
            res = engine.process(inp)
            t1 = time.perf_counter()
            pred_text = res.full_text
            lat = (t1 - t0) * 1000.0
            status = "SUCCESS"
        except AIVisionError as e:
            t1 = time.perf_counter()
            pred_text = ""
            lat = (t1 - t0) * 1000.0
            status = f"ERROR: {e}"
            
        cer = calculate_cer(ref_text, pred_text, case_sensitive=False)
        wer = calculate_wer(ref_text, pred_text, case_sensitive=False)
        em = (normalize_text(ref_text) == normalize_text(pred_text))
        
        evals.append({
            "sample_id": r["id"],
            "category": r["category"],
            "ref": ref_text,
            "pred": pred_text,
            "cer": cer,
            "wer": wer,
            "exact_match": em,
            "latency_ms": lat,
            "dimensions": [w, h],
            "channels": c,
            "status": status
        })
        
    mean_cer = float(np.mean([e["cer"] for e in evals]))
    mean_wer = float(np.mean([e["wer"] for e in evals]))
    exact_rate = float(np.mean([1.0 if e["exact_match"] else 0.0 for e in evals]))
    mean_lat = float(np.mean([e["latency_ms"] for e in evals]))
    median_lat = float(np.median([e["latency_ms"] for e in evals]))
    
    return {
        "config_name": config_name,
        "psm_config": psm_config,
        "sample_count": len(evals),
        "mean_cer": round(mean_cer, 4),
        "mean_wer": round(mean_wer, 4),
        "exact_match_rate": round(exact_rate, 4),
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "evaluations": evals
    }


def run_diagnostics(dataset_dir="data/external/whiteboards", output_json="tests/results/phase4c7-whiteboard-diagnostics.json"):
    print("================================================================================")
    print("AI GLASSES — WHITEBOARD OCR DIAGNOSTICS & ENGINE EXPERIMENTS (PHASE 4C.7)")
    print("================================================================================")
    
    meta_path = os.path.join(dataset_dir, "metadata.jsonl")
    records = [json.loads(l) for l in open(meta_path, 'r', encoding='utf-8') if l.strip()]
    
    dev_records = [r for r in records if r["id"] in DEV_SAMPLE_IDS]
    heldout_records = [r for r in records if r["id"] in HELDOUT_SAMPLE_IDS]
    
    print(f"Total Dataset Samples: {len(records)}")
    print(f"Development Set:       {len(dev_records)} samples (IDs: {', '.join(DEV_SAMPLE_IDS)})")
    print(f"Held-Out Test Set:     {len(heldout_records)} samples (IDs: {', '.join(HELDOUT_SAMPLE_IDS)})\n")
    
    # 1. Evaluate Matrix on Development Set
    print("--- 1. DEVELOPMENT SET CONFIGURATION MATRIX ---")
    matrix = [
        ("baseline", "", "1. Baseline (Full 4K, PSM 3, RGB)"),
        ("psm11", "--psm 11", "2. PSM 11 Sparse Text (Full 4K)"),
        ("psm6", "--psm 6", "3. PSM 6 Uniform Block (Full 4K)"),
        ("resize_2048", "", "4. Bounded Resize (2048 max dim, PSM 3)"),
        ("resize_2048_psm11", "--psm 11", "5. Bounded Resize + PSM 11"),
        ("gray_contrast_resize_psm11", "--psm 11", "6. Grayscale + Contrast Norm + Resize + PSM 11")
    ]
    
    dev_results = []
    for cfg_id, psm, label in matrix:
        res = evaluate_configuration(dev_records, dataset_dir, cfg_id, psm)
        res["label"] = label
        dev_results.append(res)
        print(f"{label:<55} | CER: {res['mean_cer']:.4f} | WER: {res['mean_wer']:.4f} | Latency: {res['mean_latency_ms']:.1f} ms")

    # Select Candidate (Best CER on Dev Set)
    best_candidate = min(dev_results, key=lambda x: x["mean_cer"])
    print(f"\nSelected Best Candidate from Dev Set: {best_candidate['label']} (CER: {best_candidate['mean_cer']:.4f})")
    
    # 2. Run Candidate on Held-Out Test Set
    print("\n--- 2. HELD-OUT TEST SET EVALUATION ---")
    heldout_baseline = evaluate_configuration(heldout_records, dataset_dir, "baseline", "")
    heldout_candidate = evaluate_configuration(heldout_records, dataset_dir, best_candidate["config_name"], best_candidate["psm_config"])
    
    print(f"{'Held-Out Baseline (Full 4K, PSM 3)':<55} | CER: {heldout_baseline['mean_cer']:.4f} | WER: {heldout_baseline['mean_wer']:.4f} | Latency: {heldout_baseline['mean_latency_ms']:.1f} ms")
    print(f"{'Held-Out Best Candidate (' + best_candidate['config_name'] + ')':<55} | CER: {heldout_candidate['mean_cer']:.4f} | WER: {heldout_candidate['mean_wer']:.4f} | Latency: {heldout_candidate['mean_latency_ms']:.1f} ms")
    
    # 3. Overall 21-sample Comparison
    all_baseline = evaluate_configuration(records, dataset_dir, "baseline", "")
    all_candidate = evaluate_configuration(records, dataset_dir, best_candidate["config_name"], best_candidate["psm_config"])
    
    print("\n" + "=" * 80)
    print("FULL 21-SAMPLE OVERALL SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Configuration':<50} | {'Mean CER':<10} | {'Mean WER':<10} | {'Mean Latency':<12}")
    print("-" * 80)
    print(f"{'Phase 4C.6 Baseline (Default PSM 3, Full 4K)':<50} | {all_baseline['mean_cer']:<10.4f} | {all_baseline['mean_wer']:<10.4f} | {all_baseline['mean_latency_ms']:.1f} ms")
    print(f"{'Phase 4C.7 Candidate (' + best_candidate['label'] + ')':<50} | {all_candidate['mean_cer']:<10.4f} | {all_candidate['mean_wer']:<10.4f} | {all_candidate['mean_latency_ms']:.1f} ms")
    print("=" * 80)
    
    diagnostics_data = {
        "phase": "Phase 4C.7",
        "dataset": "danielrosehill/Whiteboards",
        "splits": {
            "dev_sample_ids": DEV_SAMPLE_IDS,
            "heldout_sample_ids": HELDOUT_SAMPLE_IDS
        },
        "dev_matrix_results": dev_results,
        "selected_candidate": {
            "config_name": best_candidate["config_name"],
            "psm_config": best_candidate["psm_config"],
            "label": best_candidate["label"]
        },
        "heldout_results": {
            "baseline": heldout_baseline,
            "candidate": heldout_candidate
        },
        "full_comparison": {
            "baseline": all_baseline,
            "candidate": all_candidate
        }
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(diagnostics_data, f, indent=4)
        
    print(f"\nSaved machine-readable diagnostic results to: {output_json}")
    return diagnostics_data


if __name__ == "__main__":
    run_diagnostics()
