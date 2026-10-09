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
import pytesseract


def run_public_dataset_evaluation(dataset_dir="data/external/whiteboards", output_json="tests/results/phase4c6-whiteboard-ocr.json"):
    print("================================================================================")
    print("AI GLASSES — PUBLIC WHITEBOARD DATASET OCR EVALUATION (PHASE 4C.6)")
    print("================================================================================")
    
    meta_path = os.path.join(dataset_dir, "metadata.jsonl")
    if not os.path.exists(meta_path):
        print(f"ERROR: Dataset metadata not found at {meta_path}")
        return None
        
    engine = TesseractOCREngine()
    try:
        tess_ver = str(pytesseract.get_tesseract_version())
    except Exception:
        tess_ver = "unknown"
        
    print(f"Dataset Location:   {dataset_dir}")
    print(f"Dataset Source:     https://huggingface.co/datasets/danielrosehill/Whiteboards")
    print(f"License:            CC BY 4.0")
    print(f"Tesseract Version:  {tess_ver}")
    print("Evaluating baseline recognition on real whiteboard photographs...\n")
    
    records = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    evaluations = []
    
    for r in records:
        img_name = r["file_name"]
        img_path = os.path.join(dataset_dir, img_name)
        sample_id = r.get("id", img_name)
        category = r.get("category", "unknown")
        ref_text = r.get("transcription", "").strip()
        
        if not os.path.exists(img_path):
            evaluations.append({
                "sample_id": sample_id,
                "file_name": img_name,
                "category": category,
                "provenance": "Real Whiteboard Photograph (danielrosehill/Whiteboards)",
                "license": "CC BY 4.0",
                "reference_transcription": ref_text,
                "actual_prediction": "",
                "cer": 1.0,
                "wer": 1.0,
                "exact_match": False,
                "confidence": 0.0,
                "latency_ms": 0.0,
                "dimensions": [0, 0],
                "channels": 0,
                "regions_count": 0,
                "status": "SKIPPED_FILE_NOT_FOUND"
            })
            continue
            
        pil_img = Image.open(img_path).convert('RGB')
        w, h = pil_img.size
        arr = np.array(pil_img, dtype=np.uint8)
        
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
        try:
            res = engine.process(inp)
            t1 = time.perf_counter()
            pred_text = res.full_text
            latency_ms = (t1 - t0) * 1000.0
            status = "SUCCESS"
            avg_conf = float(np.mean([reg.confidence for reg in res.regions])) if res.regions else 0.0
            reg_count = len(res.regions)
        except AIVisionError as e:
            t1 = time.perf_counter()
            pred_text = ""
            latency_ms = (t1 - t0) * 1000.0
            status = f"ENGINE_ERROR: {e}"
            avg_conf = 0.0
            reg_count = 0
            
        cer = calculate_cer(ref_text, pred_text, case_sensitive=False)
        wer = calculate_wer(ref_text, pred_text, case_sensitive=False)
        exact_match = (normalize_text(ref_text) == normalize_text(pred_text))
        
        evaluations.append({
            "sample_id": sample_id,
            "file_name": img_name,
            "category": category,
            "provenance": "Real Whiteboard Photograph (danielrosehill/Whiteboards)",
            "license": "CC BY 4.0",
            "reference_transcription": ref_text,
            "actual_prediction": pred_text,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact_match": exact_match,
            "confidence": round(avg_conf, 4),
            "latency_ms": round(latency_ms, 2),
            "dimensions": [w, h],
            "channels": 3,
            "regions_count": reg_count,
            "status": status
        })
        
    # Aggregate Metrics
    successful_evals = [e for e in evaluations if e["status"] == "SUCCESS"]
    
    # Category summary
    categories = sorted(list(set(e["category"] for e in evaluations)))
    cat_summary = {}
    for cat in categories:
        cat_items = [e for e in evaluations if e["category"] == cat]
        cat_summary[cat] = {
            "sample_count": len(cat_items),
            "mean_cer": round(float(np.mean([e["cer"] for e in cat_items])), 4),
            "mean_wer": round(float(np.mean([e["wer"] for e in cat_items])), 4),
            "exact_match_rate": round(float(np.mean([1.0 if e["exact_match"] else 0.0 for e in cat_items])), 4),
            "mean_latency_ms": round(float(np.mean([e["latency_ms"] for e in cat_items])), 2)
        }
        
    overall_summary = {
        "dataset_name": "danielrosehill/Whiteboards",
        "dataset_url": "https://huggingface.co/datasets/danielrosehill/Whiteboards",
        "license": "CC BY 4.0",
        "total_samples": len(evaluations),
        "evaluated_samples": len(successful_evals),
        "skipped_samples": len(evaluations) - len(successful_evals),
        "engine_name": "TesseractOCREngine",
        "tesseract_version": tess_ver,
        "mean_cer": round(float(np.mean([e["cer"] for e in evaluations])), 4),
        "mean_wer": round(float(np.mean([e["wer"] for e in evaluations])), 4),
        "exact_match_count": sum(1 for e in evaluations if e["exact_match"]),
        "exact_match_rate": round(float(np.mean([1.0 if e["exact_match"] else 0.0 for e in evaluations])), 4),
        "mean_latency_ms": round(float(np.mean([e["latency_ms"] for e in evaluations])), 2),
        "median_latency_ms": round(float(np.median([e["latency_ms"] for e in evaluations])), 2),
        "category_breakdown": cat_summary,
        "sample_evaluations": evaluations
    }
    
    # Save output JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=4)
        
    # Print Console Table
    print(f"{'ID':<4} | {'Category':<12} | {'Dims (WxH)':<12} | {'CER':<7} | {'WER':<7} | {'Latency':<9} | {'Regions':<8} | {'Status'}")
    print("-" * 85)
    for e in evaluations:
        dims_str = f"{e['dimensions'][0]}x{e['dimensions'][1]}"
        print(f"{e['sample_id']:<4} | {e['category']:<12} | {dims_str:<12} | {e['cer']:<7.4f} | {e['wer']:<7.4f} | {e['latency_ms']:<7.1f}ms | {e['regions_count']:<8} | {e['status']}")
        
    print("\n" + "=" * 80)
    print("CATEGORY SUMMARY")
    print("=" * 80)
    print(f"{'Category':<15} | {'Samples':<8} | {'Mean CER':<10} | {'Mean WER':<10} | {'Mean Latency'}")
    print("-" * 80)
    for cat, data in cat_summary.items():
        print(f"{cat:<15} | {data['sample_count']:<8} | {data['mean_cer']:<10.4f} | {data['mean_wer']:<10.4f} | {data['mean_latency_ms']:.1f} ms")
        
    print("-" * 80)
    print(f"{'OVERALL':<15} | {overall_summary['total_samples']:<8} | {overall_summary['mean_cer']:<10.4f} | {overall_summary['mean_wer']:<10.4f} | {overall_summary['mean_latency_ms']:.1f} ms")
    print(f"Median Latency: {overall_summary['median_latency_ms']:.1f} ms")
    print(f"Results saved to: {output_json}\n")
    
    return overall_summary


if __name__ == "__main__":
    run_public_dataset_evaluation()
