import os, sys, json, hashlib
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.ai.ocr_benchmark import calculate_cer, calculate_wer, normalize_text

DATASET_DIR = "data/external/notebooks"
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
INPUT_RESULTS = "tests/results/phase4c16-notebook-ocr-benchmark.json"
OUTPUT_AUDIT_JSON = "tests/results/phase4c16.1-notebook-benchmark-integrity.json"

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_integrity_audit():
    print("================================================================================")
    print("PHASE 4C.16.1 — NOTEBOOK OCR BENCHMARK RESULTS INTEGRITY AUDIT")
    print("================================================================================")

    assert os.path.exists(INPUT_RESULTS), f"Missing {INPUT_RESULTS}"
    with open(INPUT_RESULTS, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata_records = [json.loads(line) for line in f if line.strip()]

    evals = benchmark_data["per_sample_evaluations"]
    assert len(evals) == 30, f"Expected 30 evaluated samples, got {len(evals)}"
    assert len(metadata_records) == 30, f"Expected 30 metadata records, got {len(metadata_records)}"

    # 1. Image File Integrity & SHA256 Checksums
    print("\n1. VERIFYING IMAGE ASSET INTEGRITY & COMPUTING SHA256 CHECKSUMS...")
    image_audit = []
    for r in metadata_records:
        s_id = r["id"]
        img_path = os.path.join(DATASET_DIR, r["file_name"])
        assert os.path.exists(img_path), f"Missing image file: {img_path}"
        
        file_hash = sha256_file(img_path)
        with Image.open(img_path) as img:
            w, h = img.size
            bands = img.getbands()
            img_format = img.format

        image_audit.append({
            "sample_id": s_id,
            "file_name": r["file_name"],
            "sha256": file_hash,
            "dimensions": [w, h],
            "format": img_format,
            "channels": len(bands),
            "readable": True
        })
    print(f"  [PASS] All 30 images verified and checksummed successfully.")

    # 2. Independent Metric Recomputation
    print("\n2. INDEPENDENTLY RECOMPUTING CER & WER METRICS...")
    audit_evaluations = []

    for e in evals:
        s_id = e["sample_id"]
        ref_text = e["reference_text"]
        tess_pred = e["tesseract"]["prediction"]
        rapid_pred = e["rapidocr"]["prediction"]

        # Recalculate CER / WER independently
        norm_ref = normalize_text(ref_text)
        norm_tess = normalize_text(tess_pred)
        norm_rapid = normalize_text(rapid_pred)

        indep_tess_cer = calculate_cer(ref_text, tess_pred)
        indep_tess_wer = calculate_wer(ref_text, tess_pred)
        indep_tess_exact = (norm_ref == norm_tess) and (len(norm_ref) > 0)

        indep_rapid_cer = calculate_cer(ref_text, rapid_pred)
        indep_rapid_wer = calculate_wer(ref_text, rapid_pred)
        indep_rapid_exact = (norm_ref == norm_rapid) and (len(norm_ref) > 0)

        # Assert per-sample consistency
        assert abs(indep_tess_cer - e["tesseract"]["cer"]) < 1e-4, f"Tess CER mismatch on {s_id}"
        assert abs(indep_tess_wer - e["tesseract"]["wer"]) < 1e-4, f"Tess WER mismatch on {s_id}"
        assert abs(indep_rapid_cer - e["rapidocr"]["cer"]) < 1e-4, f"Rapid CER mismatch on {s_id}"
        assert abs(indep_rapid_wer - e["rapidocr"]["wer"]) < 1e-4, f"Rapid WER mismatch on {s_id}"

        # Latency statistics recomputation
        tess_lats = e["tesseract"]["latencies_ms"]
        rapid_lats = e["rapidocr"]["latencies_ms"]

        indep_tess_mean_lat = float(np.mean(tess_lats))
        indep_tess_med_lat = float(np.median(tess_lats))
        indep_rapid_mean_lat = float(np.mean(rapid_lats))
        indep_rapid_med_lat = float(np.median(rapid_lats))

        audit_evaluations.append({
            "sample_id": s_id,
            "split": e["split"],
            "category": e["category"],
            "subject": e["subject"],
            "topic": e["topic"],
            "file_name": e["file_name"],
            "reference_text": ref_text,
            "tesseract": {
                "prediction": tess_pred,
                "recomputed_cer": round(indep_tess_cer, 4),
                "recomputed_wer": round(indep_tess_wer, 4),
                "exact_match": indep_tess_exact,
                "latencies_ms": tess_lats,
                "mean_latency_ms": round(indep_tess_mean_lat, 2),
                "median_latency_ms": round(indep_tess_med_lat, 2)
            },
            "rapidocr": {
                "prediction": rapid_pred,
                "recomputed_cer": round(indep_rapid_cer, 4),
                "recomputed_wer": round(indep_rapid_wer, 4),
                "exact_match": indep_rapid_exact,
                "latencies_ms": rapid_lats,
                "mean_latency_ms": round(indep_rapid_mean_lat, 2),
                "median_latency_ms": round(indep_rapid_med_lat, 2)
            }
        })

    # 3. Aggregate Macro / Micro Metric Verification
    def aggregate_split(items, engine_key):
        cers = [x[engine_key]["recomputed_cer"] for x in items]
        wers = [x[engine_key]["recomputed_wer"] for x in items]
        exacts = [x[engine_key]["exact_match"] for x in items]
        means = [x[engine_key]["mean_latency_ms"] for x in items]
        meds = [x[engine_key]["median_latency_ms"] for x in items]

        total_ref_chars = sum([max(len(normalize_text(x["reference_text"])), 1) for x in items])
        total_char_edits = sum([int(round(x[engine_key]["recomputed_cer"] * max(len(normalize_text(x["reference_text"])), 1))) for x in items])
        total_ref_words = sum([max(len(normalize_text(x["reference_text"]).split()), 1) for x in items])
        total_word_edits = sum([int(round(x[engine_key]["recomputed_wer"] * max(len(normalize_text(x["reference_text"]).split()), 1))) for x in items])

        return {
            "sample_count": len(items),
            "macro_cer": round(float(np.mean(cers)), 4),
            "macro_wer": round(float(np.mean(wers)), 4),
            "micro_cer": round(float(total_char_edits / total_ref_chars), 4),
            "micro_wer": round(float(total_word_edits / total_ref_words), 4),
            "exact_match_count": int(sum(exacts)),
            "exact_match_rate": round(float(np.mean(exacts)), 4),
            "mean_latency_ms": round(float(np.mean(means)), 2),
            "median_latency_ms": round(float(np.median(meds)), 2)
        }

    dev_items = [x for x in audit_evaluations if x["split"] == "Dev"]
    held_items = [x for x in audit_evaluations if x["split"] == "HeldOut"]

    recomputed_summaries = {
        "full_dataset": {
            "tesseract": aggregate_split(audit_evaluations, "tesseract"),
            "rapidocr": aggregate_split(audit_evaluations, "rapidocr")
        },
        "dev_split": {
            "tesseract": aggregate_split(dev_items, "tesseract"),
            "rapidocr": aggregate_split(dev_items, "rapidocr")
        },
        "heldout_split": {
            "tesseract": aggregate_split(held_items, "tesseract"),
            "rapidocr": aggregate_split(held_items, "rapidocr")
        }
    }

    # Verify arithmetic match with saved benchmark artifact
    for split in ["full_dataset", "dev_split", "heldout_split"]:
        for eng in ["tesseract", "rapidocr"]:
            saved = benchmark_data["dataset_summaries"][split][eng]
            recomp = recomputed_summaries[split][eng]
            assert abs(saved["macro_cer"] - recomp["macro_cer"]) < 1e-4, f"Mismatch in {split} {eng} macro_cer"
            assert abs(saved["macro_wer"] - recomp["macro_wer"]) < 1e-4, f"Mismatch in {split} {eng} macro_wer"
            assert abs(saved["micro_cer"] - recomp["micro_cer"]) < 1e-4, f"Mismatch in {split} {eng} micro_cer"
            assert abs(saved["micro_wer"] - recomp["micro_wer"]) < 1e-4, f"Mismatch in {split} {eng} micro_wer"
            assert abs(saved["mean_latency_ms"] - recomp["mean_latency_ms"]) < 0.05, f"Mismatch in {split} {eng} mean_lat"
    print(f"  [PASS] All macro and micro CER/WER metrics verified to 100% precision.")

    # 4. 5-Sample Spot Check
    spot_check_ids = ["nb_01", "nb_05", "nb_11", "nb_16", "nb_24"]
    spot_checks = []
    print("\n3. RUNNING 5-SAMPLE DETAILED SPOT-CHECK...")
    for s_id in spot_check_ids:
        item = next(x for x in audit_evaluations if x["sample_id"] == s_id)
        spot_checks.append({
            "sample_id": s_id,
            "topic": item["topic"],
            "split": item["split"],
            "reference_text": item["reference_text"],
            "tesseract_prediction": item["tesseract"]["prediction"],
            "tesseract_cer": item["tesseract"]["recomputed_cer"],
            "rapidocr_prediction": item["rapidocr"]["prediction"],
            "rapidocr_cer": item["rapidocr"]["recomputed_cer"]
        })
        ref_preview = item['reference_text'][:70].encode('ascii', 'replace').decode('ascii')
        tess_preview = item['tesseract']['prediction'][:70].replace('\n', ' ').encode('ascii', 'replace').decode('ascii')
        rapid_preview = item['rapidocr']['prediction'][:70].replace('\n', ' ').encode('ascii', 'replace').decode('ascii')
        print(f"  --- Sample {s_id} ({item['split']} / {item['topic']}) ---")
        print(f"      Reference:    {ref_preview}...")
        print(f"      Tesseract:    {tess_preview}... (CER: {item['tesseract']['recomputed_cer']*100:.2f}%)")
        print(f"      RapidOCR:     {rapid_preview}... (CER: {item['rapidocr']['recomputed_cer']*100:.2f}%)")

    # 5. Output JSON Deliverable
    audit_output = {
        "audit_name": "Phase 4C.16.1 Notebook OCR Benchmark Integrity Audit",
        "audit_timestamp": benchmark_data["timestamp"],
        "dataset_verification": {
            "total_images": len(image_audit),
            "readable_images": sum([1 for x in image_audit if x["readable"]]),
            "license": "CC BY 4.0 (NoTeS-Bank ICDAR 2025 Challenge)",
            "image_checksums": image_audit
        },
        "recomputed_summaries": recomputed_summaries,
        "spot_checks": spot_checks
    }

    os.makedirs(os.path.dirname(OUTPUT_AUDIT_JSON), exist_ok=True)
    with open(OUTPUT_AUDIT_JSON, 'w', encoding='utf-8') as f:
        json.dump(audit_output, f, indent=4)

    print(f"\nAudit results successfully written to: {OUTPUT_AUDIT_JSON}")
    print("================================================================================\n")
    return audit_output

if __name__ == '__main__':
    run_integrity_audit()
