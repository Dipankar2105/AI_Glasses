import os, sys, json
import numpy as np
import pytest

def test_notebook_dataset_manifest_integrity():
    manifest_path = os.path.join(os.path.dirname(__file__), "../../data/external/notebooks/metadata.jsonl")
    assert os.path.exists(manifest_path), f"Missing notebook manifest at {manifest_path}"

    with open(manifest_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 30, f"Expected 30 notebook samples, got {len(records)}"

    # Check sample uniqueness and ID formatting
    ids = [r["id"] for r in records]
    assert len(set(ids)) == 30, "Duplicate sample IDs found in notebook manifest"
    for i in range(1, 31):
        assert f"nb_{i:02d}" in ids, f"Missing sample ID nb_{i:02d}"

    # Check split partitioning
    dev_records = [r for r in records if r["split"] == "Dev"]
    held_records = [r for r in records if r["split"] == "HeldOut"]
    assert len(dev_records) == 15, f"Expected 15 Dev samples, got {len(dev_records)}"
    assert len(held_records) == 15, f"Expected 15 Held-out samples, got {len(held_records)}"

    # Check required metadata fields
    for r in records:
        assert "provenance" in r and len(r["provenance"]) > 0
        assert "license" in r and len(r["license"]) > 0
        assert "transcription" in r and len(r["transcription"]) > 0
        assert "category" in r and len(r["category"]) > 0
        assert "url" in r and r["url"].startswith("http")

def test_notebook_benchmark_results_integrity():
    results_path = os.path.join(os.path.dirname(__file__), "../../tests/results/phase4c16-notebook-ocr-benchmark.json")
    assert os.path.exists(results_path), f"Missing notebook results at {results_path}"

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert "dataset_summaries" in data
    assert "per_sample_evaluations" in data
    evals = data["per_sample_evaluations"]
    assert len(evals) == 30, f"Expected 30 evaluated samples, got {len(evals)}"

    # Check that both engines have valid non-empty evaluations
    for e in evals:
        assert "tesseract" in e
        assert "rapidocr" in e
        assert len(e["tesseract"]["latencies_ms"]) == 3, "Expected 3 repetitions per sample"
        assert len(e["rapidocr"]["latencies_ms"]) == 3, "Expected 3 repetitions per sample"
        assert e["tesseract"]["mean_latency_ms"] > 0
        assert e["rapidocr"]["mean_latency_ms"] > 0

    # Arithmetic consistency check
    full_tess = data["dataset_summaries"]["full_dataset"]["tesseract"]
    full_rapid = data["dataset_summaries"]["full_dataset"]["rapidocr"]

    computed_tess_cer = float(np.mean([e["tesseract"]["cer"] for e in evals]))
    computed_rapid_cer = float(np.mean([e["rapidocr"]["cer"] for e in evals]))

    assert abs(computed_tess_cer - full_tess["macro_cer"]) < 0.01
    assert abs(computed_rapid_cer - full_rapid["macro_cer"]) < 0.01
