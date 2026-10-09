import os
import json
import pytest

RESULT_PATH = "tests/results/phase4c18-handwriting-model-evaluation.json"

@pytest.fixture(scope="module")
def benchmark_data():
    if not os.path.exists(RESULT_PATH):
        pytest.skip(f"Benchmark results not found at {RESULT_PATH}")
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def test_result_schema_and_keys(benchmark_data):
    assert "benchmark_name" in benchmark_data
    assert "model_investigation_audit" in benchmark_data
    assert "cold_start_latencies_ms" in benchmark_data
    assert "dataset_summaries" in benchmark_data
    assert "per_sample_evaluations" in benchmark_data

def test_model_investigation_audit_contents(benchmark_data):
    audit = benchmark_data["model_investigation_audit"]
    assert "microsoft/trocr-small-handwritten" in audit["candidate_model"]
    assert "intended_task" in audit
    assert "PyTorch" in audit["framework_requirement"]
    assert "INCOMPATIBLE" in audit["environment_compatibility"]
    assert "RapidOCR" in audit["active_onnx_alternative"]

def test_per_sample_evaluations_count_and_splits(benchmark_data):
    samples = benchmark_data["per_sample_evaluations"]
    assert len(samples) == 30
    
    dev_samples = [s for s in samples if s["split"] == "Dev"]
    heldout_samples = [s for s in samples if s["split"] == "HeldOut"]
    assert len(dev_samples) == 15
    assert len(heldout_samples) == 15

def test_pipeline_metrics_integrity(benchmark_data):
    summaries = benchmark_data["dataset_summaries"]
    for split_key in ["full_dataset", "dev_split", "heldout_split"]:
        assert split_key in summaries
        split_data = summaries[split_key]
        for pipe in ["tesseract_fullpage", "rapidocr_fullpage", "rapidocr_line_clustered"]:
            assert pipe in split_data
            metrics = split_data[pipe]
            assert "macro_cer" in metrics
            assert "macro_wer" in metrics
            assert "micro_cer" in metrics
            assert "micro_wer" in metrics
            assert "mean_latency_ms" in metrics
            assert "median_latency_ms" in metrics
            assert metrics["macro_cer"] >= 0.0
            assert metrics["macro_wer"] >= 0.0
            assert metrics["mean_latency_ms"] > 0.0

def test_heldout_rapidocr_superiority_over_tesseract(benchmark_data):
    heldout = benchmark_data["dataset_summaries"]["heldout_split"]
    tess_cer = heldout["tesseract_fullpage"]["macro_cer"]
    rapid_clustered_cer = heldout["rapidocr_line_clustered"]["macro_cer"]
    # Verify line-clustered RapidOCR outperforms Tesseract on handwritten notes
    assert rapid_clustered_cer < tess_cer
