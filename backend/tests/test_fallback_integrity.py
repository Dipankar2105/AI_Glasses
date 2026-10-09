import os, sys, json
import numpy as np
import pytest

def test_fallback_benchmark_integrity():
    timing_file = os.path.join(os.path.dirname(__file__), "../../tests/results/phase4c15.2-timing.json")
    assert os.path.exists(timing_file), f"Timing result file not found at {timing_file}"

    with open(timing_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = data["per_sample_evaluations"]
    assert len(samples) == 21, f"Expected 21 samples, got {len(samples)}"
    
    # 1. Check uniqueness and completeness of sample IDs
    ids = [s["sample_id"] for s in samples]
    assert len(set(ids)) == 21, "Duplicate sample IDs found"
    for i in range(1, 22):
        assert f"{i:02d}" in ids, f"Missing sample ID {i:02d}"

    # 2. Check full precision arithmetic consistency
    strat_c_means = [s["strategy_c_mean_lat_ms"] for s in samples]
    computed_mean = float(np.mean(strat_c_means))
    reported_mean = data["dataset_summaries"]["full_dataset"]["strategy_c_mean_lat_ms"]
    assert abs(computed_mean - reported_mean) < 0.05, f"Mean mismatch: {computed_mean} vs {reported_mean}"

    # 3. Check fallback trigger correctness
    dev_fbs = [s for s in samples if s["split"] == "Dev" and s["fallback_triggered"]]
    held_fbs = [s for s in samples if s["split"] == "HeldOut" and s["fallback_triggered"]]
    assert len(dev_fbs) == 0, "Zero fallbacks allowed on Dev split"
    assert len(held_fbs) == 1, "Expected exactly 1 fallback on Held-out split"
    assert held_fbs[0]["sample_id"] == "06", "Sample 06 must be the only fallback sample"
    assert held_fbs[0]["engine_used"] == "rapidocr_fallback"

    # 4. Check accuracy replay correctness
    cers = [s["cer"] for s in samples]
    full_macro_cer = float(np.mean(cers))
    assert abs(full_macro_cer - 0.6754) < 1e-3, f"CER replay mismatch: {full_macro_cer}"
