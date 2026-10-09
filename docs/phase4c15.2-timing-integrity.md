# Phase 4C.15.2 — Controlled Fallback Timing Integrity & Metric Replay Report

**Date:** 2026-10-09  
**Status:** Verified with Strict Timing Boundaries & Integrity Assertions  
**Starting Checkpoint:** [`5d5c931`](file:///c:/Users/Routewise/AI_Glasses)  
**Evaluated Artifact:** [`tests/results/phase4c15.2-timing.json`](file:///c:/Users/Routewise/AI_Glasses/tests/results/phase4c15.2-timing.json)  
**Execution Script:** [`tests/scripts/evaluate_timing_integrity.py`](file:///c:/Users/Routewise/AI_Glasses/tests/scripts/evaluate_timing_integrity.py)  

---

## 1. Timing Integrity & Methodological Fixes

### A. Resolution of Methodological Artifacts in Earlier Scripts
In earlier experimental scripts, multiple OCR engines were executed sequentially within the same per-image loop to record comparisons. This introduced two measurement distortions:
1. **Unintended Engine Sequencing:** Executing RapidOCR ($6\text{–}8\text{s}$ multi-threaded ONNX load) immediately prior to Strategy C measured Tesseract on a CPU experiencing post-ONNX core throttling.
2. **Timing Boundary Clarity:** In Phase 4C.15.2, all 21 whiteboard images are pre-decoded into memory prior to timing. Timing is bounded strictly between `OCRInput` delivery and `OCRResult` return:
   $$\text{Timing Boundary: } t_0 = \text{perf\_counter}() \rightarrow \text{engine.process(OCRInput)} \rightarrow t_1 = \text{perf\_counter}()$$

### B. Controlled Multi-Pass Measurement (3 Repetitions)
- Warmup executed on synthetic buffer prior to measurement passes.
- 3 independent passes executed for Standalone Tesseract vs True Strategy C path.
- In Strategy C, RapidOCR is invoked **if and only if** Tesseract returns 0 regions or empty text.
- Cold-start measured separately:
  - **Tesseract Cold Start:** `4.86 ms`
  - **RapidOCR Cold Start:** `1,922.14 ms` (ONNX session initialization, weights load, and tensor allocation).

---

## 2. Verified Per-Sample Arithmetic & Timing Decomposition

| Sample ID | Split | Standalone Tesseract Mean (ms) | Strategy C True Path Mean (ms) | Fallback Fired? | Engine Used | CER | WER | Exact Match |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **`01`** | Dev | 1,614.7 ms | 1,433.8 ms | False | `tesseract` | 53.44% | 105.88% | False |
| **`02`** | Dev | 1,388.7 ms | 1,283.8 ms | False | `tesseract` | 63.58% | 106.90% | False |
| **`03`** | Dev | 1,450.5 ms | 1,319.1 ms | False | `tesseract` | 67.32% | 97.96% | False |
| **`04`** | Dev | 1,495.2 ms | 1,379.6 ms | False | `tesseract` | 58.27% | 97.96% | False |
| **`05`** | Dev | 1,431.4 ms | 1,311.6 ms | False | `tesseract` | 60.00% | 121.95% | False |
| **`06`** | **Held-Out** | **1,116.3 ms** | **5,804.4 ms** | **TRUE** | **`rapidocr_fallback`** | **57.78%** | **85.71%** | False |
| **`07`** | Dev | 1,424.6 ms | 1,691.5 ms | False | `tesseract` | 67.97% | 95.24% | False |
| **`08`** | Dev | 1,318.2 ms | 1,215.2 ms | False | `tesseract` | 75.17% | 96.43% | False |
| **`09`** | Dev | 1,657.7 ms | 1,557.5 ms | False | `tesseract` | 62.65% | 95.35% | False |
| **`10`** | Dev | 1,492.6 ms | 1,364.3 ms | False | `tesseract` | 61.07% | 89.29% | False |
| **`11`** | Dev | 1,485.5 ms | 1,467.9 ms | False | `tesseract` | 70.73% | 95.59% | False |
| **`12`** | Held-Out | 1,333.4 ms | 1,330.1 ms | False | `tesseract` | 77.51% | 100.00% | False |
| **`13`** | Held-Out | 1,653.4 ms | 1,663.2 ms | False | `tesseract` | 69.19% | 100.00% | False |
| **`14`** | Held-Out | 1,457.1 ms | 1,460.8 ms | False | `tesseract` | 77.07% | 100.00% | False |
| **`15`** | Held-Out | 1,478.7 ms | 1,492.3 ms | False | `tesseract` | 68.13% | 110.53% | False |
| **`16`** | Held-Out | 1,057.8 ms | 1,057.0 ms | False | `tesseract` | 90.20% | 100.00% | False |
| **`17`** | Dev | 1,249.1 ms | 1,247.6 ms | False | `tesseract` | 65.35% | 93.75% | False |
| **`18`** | Held-Out | 1,282.4 ms | 1,272.6 ms | False | `tesseract` | 73.03% | 120.00% | False |
| **`19`** | Held-Out | 1,257.3 ms | 1,234.0 ms | False | `tesseract` | 73.21% | 100.00% | False |
| **`20`** | Held-Out | 1,459.0 ms | 1,424.8 ms | False | `tesseract` | 74.58% | 100.00% | False |
| **`21`** | Held-Out | 1,420.9 ms | 1,412.7 ms | False | `tesseract` | 52.03% | 93.62% | False |

---

## 3. Mathematical Reconciliation of Aggregates

- **20 Non-Fallback Samples:** Total latency sum = $27,619.24 \text{ ms} \implies \text{Mean} = \mathbf{1,380.96 \text{ ms}}$.
- **1 Fallback Sample (`06`):** Latency = $\mathbf{5,804.36 \text{ ms}}$ (Tesseract attempt + RapidOCR execution).
- **Full Dataset Mathematical Mean:**
  $$\mu = \frac{27,619.24 + 5,804.36}{21} = \frac{33,423.60}{21} = \mathbf{1,591.60 \text{ ms}}$$
- **Full Dataset Median:**
  $$\text{Median} = \mathbf{1,379.60 \text{ ms}} \quad (\text{sample } 04)$$
- **Discrepancy Check:** Computed full-precision mean ($1,591.6005 \text{ ms}$) matches the reported summary mean ($1,591.60 \text{ ms}$) within $0.0005 \text{ ms}$.

---

## 4. Automated Integrity Assertions Passed

```
[PASS] Sample count is 21 with unique IDs
[PASS] Strategy C full mean (1591.60 ms) equals arithmetic mean of per-sample latencies (diff: 0.0005 ms)
[PASS] Fallback invocations verified: Dev=0/11 (0%), HeldOut=1/10 (10% on sample 06)
[PASS] Engine routing strictly matches fallback condition for all 21 samples
```

---

## 5. Summary of Validated Metrics

| Split | Sample Count | Standalone Tesseract Latency | Strategy C Latency | Fallback Rate | Tesseract Macro CER | Strategy C Macro CER | Absolute CER Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dev Split** | 11 | 1,456.3 ms | 1,392.2 ms | **0.0% (0/11)** | 64.14% | 64.14% | **0.00%** (zero regression) |
| **Held-Out Split** | 10 | 1,350.5 ms | 1,810.9 ms | **10.0% (1/10)** | 75.50% | 71.27% | **+4.23%** (diagram rescued) |
| **Full Dataset** | 21 | 1,405.9 ms | 1,591.6 ms | **4.76% (1/21)** | 69.55% | 67.54% | **+2.01%** |

---

## 6. Documented Limitations & Project Boundary Protection

1. **Production Pipeline Protection:**
   - `TesseractOCREngine` remains the default production engine in [`backend/ai/registry.py`](file:///c:/Users/Routewise/AI_Glasses/backend/ai/registry.py).
   - The fallback policy remains an offline evaluated candidate algorithm; no changes were made to `backend/vision/` (Phase 4B frozen baseline intact).
2. **Partial Corruption Boundary:**
   - Zero-Detection Fallback triggers only on complete detection failures (`len(regions) == 0`).
   - It does not repair partially garbled text where Tesseract outputs invalid character boxes (e.g., sample `16` CER 90.20% with 12 boxes).
3. **Hardware Latency Boundary:**
   - Live camera frame OCR on raw 4K images on CPU takes $\sim 1.4\text{s}$ (Tesseract) and $\sim 5.8\text{s}$ (fallback). Resolution scaling (e.g. 2048px) should be considered if live interactive framerates are required in later phases.
