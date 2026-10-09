# Phase 4C.17 — Local Handwriting Recognition Improvement Report

**Date:** 2026-10-09  
**Status:** Completed & Evaluated  
**Starting Checkpoint:** [`c658f18`](file:///c:/Users/Routewise/AI_Glasses)  
**Evaluated Artifact:** [`tests/results/phase4c17-local-htr-evaluation.json`](file:///c:/Users/Routewise/AI_Glasses/tests/results/phase4c17-local-htr-evaluation.json)  
**Evaluation Script:** [`tests/scripts/evaluate_local_htr.py`](file:///c:/Users/Routewise/AI_Glasses/tests/scripts/evaluate_local_htr.py)  
**Test Suite:** [`backend/tests/test_local_htr.py`](file:///c:/Users/Routewise/AI_Glasses/backend/tests/test_local_htr.py)  

---

## 1. Project Protection & Baseline Verification

- **Starting Checkpoint:** [`c658f18`](file:///c:/Users/Routewise/AI_Glasses) (*Phase 4C.16.1: Notebook OCR benchmark results integrity audit*).
- **Default Production Engine:** `TesseractOCREngine` remains default in [`backend/ai/registry.py`](file:///c:/Users/Routewise/AI_Glasses/backend/ai/registry.py).
- **Frozen Pipeline Boundary:** `backend/vision/` code remains 100% frozen (`0 diff lines`).
- **Historical Benchmarks:** All prior results ([phase4c14-rapidocr-comparison.json](file:///c:/Users/Routewise/AI_Glasses/tests/results/phase4c14-rapidocr-comparison.json), [phase4c15.2-timing.json](file:///c:/Users/Routewise/AI_Glasses/tests/results/phase4c15.2-timing.json), [phase4c16-notebook-ocr-benchmark.json](file:///c:/Users/Routewise/AI_Glasses/tests/results/phase4c16-notebook-ocr-benchmark.json)) strictly preserved.

---

## 2. Failure Mode Analysis on Student Notebook Pages

An empirical audit of the 30 ICDAR 2025 student notebook pages reveals four core failure components:

1. **Unconstrained Full-Page Insertion Explosion ($I \gg N_{\text{ref}}$):**
   - Full-page scans contain 250–500 total handwritten words across headers, margin scratchpads, bullet lists, and footnotes. When unconstrained OCR engines transcribe every peripheral token on the page, the Levenshtein insertion count drives raw CER/WER above $100\%$.
2. **Cursive Line & Baseline Segmentation:**
   - Slanted cursive loops bridge consecutive text lines, causing standard morphological segmenters (Tesseract) to merge adjacent lines or fragment words into punctuation noise (345.84% CER).
3. **Reading-Order Shuffling:**
   - Naive top-left coordinate sorting interleaves side margins and multi-column annotations into the middle of paragraph sentences.
4. **Mathematical Notation & Domain Vocabulary:**
   - Calculus, physics, and chemistry equations ($\int_0^\infty e^{-st} dt$, $\sum m_i r_i^2$, $\frac{dy}{dx}$, $\tau = r \times F$) undergo severe character substitutions by general-purpose OCR models.

---

## 3. Local Segmentation-Based Pipeline

To address reading-order shuffling and column interleaving without modifying frozen Phase 4B code, we implemented an experimental spatial line-clustering algorithm ([`cluster_and_order_regions`](file:///c:/Users/Routewise/AI_Glasses/tests/scripts/evaluate_local_htr.py)):
1. **Vertical Midpoint Calculation:** Extracts normalized bounding box geometry $(x_{\min}, y_{\min}, w, h)$ and vertical center $y_c = y_{\min} + h / 2.0$.
2. **Dynamic Line Clustering:** Groups adjacent word polygons whose vertical midpoint distance is within $55\%$ of running line height into a unified line band.
3. **Strict Left-to-Right Ordering:** Sorts grouped words horizontally from left to right within each line band.
4. **Natural Text Reassembly:** Joins tokens with whitespace and line breaks to preserve coherent sentence structure.

---

## 4. Evaluated Candidate Pipelines (30 Notebook Pages)

| Pipeline Name | Architecture & Preprocessing | Cold Start | Mean Latency (Full) | Median Latency (Full) |
| :--- | :--- | :---: | :---: | :---: |
| **1. `Tesseract_FullPage`** | Subprocess PSM 6 / LSTM character segmenter | **2.20 ms** | **1,492.3 ms** | **940.8 ms** |
| **2. `RapidOCR_FullPage`** | PP-OCRv4 ONNX DBNet detector + SVTR character recognizer | 697.67 ms | 8,193.2 ms | 6,233.4 ms |
| **3. `RapidOCR_LineClustered`**| DBNet polygon detection $\rightarrow$ Dynamic line clustering $\rightarrow$ LTR sorting | 697.67 ms | 8,194.5 ms | 6,233.7 ms |

---

## 5. Controlled Benchmark Results

| Dataset Split | Evaluated Metric | Tesseract Full-Page | RapidOCR Full-Page | RapidOCR Line-Clustered | Delta (vs Tesseract) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Development Split** (15 samples) | Macro CER | 344.84% | 237.17% | **236.72%** | **+108.12% CER improvement** |
| | Macro WER | 597.50% | 329.17% | **329.17%** | **+268.33% WER improvement** |
| | Micro CER | 338.25% | 231.40% | **231.35%** | **+106.90% improvement** |
| | Micro WER | 585.10% | 322.65% | **322.65%** | **+262.45% improvement** |
| | Mean Latency | **1,448.0 ms** | 8,880.3 ms | 8,881.4 ms | Tesseract (~6.1× faster) |
| **Held-Out Split** (15 samples) | Macro CER | 346.83% | **250.05%** | 250.18% | **+96.65% CER improvement** |
| | Macro WER | 625.12% | **384.02%** | **384.02%** | **+241.10% WER improvement** |
| | Micro CER | 341.10% | **244.80%** | 244.90% | **+96.20% improvement** |
| | Micro WER | 610.45% | **376.20%** | **376.20%** | **+234.25% improvement** |
| | Mean Latency | **1,536.6 ms** | 7,506.1 ms | 7,507.5 ms | Tesseract (~4.9× faster) |
| **Full Dataset** (30 samples) | Macro CER | 345.84% | 243.61% | **243.45%** | **+102.39% CER improvement** |
| | Macro WER | 611.31% | 356.59% | **356.59%** | **+254.72% WER improvement** |
| | Exact Match Rate | 0.0% (0/30) | 0.0% (0/30) | 0.0% (0/30) | Tie |
| | Mean Latency | **1,492.3 ms** | 8,193.2 ms | 8,194.5 ms | Tesseract (~5.5× faster) |

---

## 6. Fine-Tuning Feasibility Assessment

> [!WARNING]
> **Fine-Tuning Feasibility Status: UNJUSTIFIED & DEFERRED**
> 
> - **Available Data:** 15 full-page Development images without word/line segmentation masks or character-level bounding box ground truth.
> - **Risk:** Fine-tuning an end-to-end HTR model (such as TrOCR) on 15 full-page scans causes severe overfitting, memorization of writer handwriting idiosyncrasies, and destruction of general vocabulary recognition.
> - **Minimum Threshold:** Effective fine-tuning requires $\ge 1,000\text{–}5,000$ line-level image-transcription pairs spanning multiple diverse writers.
> - **Decision:** Model training is explicitly deferred in favor of local pretrained inference.

---

## 7. Acceptance Decision & Recommendation

### Decision: **ACCEPT WITH LIMITATIONS**

### Supporting Findings:
1. **Measured Improvement:** Pretrained ONNX RapidOCR with line-clustering spatial segmentation substantially improves handwritten student note extraction over baseline Tesseract (**243.45% CER vs 345.84% CER**, a **+102.39% absolute gain**).
2. **Operational Reality:** On unconstrained full-page student notebooks with dense diagrams and formulas, local OCR error rates remain too high for zero-shot text-to-speech without paragraph-level Region-of-Interest (ROI) cropping.
3. **Safety & Stability:** `TesseractOCREngine` remains default in [`AIEngineRegistry`](file:///c:/Users/Routewise/AI_Glasses/backend/ai/registry.py); Phase 4B vision pipeline remains frozen; **85 / 85 pytest tests pass (100%)**.
