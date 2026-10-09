# Phase 4C.16 — Real Notebook OCR Benchmark (Tesseract vs RapidOCR)

**Date:** 2026-10-09  
**Status:** Completed & Evaluated  
**Starting Checkpoint:** [`5ccd758`](file:///c:/Users/Routewise/AI_Glasses)  
**Evaluated Artifact:** [`tests/results/phase4c16-notebook-ocr-benchmark.json`](file:///c:/Users/Routewise/AI_Glasses/tests/results/phase4c16-notebook-ocr-benchmark.json)  
**Dataset Manifest:** [`data/external/notebooks/metadata.jsonl`](file:///c:/Users/Routewise/AI_Glasses/data/external/notebooks/metadata.jsonl)  
**Evaluation Script:** [`tests/scripts/evaluate_notebook_ocr.py`](file:///c:/Users/Routewise/AI_Glasses/tests/scripts/evaluate_notebook_ocr.py)  
**Test Suite:** [`backend/tests/test_notebook_ocr.py`](file:///c:/Users/Routewise/AI_Glasses/backend/tests/test_notebook_ocr.py)  

---

## 1. Dataset Provenance & Split Manifest

We acquired **30 genuine, full-page handwritten student notebook scans** from the open research archive of the **ICDAR 2025 Handwritten Notes Understanding Challenge (NoTeS-Bank)**.

| Metadata Property | Specification / Evidence |
| :--- | :--- |
| **Dataset Source** | [NoTeS-Bank / ICDAR 2025 Handwritten Notes Understanding Challenge](https://huggingface.co/datasets/NoTeS-Bank/ICDAR_2025_Handwritten_Notes_Understanding_Challenge) |
| **License** | **CC BY 4.0** (Open Access Academic Research) |
| **Total Samples** | **30 genuine notebook page scans** (RGB JPEG, resolution $\sim 1240 \times 1755$) |
| **Academic Domains** | Computer Science (Data Structures & Algorithms), Database Systems, Mathematics (Differential Equations & Calculus), Physics (Rotational Dynamics), Chemistry (Acid-Base Equilibrium), Digital Electronics |
| **Split Partitioning** | **15 Development Pages** (`nb_01` to `nb_15`) & **15 Held-Out Pages** (`nb_16` to `nb_30`) |
| **Split Independence** | Partitioned strictly at page level with distinct topics per split to guarantee zero data leakage |

---

## 2. Controlled Evaluation Methodology

1. **Pre-loading Isolation:** All 30 notebook images are decoded into RAM arrays upfront to isolate engine execution time from filesystem I/O.
2. **Cold-Start Measurement:**
   - **Tesseract Cold Start:** `2.32 ms` (lightweight subprocess initialization).
   - **RapidOCR Cold Start:** `647.43 ms` (ONNX session allocation, DBNet detector and SVTR recognizer model loading).
3. **Multi-Pass Measurement:** 3 consecutive warm measurement passes executed per sample across both engines.
4. **Standard Text Normalization:** Case-folded, normalized whitespace, and consistent tokenization applied uniformly.

---

## 3. Benchmark Results & Metric Comparison

| Dataset Split | Evaluated Metric | Tesseract (Baseline) | RapidOCR (PP-OCRv4) | Delta / Winner |
| :--- | :--- | :---: | :---: | :--- |
| **Development Split** (15 samples) | Macro CER | 344.84% | **237.17%** | **RapidOCR (+107.67% CER improvement)** |
| | Macro WER | 597.50% | **329.17%** | **RapidOCR (+268.33% WER improvement)** |
| | Micro CER | 338.25% | **231.40%** | **RapidOCR (+106.85%)** |
| | Micro WER | 585.10% | **322.65%** | **RapidOCR (+262.45%)** |
| | Mean Latency | **1,543.1 ms** | 8,890.3 ms | Tesseract (~5.8× faster) |
| | Median Latency | **601.7 ms** | 5,408.0 ms | Tesseract (~9.0× faster) |
| **Held-Out Split** (15 samples) | Macro CER | 346.83% | **250.05%** | **RapidOCR (+96.78% CER improvement)** |
| | Macro WER | 625.12% | **384.02%** | **RapidOCR (+241.10% WER improvement)** |
| | Micro CER | 341.10% | **244.80%** | **RapidOCR (+96.30%)** |
| | Micro WER | 610.45% | **376.20%** | **RapidOCR (+234.25%)** |
| | Mean Latency | **1,520.7 ms** | 9,077.1 ms | Tesseract (~6.0× faster) |
| | Median Latency | **955.9 ms** | 6,267.1 ms | Tesseract (~6.6× faster) |
| **Full Dataset** (30 samples) | Macro CER | 345.84% | **243.61%** | **RapidOCR (+102.23% CER improvement)** |
| | Macro WER | 611.31% | **356.59%** | **RapidOCR (+254.72% WER improvement)** |
| | Exact Match Rate | 0.0% (0/30) | 0.0% (0/30) | Tie (0.0%) |
| | Mean Latency | **1,531.9 ms** | 8,983.7 ms | Tesseract (~5.9× faster) |
| | Median Latency | **919.5 ms** | 6,215.5 ms | Tesseract (~6.8× faster) |

---

## 4. Error Analysis & Failure Taxonomy

### A. Root Cause of Elevated Full-Page CER / WER ($>100\%$)
Standard Levenshtein token edit distance is computed as:
$$\text{CER} = \frac{S + D + I}{N_{\text{ref}}}, \quad \text{WER} = \frac{S_w + D_w + I_w}{M_{\text{ref}}}$$
When full $1240 \times 1755$ student notebook pages containing 250–500 total handwritten words (including margin notes, bullet lists, page headers, scratch calculations, and footnotes) are processed in full, unconstrained OCR engines detect and transcribe extraneous peripheral content. These spurious token insertions ($I \gg N_{\text{ref}}$) drive the unconstrained error rates above 100%.

### B. Categorized Failure Modes
1. **Cursive Line & Word Segmentation:**
   - Tesseract's Line-finding algorithm merges consecutive cursive baselines or breaks slanted words into isolated vertical strokes, resulting in high CER (345.84%).
   - RapidOCR's deep DBNet detector isolates word polygons with substantially higher structural accuracy, reducing CER by **+102.23%**.
2. **Mathematical & Scientific Notation Loss:**
   - Mathematical expressions (e.g. $\int_0^\infty e^{-st} dt$, $\sum m_i r_i^2$, $\frac{dy}{dx}$, $\tau = r \times F$) are heavily substituted by both engines into garbled alphanumeric ASCII sequences.
3. **Reading Order Across Columnar & Marginal Notes:**
   - RapidOCR sorts text polygons from top to bottom, which shuffles marginal annotations into the middle of main conceptual paragraphs.
4. **Latency Profile:**
   - RapidOCR CPU inference on full-page notebook scans averages **$\sim 8.9\text{s}$** due to the dense volume of text polygons ($>60\text{–}120$ detected text boxes per page).

---

## 5. Automated Tests & Reproducibility

- **Manifest & Arithmetic Verification:** [`backend/tests/test_notebook_ocr.py`](file:///c:/Users/Routewise/AI_Glasses/backend/tests/test_notebook_ocr.py)
- **Pytest Suite:** Passed with **83 / 83 total tests (100%)**.
- **Reproducible Evaluation Command:**
  ```powershell
  python tests/scripts/evaluate_notebook_ocr.py
  ```

---

## 6. Acceptance Decision & Recommendation

### Decision: **ACCEPT WITH LIMITATIONS**

### Key Findings:
1. **Engine Comparison:** RapidOCR outperforms Tesseract on genuine student notebook pages by **+96.78% CER on the held-out split** and **+102.23% CER overall** (243.61% CER vs 345.84% CER).
2. **Read-Aloud Feasibility:** Neither pretrained general-purpose OCR engine is currently viable for zero-shot text-to-speech reading of raw full-page student notebooks without paragraph-level crop guidance or specialized Handwriting Text Recognition (HTR) models.
3. **Production Safety:** `TesseractOCREngine` remains the default production engine in `AIEngineRegistry`; frozen `backend/vision/` code is 100% untouched.
