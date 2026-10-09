# Phase 4C.7: Whiteboard OCR Diagnostics & Alternative Engine Feasibility

## 1. Executive Summary
Phase 4C.7 conducted an in-depth diagnostic audit of Tesseract 5.4.0's performance on the 21 real whiteboard photographs (`danielrosehill/Whiteboards`). Through a controlled development/held-out split and a 6-configuration parameter matrix, we established:
1. **Configuration Tuning Yields Only Marginal Gains:** Tuning Page Segmentation Modes (PSM 3 $\rightarrow$ PSM 6) modestly reduced overall CER from **69.55%** to **66.96%** (a 2.6% absolute reduction).
2. **Fundamental Architectural Barrier:** Traditional printed-text OCR (Tesseract LSTM) cannot overcome non-linear diagram layouts, arrow connections, and unconstrained human handwriting strokes.
3. **Resolution Optimization:** Downscaling 4K images to a bounded 2048 max-dimension delivered a **2.3× speedup** (~639 ms vs ~1474 ms) with virtually identical accuracy.
4. **Alternative Engine Assessment:** Identified **RapidOCR (ONNX Runtime)** as the most realistic lightweight local engine candidate (<40 MB total footprint, Apache 2.0), and **TrOCR** as the high-accuracy handwritten line model.

---

## 2. Experimental Diagnostic Matrix & Results

### A. Deterministic Dataset Splits:
* **Development Subset (11 samples):** `01`, `02`, `03`, `04`, `05`, `07`, `08`, `09`, `10`, `11`, `17` (Prose, lists, tables, flowchart, architecture diagram).
* **Held-Out Test Subset (10 samples):** `06`, `12`, `13`, `14`, `15`, `16`, `18`, `19`, `20`, `21` (Dense diagrams, tables, progressive crops, mixed layouts).

---

### B. Development Set Matrix Evaluation:
| Configuration Name | Image Preprocessing | Tesseract Mode | Dev Mean CER | Dev Mean WER | Mean Latency | Analysis / Observation |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Baseline** | Full 4K ($4096 \times 3072$), RGB | Default PSM 3 | 0.6414 | 0.9966 | 1474.3 ms | Standard baseline layout analysis. |
| **2. Sparse Text** | Full 4K, RGB | `--psm 11` | 0.7058 | 0.9840 | 1487.6 ms | Increased false character detections on diagram shapes. |
| **3. Uniform Block** | Full 4K, RGB | `--psm 6` | **0.6324** | **1.0005** | 1448.6 ms | **Best dev CER:** Treats notes as uniform text blocks. |
| **4. Bounded Resize** | Longest edge max 2048, RGB | Default PSM 3 | 0.6694 | 1.0152 | **639.0 ms** | **2.3× speedup** with minimal CER degradation (+2.8%). |
| **5. Resize + Sparse** | Longest edge max 2048, RGB | `--psm 11` | 0.6976 | 1.0271 | 646.0 ms | Slight degradation on resized diagram strokes. |
| **6. Grayscale + Norm** | Max 2048, Gray + Min-Max | `--psm 11` | 0.6828 | 1.0745 | **340.0 ms** | **4.3× speedup**, but contrast stretch added noise artifacts. |

*Selected Best Candidate from Development Set:* **Configuration 3 (PSM 6, Uniform Block)** with CER 0.6324.

---

### C. Held-Out Set & Overall 21-Sample Evaluation:
| Evaluation Subset | Configuration | Mean CER | Mean WER | Mean Latency | Exact Match Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Held-Out (10 samples)** | Baseline (PSM 3, Full 4K) | 0.7550 | 1.0241 | 1486.8 ms | 0.0% |
| **Held-Out (10 samples)** | Candidate (PSM 6, Full 4K) | **0.7104** | **1.0222** | 1475.9 ms | 0.0% |
| **Full Dataset (21 samples)** | Phase 4C.6 Baseline (PSM 3) | 0.6955 | 1.0097 | 1511.4 ms | 0.0% |
| **Full Dataset (21 samples)** | Phase 4C.7 Candidate (PSM 6) | **0.6696** | **1.0108** | 1440.5 ms | 0.0% |

*Result Verification:* Machine-readable record preserved in `tests/results/phase4c7-whiteboard-diagnostics.json`.

---

## 3. Systematic Failure Mode Classification

Detailed inspection of per-sample recognition errors revealed 5 primary failure categories:

```
+---------------------------------------------------------------------------------------+
|                              WHITEBOARD OCR FAILURE MODES                             |
+---------------------------------------------------------------------------------------+
| 1. Diagram / Box / Arrow Interference (Samples 06, 11, 14, 16)                        |
|    - Rectangular architecture boxes and connecting arrows confuse layout analysis.     |
|    - Sample 06 produced 0 detections (CER 1.0000) due to dense diagram geometry.     |
+---------------------------------------------------------------------------------------+
| 2. Non-Linear & Multi-Column Reading Order (Samples 18, 19, 07)                       |
|    - Tabular calendar cells and 2-column sketches read left-to-right across columns.  |
|    - Words from separate columns merged into single nonsensical text lines.           |
+---------------------------------------------------------------------------------------+
| 3. Cursive Handwriting & Stroke Antialiasing (Samples 01, 04, 05, 17)                 |
|    - Marker pressure variations produce broken glyphs or joined cursive words.        |
|    - 'to AI' recognized as 'tol'; bullet dashes recognized as stray punctuation.      |
+---------------------------------------------------------------------------------------+
| 4. Markdown & Structural Formatting Ground-Truth Gap                                  |
|    - Ground-truth references contain markdown syntax (`# Header`, `- [ ]`, `->`).     |
|    - Tesseract cannot emit markdown structure, adding deletion penalties to metrics.  |
+---------------------------------------------------------------------------------------+
| 5. Glare & Illumination Gradients (Sample 12, 16)                                     |
|    - Whiteboard gloss reflections wash out dry-erase contrast on edges.               |
+---------------------------------------------------------------------------------------+
```

---

## 4. Alternative Engine Feasibility Assessment

To determine the roadmap beyond Tesseract for handwriting and whiteboard text:

### Candidate 1: RapidOCR / ONNX-Runtime (Recommended Next Evaluation)
* **Architecture:** DBNet (Text Detection) + SVTR / CRNN (Direction & Character Recognition).
* **Python Compatibility:** Python 3.10 – 3.14 via `rapidocr-onnxruntime` + `onnxruntime`.
* **Dependencies & Footprint:**
  - `rapidocr-onnxruntime` (~5 MB) + `onnxruntime` (~15 MB) + model weights (~15 MB).
  - Total download: **~35 MB** (extremely lightweight, no PyTorch needed).
* **CPU Inference Feasibility:** ~100–250 ms per image on standard CPU via optimized ONNX C++ runtime.
* **License:** Apache 2.0, 100% offline execution.
* **Pipeline Integration:** Clean drop-in implementing existing `OCREngine` contract.

### Candidate 2: Microsoft TrOCR (Handwritten Specialist)
* **Architecture:** Vision Transformer (ViT / DeiT) Encoder + RoBERTa Decoder.
* **Python Compatibility:** Python 3.10 – 3.12 (Windows PyTorch wheels on Python 3.14 currently restricted).
* **Dependencies & Footprint:** `torch` (~2.5 GB) + `transformers` + weights (~334 MB – 1.3 GB).
* **CPU Inference:** ~800–2500 ms per text line (requires separate line cropping frontend).
* **License:** MIT License.

---

## 5. Hardware Compatibility & Firmware Safety
* **Host vs. Embedded Boundary:** All diagnostic experiments, image transformations, and Tesseract configurations executed strictly on the host-side backend.
* **Firmware Safety:** Zero desktop OCR binaries, heavy libraries, or neural weights are present in ESP32 firmware requirements.
* **Camera HAL Boundary:** `CameraFrame` remains untouched and ready for hardware ingestion when the physical OV3660 camera becomes available.

---

## 6. Definitive Phase 4C.7 Recommendation

$$\mathbf{Recommendation:\ B.\ Evaluate\ RapidOCR\ in\ the\ Next\ Phase}$$

* **Justification:** Tuning Tesseract parameters on real whiteboard images has reached its diminishing-returns limit (66.96% CER). A modern scene-text detector with decoupled line recognition (such as RapidOCR/ONNX) is required to handle arbitrary rotated text, boxes, and unconstrained whiteboard layouts while maintaining a <40 MB footprint and <250 ms CPU latency.
