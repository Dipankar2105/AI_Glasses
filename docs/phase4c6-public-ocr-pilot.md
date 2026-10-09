# Phase 4C.6: Public OCR Dataset Pilot & Real-Image Whiteboard Evaluation

## 1. Executive Summary
Phase 4C.6 evaluates the host-side Tesseract 5.4.0 OCR engine on the **Whiteboards** pilot dataset (`danielrosehill/Whiteboards`), comprising 21 real-world high-resolution (4096×3072) photographs of a physical whiteboard with human-authored verbatim markdown transcriptions.

### Key Metadata:
* **Dataset Source:** [HuggingFace: danielrosehill/Whiteboards](https://huggingface.co/datasets/danielrosehill/Whiteboards)
* **Access Date:** 2026-10-09
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Sample Count:** 21 real photographic images (`01.webp` through `21.webp`)
* **Ground-Truth Availability:** 100% (all 21 samples have human-authored verbatim reference transcriptions in `metadata.jsonl`)
* **Storage Location:** `data/external/whiteboards/` (ignored by Git per `.gitignore`)

---

## 2. Dataset Properties & Provenance Limitations
* **Physical Setup:** One wall-mounted whiteboard, one black marker, single author, photographed with a mobile phone.
* **Content Domains:** 6 categories:
  - `diagram` (9 samples) — Box-and-arrow architectures, system workflows
  - `list` (6 samples) — Hierarchical notes, bulleted task lists
  - `prose` (2 samples) — Free-form sentence notes
  - `table` (2 samples) — Tabular rows/columns (calendar, structured notes)
  - `flowchart` (1 sample) — Decision tree with conditional branches
  - `mixed` (1 sample) — Multi-column layout with diagrams and text
* **Limitations:**
  - Non-linear spatial layout (arrows, branched flows, callouts).
  - High resolution (4096×3072, 12.5 Megapixels).
  - Single marker color and single author handwriting style.
  - Does not represent clean printed book text or multi-author classroom boards.

---

## 3. Real Whiteboard Baseline Performance (Tesseract 5.4.0)

Executed without parameter tuning via `tests/scripts/evaluate_public_ocr_dataset.py`. Machine-readable results saved to `tests/results/phase4c6-whiteboard-ocr.json`:

### A. Aggregate Category Breakdown:
| Category | Samples | Mean CER | Mean WER | Exact Match Rate | Mean Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Prose** | 2 | **0.5939** (59.4%) | 0.9981 | 0.0% | 1123.3 ms |
| **List** | 6 | **0.6215** (62.2%) | 1.0157 | 0.0% | 1164.5 ms |
| **Table** | 2 | **0.7050** (70.5%) | 1.0762 | 0.0% | 1065.4 ms |
| **Mixed** | 1 | **0.7321** (73.2%) | 1.0000 | 0.0% | 1026.0 ms |
| **Flowchart** | 1 | **0.7517** (75.2%) | 0.9643 | 0.0% | 1011.7 ms |
| **Diagram** | 9 | **0.7549** (75.5%) | 0.9997 | 0.0% | 1161.0 ms |
| **Overall Dataset** | **21** | **0.6955** (69.6%) | **1.0097** (~100%) | **0.0%** | **1135.8 ms** (Median: 1108.6 ms) |

---

### B. Per-Sample Detailed Recognition Breakdown:
| ID | Category | Dims | CER | WER | Latency | Detected Regions | Recognition Outcome & Failure Patterns |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `01` | prose | 4096×3072 | 0.5344 | 1.0588 | 1217.1 ms | 40 | Partially recovered isolated words; missed line wraps. |
| `02` | list | 4096×3072 | 0.6358 | 1.0690 | 1054.8 ms | 35 | Bullet markers recognized as stray commas/periods. |
| `03` | list | 4096×3072 | 0.6732 | 0.9796 | 1102.6 ms | 42 | Merged words across indentations. |
| `04` | list | 4096×3072 | 0.5827 | 0.9796 | 1174.4 ms | 45 | Partial transcription; punctuation artifacts. |
| `05` | list | 4096×3072 | 0.6000 | 1.2195 | 1108.6 ms | 52 | Hallucinated character fragments from dashes. |
| `06` | diagram | 4096×3072 | 1.0000 | 1.0000 | 789.2 ms | 0 | **Complete failure:** Dense box arrows suppressed text detection. |
| `07` | table | 4096×3072 | 0.6797 | 0.9524 | 1089.0 ms | 16 | Table grid lines fragmented column headers. |
| `08` | flowchart | 4096×3072 | 0.7517 | 0.9643 | 1011.7 ms | 17 | Decision diamonds misclassified as stray noise. |
| `09` | list | 4096×3072 | 0.6265 | 0.9535 | 1345.9 ms | 39 | Heading recognized; sub-items merged into single stream. |
| `10` | list | 4096×3072 | 0.6107 | 0.8929 | 1200.8 ms | 24 | Moderate word recovery on large uppercase headers. |
| `11` | diagram | 4096×3072 | 0.7073 | 0.9559 | 1234.6 ms | 49 | Architecture boxes fragmented into partial strings. |
| `12` | diagram | 4096×3072 | 0.7751 | 1.0000 | 1103.2 ms | 16 | Weak contrast on right edge led to omitted text. |
| `13` | diagram | 4096×3072 | 0.6919 | 1.0000 | 1409.7 ms | 58 | Agent workflow labels partially transcribed. |
| `14` | diagram | 4096×3072 | 0.7707 | 1.0000 | 1241.5 ms | 32 | Box boundaries merged into adjacent word tokens. |
| `15` | diagram | 4096×3072 | 0.6813 | 1.1053 | 1281.5 ms | 21 | Close-up crop improved character contrast slightly. |
| `16` | diagram | 4096×3072 | 0.9020 | 1.0000 | 830.3 ms | 3 | Diagram shapes caused early line segmentation abort. |
| `17` | prose | 4096×3072 | 0.6535 | 0.9375 | 1029.5 ms | 14 | Uneven marker pressure produced missing characters. |
| `18` | table | 4096×3072 | 0.7303 | 1.2000 | 1041.8 ms | 30 | Numbers in table cells misread or dropped. |
| `19` | mixed | 4096×3072 | 0.7321 | 1.0000 | 1026.0 ms | 33 | Two-column text scrambled into single linear reading order. |
| `20` | diagram | 4096×3072 | 0.7458 | 1.0000 | 1206.0 ms | 37 | Subscripted identifiers (`A_1`) dropped. |
| `21` | diagram | 4096×3072 | 0.5203 | 0.9362 | 1352.5 ms | 46 | Highest word recovery on clean uppercase labels. |

---

## 4. Key Findings: Synthetic vs. Real-World Performance Gap

1. **Synthetic vs. Real Discrepancy:**
   - On synthetic straight-line board renders (Phase 4C.4), Tesseract achieved $\text{CER} \approx 1.2\%$.
   - On real 4K phone photographs of human whiteboard handwriting, Tesseract achieved $\text{CER} \approx 69.6\%$ and $\text{WER} \approx 100\%$.
2. **Root Causes of Tesseract Failure on Real Whiteboards:**
   - **Line Segmentation Failure:** Tesseract's Page Segmentation Mode (PSM) expects horizontal text lines. Arrows, bounding boxes, and multi-directional diagram connections confuse its layout engine.
   - **Handwriting Cursive & Stroke Thinning:** Dry-erase marker strokes vary in thickness and antialiasing, causing standard Otsu binarization to break strokes or introduce false holes.
   - **Scale & Resolution:** At 4096×3072 pixels, character heights relative to image dimensions require heavy computation (~1.1s latency) while offering no improvement in layout understanding.

---

## 5. Handwritten-Paper Dataset Assessment (IAM Database)
* **Source:** [IAM Handwriting Database (FKI / University of Bern)](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)
* **Assessment Result:** **BLOCKED Pending Human Registration & Credentials**
* **Reasoning:**
  - Access requires an authorized academic user account with manual login authentication.
  - Automated or unauthenticated downloads are prohibited.
  - Per project instructions, registration was not bypassed.

---

## 6. Candidate Datasets for Next Steps

| Domain Category | Recommended Public Dataset | Size & Access | License | Suitability for AI Glasses |
| :--- | :--- | :--- | :--- | :--- |
| **Printed Documents** | **DocVQA / RVL-CDIP** | Open access via HuggingFace | Open / Non-Commercial | Standard scanned printed documents, multi-column layouts |
| **Scene & Digital Text** | **Meta TextOCR (OpenImages)** | ~28k images, open access | CC BY 4.0 | Real-world camera captures, signs, monitors, perspective tilt |
| **Mathematical Formulas** | **CROHME / IM2LATEX-100k** | ~100k formulas | Open Research | Benchmark for mathematical expression parsing |

---

## 7. Camera Hardware & Architecture Boundary
* **Host vs. Embedded Boundary:** Real dataset evaluation executes strictly on the host development machine.
* **Firmware Safety:** ESP32 firmware remains clean with zero OCR dependencies.
* **Camera Contracts:** Handled via standard `OCRInput` (NumPy `uint8` buffer) without mutating image sources.
* **Physical Hardware:** OV3660 camera streaming and on-device processing remain pending physical hardware availability.
