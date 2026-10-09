# Phase 4C.4: Multi-Type OCR Robustness & Handwriting Benchmark

## 1. Executive Summary & Objective
Phase 4C.4 establishes a standardized, reproducible multi-type OCR evaluation framework for the AI Glasses host-side vision system. The benchmark evaluates genuine OCR performance across 5 distinct real-world reading categories (15 deterministic fixtures):
1. **Printed text** in books, notebooks, and digital documents.
2. **Handwritten text** on paper notes (synthetic handwriting-style stress fixtures).
3. **Digital displays and projected slides** (dark themes, projector glare).
4. **Classroom boards** (dry-erase whiteboards, dark chalkboards, perspective-skewed views).
5. **Mathematical expressions and formulas** (arithmetic, elementary physics, calculus).

---

## 2. Benchmark Dataset & Provenance

Every sample is strictly categorized by domain and provenance:

| Sample ID | Category | Provenance | Description | Reference / Target String |
| :--- | :--- | :--- | :--- | :--- |
| `A1_printed_simple` | A. Printed Text | Synthetic Printed | High-contrast clean uppercase | `HELLO WORLD` |
| `A2_printed_mixed_case` | A. Printed Text | Synthetic Printed | Clean mixed-case sentence & year | `Optical Character Recognition 2026` |
| `A3_printed_multiline` | A. Printed Text | Synthetic Printed | Multi-line document paragraph | `AI Glasses Vision Pipeline Real Time Host Processing Frame Sequence 101` |
| `A4_printed_low_contrast` | A. Printed Text | Synthetic Printed | Low contrast gray-on-gray document | `Low Contrast Document` |
| `B1_handwritten_note` | B. Handwritten Paper | Synthetic Handwriting Style | Casual note with paper background | `Welcome to AI Glasses class` |
| `B2_handwritten_numbers` | B. Handwritten Paper | Synthetic Handwriting Style | Casual room number & time note | `Room 302 at 4:30 PM` |
| `B3_handwritten_blurred` | B. Handwritten Paper | Synthetic Handwriting Style | Hand note with mild optical blur | `Meeting on Friday morning` |
| `C1_digital_slide_dark` | C. Digital Boards | Synthetic Board Render | Inverted dark-theme slide presentation | `EDGE AI ARCHITECTURE Low Latency Processing Host Side OCR Engine` |
| `C2_digital_slide_glare` | C. Digital Boards | Synthetic Board Render | Slide with simulated projector glare | `PROJECTOR DISPLAY TEST` |
| `D1_whiteboard_marker` | D. Classroom Boards | Synthetic Board Render | Blue dry-erase marker on whiteboard | `Assignment Due Next Tuesday` |
| `D2_blackboard_chalk` | D. Classroom Boards | Synthetic Board Render | White chalk on dark slate blackboard | `Physics 101: Newton Laws` |
| `D3_board_perspective` | D. Classroom Boards | Synthetic Board Render | Angled whiteboard perspective skew | `CLASSROOM LECTURE NOTES` |
| `E1_math_arithmetic` | E. Mathematics | Synthetic Printed | Elementary arithmetic equation | `12 + 34 = 46` |
| `E2_math_formula` | E. Mathematics | Synthetic Printed | Physics equation with exponents | `E = mc^2 and F = ma` |
| `E3_math_calculus` | E. Mathematics | Synthetic Printed | Derivative calculus equation | `dy/dx = 2x + 5` |

---

## 3. Evaluation Metrics & Normalization Policy
* **Text Normalization:**
  - Standardizes multiple spaces, tabs, and newline sequences (`\r\n`, `\n`) into a single space.
  - Case normalization (case-insensitive evaluation by default, case-sensitive mode configurable).
* **Character Error Rate (CER):**
  $$\text{CER} = \frac{\text{LevenshteinDistance}(\text{ref\_chars}, \text{hyp\_chars})}{\max(1, \text{len}(\text{ref\_chars}))}$$
* **Word Error Rate (WER):**
  $$\text{WER} = \frac{\text{LevenshteinDistance}(\text{ref\_words}, \text{hyp\_words})}{\max(1, \text{len}(\text{ref\_words}))}$$
* **Exact Match (EM):** Boolean match between normalized reference and hypothesis.

---

## 4. Tesseract 5.4.0 Measured Benchmark Results

Executed on Host CPU (Windows 64-bit, Tesseract v5.4.0.20240606, Python 3.14):

### A. Per-Category Performance Breakdown:
| Category | Sample Count | Mean CER | Mean WER | Exact Match Rate | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A. Printed Text** | 4 | **0.04** (4%) | 0.40 | 25.0% | ~71.2 ms |
| **B. Handwritten Text (Paper)** | 3 | **0.09** (9%) | 0.40 | 33.3% | ~74.5 ms |
| **C. Digital Boards & Slides** | 2 | **0.03** (3%) | 0.20 | 50.0% | ~73.8 ms |
| **D. Classroom Boards** | 3 | **0.01** (1%) | 0.08 | **66.7%** | ~72.1 ms |
| **E. Mathematics & Formulas** | 3 | **1.00** (100%) | 1.00 | 0.0% | ~76.0 ms |
| **Overall Dataset** | **15** | **0.24** (24%) | **0.43** | **33.3%** | **73.45 ms** |

---

## 5. Key Findings & Detailed Analysis

1. **Printed Text & Digital Slides (Strong Baseline):**
   - High character-level recovery (CER $\le 4\%$).
   - Word boundaries are occasionally merged when inter-character spacing is tight (`HELLO WORLD` $\rightarrow$ `HELLOWORLD`).
2. **Chalkboards & Whiteboards (Excellent Contrast Adaptation):**
   - Tesseract effectively recognized both dark-on-light (whiteboard marker) and light-on-dark (chalkboard) renderings (CER $1\%$).
3. **Mathematics Failure Mode (Zero Recognition):**
   - Default Tesseract LSTM page segmentation models (PSM) rejected isolated single-line mathematical equations (`12 + 34 = 46`, `E = mc^2`, `dy/dx = 2x + 5`), returning empty strings (`""`).
   - *Conclusion:* Standard general-purpose OCR cannot recognize mathematical structures or formulas without specialized segmentation, math-symbol trained models, or LaTeX parsing models.
4. **Handwriting Limitations:**
   - Synthetic script fonts exhibited 9% CER on clean rendered text, but errors quickly appeared with mixed spacing and numbers (`Room 302 at 4:30 PM` $\rightarrow$ `oom 302at 430 PM`).
   - Real human handwriting exhibits high variability, non-linear baselines, and connected cursive strokes which Tesseract cannot reliably transcribe.

---

## 6. Controlled Board Preprocessing & Perspective Experiments

Evaluated on angled classroom board fixture (`D3_board_perspective`):

| Preprocessing Technique | Recognized Text Output | CER | WER | Evaluation |
| :--- | :--- | :---: | :---: | :--- |
| **1. Raw Perspective Skew** | `CLASSROOM LECTURE NOTES` | **0.00** | **0.00** | **Exact Recovery** |
| **2. Local Binarization (Threshold < 128)** | `GLASSAGOM LECTURE NOTES` | 0.13 | 0.33 | Severe artifacts introduced |
| **3. Perspective Rectification (Inverse Affine)** | `CLASSROOM LECTURE NOTES` | **0.00** | **0.00** | **Exact Recovery** |
| **4. Rectification + Thresholding** | `GLASSACOM LEGTURENOTES` | 0.22 | 1.00 | Degraded recognition |

*Key Insight:* Hard global/local binarization frequently fragments antialiased glyphs. Contrast normalization and affine rectification preserve smooth character strokes for the OCR engine.

---

## 7. Handwriting-Capable Alternative Evaluation (TrOCR / EasyOCR / Nougat)

An architectural audit of alternative handwriting and mathematical models was conducted:

| Recognizer Model | Target Domain | Model Size | Runtime Dependency | Python 3.14 / Host Compatibility | Host CPU Feasibility | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tesseract 5.4.0** | Printed / Boards | ~30 MB | Native C++ binary | Fully Verified | **~65–75 ms** | **Active Production Default** |
| **TrOCR (Handwritten)** | Cursive / Human Hand | ~334 MB – 1.3 GB | PyTorch + Transformers | Blocked by Windows PyTorch wheel availability in active env | ~250–600 ms (CPU) | Documented Integration Candidate |
| **Nougat / Pix2Text** | Math Formulas & Tables | ~500 MB – 1.8 GB | PyTorch + Torchvision | Requires PyTorch C++ extensions | ~800–1500 ms (CPU) | Future Math OCR Phase Candidate |

---

## 8. Hardware Compatibility & Architecture Boundary
* **Host vs. Embedded Boundary:** Tesseract, image transformations, and benchmark suites remain 100% host-side Python backend components.
* **Firmware Safety:** Zero desktop OCR or neural network dependencies are added to ESP32 firmware.
* **Camera Contracts:** Tested against existing `CameraFrame` and `VisionFrame` contracts.
* **Pending Hardware Tests:** Live camera streaming, physical board illumination, and autofocus from the OV3660 camera remain pending physical board connection.
