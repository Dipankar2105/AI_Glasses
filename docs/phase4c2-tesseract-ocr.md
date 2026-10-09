# Phase 4C: Tesseract OCR Implementation & Validation

## 1. Verified Environment & Windows Configuration
* **Native Executable:** `C:\Program Files\Tesseract-OCR\tesseract.exe`
* **Tesseract Engine Version:** `5.4.0.20240606` (Leptonica 1.84.1)
* **Available Language Packs:** `eng` (English), `osd` (Orientation & Script Detection)
* **Configuration Discovery Mechanism:**
  1. Explicit configuration via `TESSERACT_CMD` environment variable:
     ```powershell
     $env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
     ```
  2. System `PATH` discovery (`shutil.which("tesseract")`)
  3. Default Windows installation paths fallback (`C:\Program Files\Tesseract-OCR\tesseract.exe`)
* **Python Runtime:** Python 3.14 + `pytesseract` + `Pillow` + `NumPy`

## 2. Engine Architecture & Registration
The codebase provides `TesseractOCREngine` which extracts textual regions, normalized bounding box coordinates `[0.0, 1.0]`, and normalized confidence metrics `[0.0, 1.0]` from real `uint8` pixel arrays.
The OCR engine is encapsulated behind the `AIEngineRegistry` boundary:
```python
from backend.ai.registry import AIEngineRegistry
registry = AIEngineRegistry()
registry.load_production_ocr(set_active=True)
```
Default tests and development fixtures continue to support `MockOCREngine` for isolated deterministic execution.

## 3. End-to-End Pipeline & Orchestrator Integration
The genuine OCR pipeline executes across the complete software path:
`CameraFrame → VisionPipeline → VisionFrame → OCRInput → TesseractOCREngine → OCRResult → VisionOrchestrator → UnifiedVisionResult`

- **Execution Command:**
  ```powershell
  pytest backend/tests/test_tesseract_ocr.py -v
  ```
- **Validation Features Tested:**
  - Synthetic high-contrast words (`HELLO`, lowercase `world`, numeric `12345`, `AI GLASSES`).
  - Blank white images return empty text `""` and 0 regions without fabricating characters.
  - Grayscale (`(H, W)`) and RGB (`(H, W, 3)`) `uint8` pixel buffers with range `(0, 255)`.
  - Bounding box coordinates clamped within normalized `[0.0, 1.0]` bounds.
  - Failure isolation and error handling: invalid input dimensions or float dtypes raise `AIVisionError(INVALID_INPUT)`.

## 4. Controlled Preprocessing Experiments & Latency
Executed via `python tests/scripts/ocr_experiments.py`:

### Preprocessing Comparison Matrix (Input: "AI GLASSES TEST"):
| Variant | Preprocessing Method | Recognized Output | Recovered | Confidence |
| :--- | :--- | :--- | :---: | :---: |
| **A. RGB** | Identity RGB frame | `AIGLASSES TEST` | Partial | 0.79 |
| **B. Grayscale** | Phase 4B `convert_format` (Luminance) | `AIGLASSES TEST` | Partial | 0.79 |
| **C. Contrast Norm** | Phase 4B `contrast_normalize` (Min-Max) | `AIGLASSES TEST` | Partial | 0.82 |
| **D. Thresholding** | OCR-local binarization (`arr < 128`) | `AIGLASSES TEST` | Partial | 0.84 |

*Observations:* Contrast normalization and thresholding improved OCR confidence scores on synthetic text. Phase 4B algorithms remained frozen and unmodified.

### Latency Measurements (Host CPU):
* **Sample Count:** 5 repeated iterations (after 1 warm-up call)
* **Image Size:** 300x50 pixels
* **Median Latency:** ~65.3 ms
* **Max Latency:** ~67.4 ms

## 5. Hardware-Readiness & ESP32 Boundary
* **Host vs. Embedded Boundary:** Tesseract OCR, Leptonica, and Pillow dependencies execute strictly on the host-side Python backend. They are **NOT** deployed into Seeed Studio XIAO ESP32-S3 Sense firmware.
* **Camera HAL Boundary:** `CameraFrame` transmits raw or encoded byte buffers from the HAL transport; decoding to `VisionFrame` and `OCRInput` happens on the host pipeline.
* **Pending Hardware Validation:** Physical board connectivity, live sensor capture from the OV3660 camera, and Wi-Fi/Bluetooth frame transport remain pending physical hardware availability.
