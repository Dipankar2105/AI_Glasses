# Phase 4C.2: Tesseract OCR Implementation

## 1. Engine Selection and Registration
The codebase now includes a genuine `TesseractOCREngine` which extracts textual regions, coordinates, and normalized confidence metrics from real pixels.
The OCR engine remains safely behind the `AIEngineRegistry` boundary.
By default, integration pipelines test against a `MockOCREngine` for determinism without requiring host-level binary dependencies.
To activate the genuine Tesseract engine in production logic, invoke:
```python
from backend.ai.registry import AIEngineRegistry
registry = AIEngineRegistry()
registry.load_production_ocr(set_active=True)
```
This binds the `pytesseract`-backed solver to the orchestrator.

## 2. Dependencies vs Host Executables
- **Python**: `pytesseract` and `Pillow` are required dependencies to run the python wrapper.
- **Host System**: The python wrapper strictly demands the **Tesseract OCR executable** (`tesseract.exe`) be locally installed and available on the system PATH.
If the executable is missing, the engine explicitly raises an `AIVisionError` noting the missing binary instead of failing silently.

## 3. Supported Image Formats
- Dtype **MUST** be strictly `uint8`. Attempting to pass `float32` will trip pipeline safety bounds and throw `AIVisionError(INVALID_INPUT)`.
- Numerical range must exactly map to `(0, 255)`.
- Dimensions supported: `(H, W)` for grayscale, and `(H, W, 3)` for RGB arrays.

## 4. Tests
Tests are located in `backend/tests/test_tesseract_ocr.py`.
- **Unit Tests**: Executed entirely isolated from the host OS utilizing `unittest.mock.patch` mocking the internal `image_to_data` boundaries.
- **Genuine OCR Integration**: If Tesseract is detected locally, `test_tesseract_ocr_genuine_integration()` executes against a locally generated synthetic PIL image. If Tesseract is unavailable, Pytest natively marks the integration block as **SKIPPED** while allowing the mock tests to PASS successfully.

## 5. Limitations
- **No Physical Hardware OCR Yet**: The OCR executes entirely on backend/host validation layers; it does not deploy C++ tesseract to the microcontrollers.
- **Windows Setup**: Windows users must download the Tesseract installer separately and add the install directory (e.g., `C:\Program Files\Tesseract-OCR`) to system environment variables.
- **Bounding Boxes**: Output boxes are globally normalized `[0.0, 1.0]`. If coordinate conversions lose sub-pixel accuracy against microscopic images, minor padding discrepancies might appear on downstream highlighting routines.

## 6. Preprocessing Experiments & Latency Observations
Since the Tesseract executable is heavily OS-dependent, the local experiments script (	ests/scripts/ocr_experiments.py) handles graceful fallback skipping. When tested with local installation, we observe deterministic processing of uppercase, lowercase, numeric, blank, grayscale and RGB synthetic Pillow fixtures. Performance latency metrics heavily depend on the local CPU cores allocating thread-counts to the tesseract C++ engine. The framework records these latencies deterministically.

## 7. Explicit TESSERACT_CMD Configuration
The engine initialization securely checks for the TESSERACT_CMD environment variable. If defined locally, it bridges the Tesseract python wrapper directly to this executable binary. This entirely prevents hardcoding native paths into the python codebase, keeping it strictly platform-agnostic.
