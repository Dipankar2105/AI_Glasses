# Phase 4C.8 — Alternative OCR Engine Feasibility Assessment

**Date:** 2026-10-09  
**Status:** Completed (Decision Record)  
**Parent Phase:** [Phase 4C.7 Diagnostics](file:///c:/Users/Routewise/AI_Glasses/docs/phase4c7-whiteboard-diagnostics.md)  
**Starting Checkpoint:** [`a419a16`](file:///c:/Users/Routewise/AI_Glasses)

---

## 1. Problem Definition & Target Workload

Phase 4C.7 diagnostics on real whiteboard images (`danielrosehill/Whiteboards`, 21 photos at 4096×3072) demonstrated that parameter tuning (PSM 3 vs 6 vs 11, grayscale normalization) on Tesseract v5.4.0 yielded only marginal CER improvements (**69.55% down to 66.96%**).

The dominant real-world failure modes identified were:
1. **Diagram / Line / Geometry Interference (40% of samples):** Flowchart arrows, table grid lines, and bounding boxes cause Tesseract's page segmenter to swallow or merge text regions entirely (e.g. 0 detections on sample `06`).
2. **Arbitrary Multi-Column & Reading Order Fragmentation (35% of samples):** Non-linear whiteboard layouts produce out-of-order text reading.
3. **Cursive & Marker Stroke Disconnections (75% of samples):** Marker antialiasing and irregular stroke widths confound Tesseract's single-line LSTM binarizer.

To address these failure modes without introducing unfeasible runtime overhead or breaking embedded constraints, we evaluate potential alternative OCR engines on paper.

---

## 2. Evaluation Matrix of OCR Engine Candidates

| Dimension | Baseline: **Tesseract v5.4.0** | Candidate 1: **RapidOCR (ONNX)** | Candidate 2: **TrOCR (HuggingFace/PyTorch)** | Candidate 3: **PaddleOCR (PaddlePaddle)** |
| :--- | :--- | :--- | :--- | :--- |
| **Architecture** | Hybrid: Otsu Binarization + Line-finding + LSTM | 2-Stage: DBNet (Det) + Direction Class + SVTR/CRNN (Recog) | End-to-End Vision Transformer (Encoder-Decoder) | 2-Stage: DBNet++ + SVTR-LCNet / PP-OCRv4 |
| **Model Footprint** | ~75 MB (Windows setup + `eng.traineddata`) | **~15 MB – 35 MB total (ONNX weights)** | **~1.5 GB – 3.0 GB (Torch + Model weights)** | ~500 MB (Paddle binaries + ~35 MB weights) |
| **Dependencies** | `pytesseract`, `tesseract.exe` | `onnxruntime`, `numpy`, `pillow` / `opencv` | `torch`, `torchvision`, `transformers`, `huggingface-hub` | `paddlepaddle`, `shapely`, `pyclipper`, `opencv-python` |
| **Host CPU Latency** | ~640 ms (2048px) – ~1,500 ms (4K) | **~150 ms – 300 ms (Optimized C++ ONNX CPU)** | ~2,500 ms – 6,000 ms (Autoregressive beam search on CPU) | ~250 ms – 450 ms (Paddle C++ / Python CPU) |
| **Memory (RAM)** | ~120 MB working set | **~150 MB – 250 MB working set** | **~2.5 GB – 4.0 GB working set** | ~600 MB – 1.2 GB working set |
| **Bounding Box Output** | Bounding box per word via `image_to_data` | **Polygon / Quadrilateral per detected text box** | None (line-only text generator; requires separate detector) | Quadrilateral per detected box |
| **License** | Apache 2.0 | **Apache 2.0** | Apache 2.0 (weights: MIT / Apache) | Apache 2.0 |
| **Offline Capability** | 100% Offline (Local binary) | **100% Offline (Local ONNX models)** | 100% Offline (after local caching of weights) | 100% Offline (after local caching) |
| **Python 3.14 Compatibility** | Full (via subprocess / `pytesseract`) | Pre-built wheels for <=3.13; requires standard build/venv for 3.14 | Pre-built wheels for <=3.13; PyTorch 3.14 wheels pending | PaddlePaddle wheels limited on newest Python versions |
| **Official Source** | [tesseract-ocr](https://github.com/tesseract-ocr/tesseract) | [RapidOCR](https://github.com/RapidAI/RapidOCR) | [microsoft/trocr](https://huggingface.co/microsoft/trocr-base-stage1) | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |

---

## 3. Detailed Technical Analysis

### Candidate 1: RapidOCR (`rapidocr-onnxruntime`) — Recommended
- **Why it fits:**
  - **Decoupled Detection & Recognition:** Uses DBNet to locate polygonal text regions regardless of surrounding diagram arrows, flowchart boxes, or calendar grids. This directly fixes the #1 failure mode observed in Phase 4C.7.
  - **Lightweight Embedded Profile:** The entire package and models (`ch_PP-OCRv4_det_infer.onnx`, `ch_PP-OCRv4_rec_infer.onnx`, `ch_ppocr_mobile_v2.0_cls_infer.onnx`) total less than **35 MB**.
  - **High CPU Throughput:** ONNX Runtime provides multi-threaded CPU SIMD/AVX2 acceleration without requiring GPU or PyTorch installations.
  - **Seamless Integration with `backend/ai/contracts.py`:**
    ```python
    class RapidOCREngine(OCREngine):
        def process(self, frame: OCRInput) -> OCRResult:
            # 1. Convert frame.image to RGB array
            # 2. Run engine(img) -> [[dt_boxes, rec_res, score], ...]
            # 3. Normalize bounding boxes to [0, 1] relative coordinates
            # 4. Construct BoundingBox and TextRegion contracts
    ```

### Candidate 2: TrOCR (`transformers` / PyTorch) — Rejected for Embedded / Fast Inference
- **Why it fails constraints:**
  - **Requires Separate Text Detector:** TrOCR is strictly a single-line text generator. Passing a full 4K whiteboard photo to TrOCR yields hallucinations or catastrophic degradation unless an auxiliary text detector crops every word line first.
  - **Excessive Resource Footprint:** Requires `torch` (~1.5 GB), `transformers`, and 350M parameter vision transformers (~1.3 GB download), consuming >3 GB RAM and 4–8 seconds per line on CPU.

### Candidate 3: PaddleOCR (`paddlepaddle`) — Rejected in Favor of RapidOCR
- RapidOCR is the direct, lightweight ONNX extraction of PaddleOCR's algorithms without the heavyweight 800 MB `paddlepaddle` framework dependencies.

---

## 4. Python Environment Strategy

- Current environment: **Python 3.14.6 (AMD64, Windows 11)**.
- If native wheels for `onnxruntime` or C++ extensions require specific compiler setups on Python 3.14, the recommended strategy is:
  1. Isolate alternative engine prototyping in a distinct Python 3.12 virtual environment or isolated microservice adapter.
  2. Keep the core vision pipeline contracts (`OCRInput`, `OCRResult`, `VisionPipeline`) strictly modular so any engine can be swapped seamlessly via the [EngineRegistry](file:///c:/Users/Routewise/AI_Glasses/backend/ai/engine_registry.py).

---

## 5. Technical Decision

1. **Retain Tesseract v5.4.0 as the active baseline** for all standard tests and Phase 4C printed/synthetic benchmarks.
2. **Select RapidOCR (ONNX)** as the primary candidate engine to evaluate in future experimental phases for scene text, whiteboards, and diagrams.
3. **Do not install heavyweight PyTorch / TrOCR dependencies.**
