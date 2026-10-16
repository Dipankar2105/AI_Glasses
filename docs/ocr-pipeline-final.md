# NextSight Smart Glasses — OCR Subsystem Specification & Operations Guide

## 1. Subsystem Architecture Overview

NextSight provides an intelligent multi-engine optical character recognition (OCR) subsystem designed for wearable visual edge assistance. The pipeline balances ultra-low latency on-device triage with high-accuracy deep learning recognition and optional cloud vision multimodal models.

```mermaid
graph TD
    Frame[VisionFrame / Camera Frame / Upload] --> Ingest[Frame Ingestion & Bounds Check]
    Ingest --> StaleInv[Invalidate Cached OCR Text in Context]
    Ingest --> EngineRouter{Engine Selector}
    
    EngineRouter -->|engine=tesseract| Tess[Tesseract OCREngine (PSM 3/6)]
    EngineRouter -->|engine=rapidocr| Rapid[RapidOCR (PP-OCRv4 ONNX)]
    EngineRouter -->|engine=notebook_pipeline| Pipe[NotebookOCRPipeline]
    EngineRouter -->|engine=cloud_vision| Cloud[CloudVisionOCRAdapter (OpenAI / Gemini)]
    
    Pipe --> Prep[Safe Preprocessing: Dim Bounding + Adaptive CLAHE]
    Prep --> DLRec[RapidOCR Deep Text Detection & Recog]
    DLRec --> Layout[Column & Paragraph Layout Reconstruction]
    Layout --> Fallback[Fallback to Raw Output if Layout Empty]
    
    Tess --> Res[OCRResult Structure]
    Rapid --> Res
    Fallback --> Res
    Cloud --> Res
    
    Res --> APIOut[REST HTTP Response]
    Res --> WSOut[WebSocket ocr_result]
    Res --> Cache[Session Context Cache: last_ocr_text]
    Cache --> TTS[Voice READ_TEXT_OCR -> Audio TTS Synthesis]
```

---

## 2. Supported Engines & Configuration

| Engine Name | Implementation Class | Runtime Backend | Latency Profile | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| `tesseract` | `TesseractOCREngine` | Native C++ libtesseract via wrapper | **110–210 ms** | Fast signage, road signs, clear printed book scans, receipts |
| `rapidocr` | `RapidOCREngine` | PP-OCRv4 ONNX Runtime (CPU) | **1,100–3,200 ms** | Complex printed documents, scene text, high-accuracy multilingual lines |
| `notebook_pipeline` | `NotebookOCRPipeline` | Adaptive CLAHE + RapidOCR + Layout cluster | **1,200–3,300 ms** | Handwritten notes, multi-column layouts, mixed math/prose |
| `cloud_vision` | `CloudVisionOCRAdapter` | OpenAI (`gpt-4o`) / Google Gemini (`gemini-1.5-flash`) | Network bound (~1.5–4.0s) | Opt-in dense full-page textbook scans and complex multimodal pages |

---

## 3. Offline vs. Cloud Behavior & Privacy Implications

1. **Local-First Privacy:** By default, all OCR processing executes entirely locally on the host CPU. No camera frames, image pixels, or extracted text are transmitted over external networks.
2. **Explicit Cloud Opt-In:** `CloudVisionOCRAdapter` enforces mandatory opt-in (`opt_in=True`) and valid API credentials (`OPENAI_API_KEY` or `GEMINI_API_KEY`). If opt-in is False or credentials are missing, the adapter immediately halts with `AIVisionError(ENGINE_UNAVAILABLE)` and performs zero network requests.
3. **Zero Credential Leakage:** API keys are never written to log files, cached in session context, or returned in HTTP/WebSocket responses.

---

## 4. REST API Usage & Request Examples

### Endpoint: `POST /api/v1/vision/ocr`
Extracts text from an image payload (multipart file upload or JSON Base64).

#### Query Parameters
- `engine` *(optional, string)*: Specify engine name (`tesseract`, `rapidocr`, `notebook_pipeline`, `cloud_vision`). If omitted, uses active default (`tesseract`). Invalid engine names return **HTTP 400 Bad Request**.

#### Multipart Upload Example
```bash
curl -X POST "http://localhost:8000/api/v1/vision/ocr?engine=rapidocr" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document_scan.png"
```

#### JSON Base64 Upload Example
```bash
curl -X POST "http://localhost:8000/api/v1/vision/ocr?engine=notebook_pipeline" \
     -H "Content-Type: application/json" \
     -d '{"image_base64": "<base64_encoded_jpeg_or_png>"}'
```

#### Successful Response Format (HTTP 200)
```json
{
  "success": true,
  "engine": "rapidocr",
  "text": "SPEED LIMIT 50",
  "lines": ["SPEED LIMIT 50"],
  "paragraphs": ["SPEED LIMIT 50"],
  "confidence": 0.9654,
  "regions_count": 1,
  "latency_ms": 1120.45
}
```

#### Error Responses
- **HTTP 400 Bad Request:** Empty image buffer, corrupted image format, or invalid `?engine=unknown_xyz`.
- **HTTP 413 Payload Too Large:** Image exceeds maximum allowable upload size (10 MB).

---

## 5. Voice-to-OCR (`READ_TEXT_OCR`) Workflow

The smart glasses support hands-free reading via voice intent recognition:

1. **User Voice Query:** User utters *"Read what is in front of me"*, *"Read text"*, or *"What does the sign say"*.
2. **Intent Matching:** Intent classifier extracts `VoiceCommandIntent.READ_TEXT_OCR`.
3. **Frame Association:**
   - The command router inspects the session context metadata for `last_image`.
   - If no image has been captured yet, returns: *"No image has been captured yet. Please capture an image first."*
4. **Stale Invalidation Guarantee:** Whenever a new visual frame arrives over WebSocket (`/ws/device`), `last_ocr_text` is automatically set to `None`. This guarantees that OCR is never stale or mistakenly repeated for an old scene.
5. **TTS Dispatch:** Extracted text is spoken to the user via TTS speech synthesis and transmitted back as `BP2` binary audio packets. If no text was recognized, returns: *"No readable text was found in the current image."*

---

## 6. Reproducible Benchmark Commands & Artifacts

### Verification Commands
```powershell
# 1. Run independent OCR reproducibility gate
python tests/scripts/reproduce_ocr_benchmarks.py

# 2. Run full workspace validation and benchmarks
python run_full_validation.py
```

### Artifact Locations
- **Audit Verification Artifact:** `tests/results/phase4c21-reproducibility-audit.json`
- **Candidate Benchmark History:** `tests/results/phase4c20-candidate-notebook-ocr.json`
- **Whiteboard Pilot Baseline:** `tests/results/phase4c6-whiteboard-ocr.json`
- **Full Performance Benchmarks:** `tests/results/phase11-performance-reliability.json`

---

## 7. Measured Accuracy & Dataset Limitations

| Evaluation Dataset | Split & Target Classification | Evaluated Engine | Macro CER | Macro WER | Exact Match Rate |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Printed & Scene Text** (15 samples) | Verbatim Line/Document Scan | **RapidOCR / NotebookOCRPipeline**<br>Tesseract (PSM 3) | **1.73%**<br>6.39% | **14.10%**<br>30.07% | **53.33%** (8/15)<br>33.33% (5/15) |
| **Genuine Handwriting** (12 samples) | Verbatim Historic/Modern Note | **NotebookOCRPipeline**<br>RapidOCR (PP-OCRv4)<br>Tesseract (PSM 3) | **1.96%**<br>2.43%<br>3.82% | **18.21%**<br>22.95%<br>29.81% | **50.00%** (6/12)<br>41.67% (5/12)<br>0.00% (0/12) |
| **Dense Notebooks** (30 samples) | Conceptual Textbook Summaries | Unconstrained Local OCR | **NOT QUANTITATIVELY VALIDATED — VERBATIM DENSE-NOTEBOOK ACCURACY** | N/A | 0.00% (0/30) |

### Known Limitations
- **Dense Classroom Notebooks:** Full-page student notes containing diagrams, non-linear margin notes, and multi-step derivations cannot be evaluated for verbatim accuracy against `data/external/notebooks/metadata.jsonl` because the references represent single-sentence conceptual summaries. Full-page verbatim transcription accuracy for this category is formally classified as:
  **NOT QUANTITATIVELY VALIDATED — VERBATIM DENSE-NOTEBOOK ACCURACY**.
  While preprocessing and layout heuristics (such as aspect-ratio bounding, Hough-line deskewing, and adaptive CLAHE) have been implemented to handle heterogeneous page structures safely, no claim of increased transcription accuracy is made in the absence of verified verbatim full-page ground truth.
- **Cursive Stroke Merging:** Natural handwriting connections frequently merge prepositions and conjunctions (`Orders of` $\rightarrow$ `Ordersof`), which elevates Word Error Rate (WER) despite near-zero Character Error Rate (CER).
- **Mathematical Formulations:** Complex fraction bars, multi-line integral bounds, and custom matrices are preserved in plain text without LaTeX wrapping to maintain clean screen-reader synthesis.

---

## 8. Release Verification Status

- **Host Software Regression Tests:** **PASS** (392 passed, 3 skipped, 0 failed).
- **Firmware ESP-IDF Build:** **PASS** (Target: Seeed Studio XIAO ESP32-S3 Sense, SHA-256: `D1838C5F5619B8C336E5A26391F71A90010C1A236078E087F141DBB09488BE42`).
- **Mocked Provider Integration:** **PASS** (OpenAI and Gemini mock transport verified).
- **Live External Cloud APIs:** **NOT RUN** (Opt-in disabled and credentials unset; live calls skipped).
- **Physical Hardware In-Situ Flashing:** **HARDWARE-BLOCKED** (Physical microcontroller hardware unavailable in the current host environment).
