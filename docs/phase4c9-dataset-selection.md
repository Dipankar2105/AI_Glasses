# Phase 4C.9 — Printed-Text and Digital-Board Dataset Planning

**Date:** 2026-10-09  
**Status:** Completed (Selection Document)  
**Parent Phase:** [Phase 4C.8 Engine Feasibility](file:///c:/Users/Routewise/AI_Glasses/docs/phase4c8-ocr-engine-feasibility.md)  
**Starting Checkpoint:** [`95561cb`](file:///c:/Users/Routewise/AI_Glasses)

---

## 1. Domain Clarification & Scope Boundaries

In the context of the AI Glasses project, optical text capture encompasses distinct visual domains with divergent optical characteristics:

1. **Printed Documents / Books:** High contrast, black ink on white/cream paper, rectilinear font layouts, minimal perspective distortion.
2. **Digital Boards / Monitors / Projected Screens:** Emissive or reflected digital displays, pixel grid artifacts, moiré patterns, variable glare, high dynamic range.
3. **Natural Scene Text / Signage:** Real-world outdoor/indoor signs, perspective warp, lighting variation, arbitrary aspect ratios.

> [!IMPORTANT]
> A dataset of clean scanned printed pages **cannot** be used to validate digital projections or monitor captures. Each domain requires explicit evaluation.

---

## 2. Candidate Dataset Evaluation Matrix

| Criterion | Candidate 1: **TextOCR (Scene / Board Text Subset)** | Candidate 2: **DocVQA / Scanned Document Subset** | Candidate 3: **ICDAR 2015 (Incidental Scene Text)** |
| :--- | :--- | :--- | :--- |
| **Official Source** | [textvqa.org/textocr](https://textvqa.org/textocr/) / [Facebook Research](https://github.com/facebookresearch/TextOCR) | [docvqa.org](https://www.docvqa.org/) / [HuggingFace DocVQA Subsample](https://huggingface.co/datasets/nielsr/docvqa_1200_subsample_val) | [rrc.cvc.uab.es](https://rrc.cvc.uab.es/?ch=4) |
| **License** | **CC BY 4.0** (Annotations) / CC BY 2.0 (Images) | **CC BY 4.0 / Open Academic** | Research / Academic evaluation |
| **Commercial Use** | Permitted with attribution | Permitted under CC-BY / Open access | Non-commercial research terms |
| **Registration / Access** | **Public, Direct download / HuggingFace API** | **Public, Direct download / HuggingFace API** | Requires manual web registration |
| **Total Sample Count** | ~28,000 images, ~900k text instances | 1,200 evaluation sample pages | 1,500 total images (1,000 train, 500 test) |
| **Typical Dimensions** | Variable (1024×768 to 2048×1536) | Standard Document (1654×2338 / 300 DPI) | 1280×720 (Google Glass captured video frames) |
| **Ground Truth Type** | Bounding Polygons + UTF-8 string labels | Full-page transcription + Question-Answer pairs | Word-level quadrilateral bounding boxes + text |
| **Download Footprint** | ~5 MB for pilot subset of 15–20 images | ~10 MB for pilot subset of 15–20 images | ~200 MB archive |
| **Domain Relevance** | **Signs, storefronts, placards, digital boards** | **Printed books, receipts, formal documents** | Google Glass wearable incidental scene text |
| **Label Compatibility** | Directly converts to `backend/ai/ocr_benchmark.py` schema | Directly converts to reference string format | Requires line-level reconstruction |
| **Privacy / PII Risk** | Low (Public Open Images assets, anonymized) | Low (Industry legacy documents, redacted) | Low (Public urban scenes) |

---

## 3. Reasoned Selection for Phase 4C.10 Pilot

To evaluate Tesseract against both **printed documents** and **real scene/board text** without downloading unwieldy multi-gigabyte corpora, we select a balanced, multi-source pilot subset:

### Primary Pilot Dataset: **Public Printed & Scene Text Benchmark (15 samples)**
1. **Printed Document Stream (7 samples):** High-resolution open-access scanned book pages and document records from the Public Domain / CC-BY document collections (incorporating multi-column print, headers, and standard serif/sans-serif typography).
2. **Scene & Board Text Stream (8 samples):** Curated natural scene and board text samples from CC-BY open datasets (representing signage, digital screens, and indoor boards).

### Technical Integration Specification for Phase 4C.10
- **Pilot Size:** Exactly 15 verified, open-licensed samples with unambiguous ground truth.
- **Storage Location:** `data/external/printed_scene/` (configured in `.gitignore`).
- **Evaluation Engine:** Baseline `TesseractOCREngine` (v5.4.0) via `tests/scripts/evaluate_printed_scene_dataset.py`.
- **Metrics Collected:** Character Error Rate (CER), Word Error Rate (WER), Exact Match (Boolean), per-sample host latency (ms), confidence score, image dimensions.
- **Output Artifact:** `tests/results/phase4c10-printed-scene-ocr.json`.
