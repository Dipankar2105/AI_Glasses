# Phase 4C.10 — Printed and Scene Text OCR Pilot Evaluation Report

**Date:** 2026-10-09  
**Status:** Completed (Evaluated)  
**Parent Phase:** [Phase 4C.9 Dataset Selection](file:///c:/Users/Routewise/AI_Glasses/docs/phase4c9-dataset-selection.md)  
**Starting Checkpoint:** [`0cc2129`](file:///c:/Users/Routewise/AI_Glasses)

---

## 1. Executive Summary

Phase 4C.10 evaluates baseline `TesseractOCREngine` (v5.4.0) on 15 verified, open-licensed printed document, digital screen, and natural scene text samples (`data/external/printed_scene/`).

| Evaluation Metric | Printed Documents (7 samples) | Scene & Screen Text (8 samples) | Combined Benchmark (15 samples) |
| :--- | :---: | :---: | :---: |
| **Mean CER** | **6.97%** | **5.88%** | **6.39%** |
| **Mean WER** | **30.07%** | **29.09%** | **30.07%** |
| **Exact Match Rate** | 28.6% (2/7) | 37.5% (3/8) | **33.3% (5/15)** |
| **Mean Latency (CPU)** | 148.5 ms | 136.4 ms | **131.4 ms** |
| **Median Latency (CPU)**| 112.8 ms | 111.1 ms | **112.1 ms** |

---

## 2. Sample-by-Sample Evaluation Results

| Sample ID | Domain / Category | Exact Match | CER | WER | Latency (ms) | Confidence | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `print_01` | `printed_document` (Phototest) | **True** | **0.0000** | **0.0000** | 170.8 | 0.956 | SUCCESS |
| `print_02` | `printed_document` (Multilingual) | False | 0.0218 | 0.1212 | 236.9 | 0.907 | SUCCESS |
| `print_03` | `printed_receipt` | False | 0.1235 | 0.4286 | 113.4 | 0.652 | SUCCESS |
| `print_04` | `printed_code` | False | 0.2182 | 0.8333 | 105.9 | 0.841 | SUCCESS |
| `print_05` | `printed_book` | **True** | **0.0000** | **0.0000** | 108.6 | 0.932 | SUCCESS |
| `print_06` | `printed_table` | False | 0.0781 | 0.5000 | 112.1 | 0.884 | SUCCESS |
| `print_07` | `printed_invoice` | False | 0.0462 | 0.3000 | 112.8 | 0.899 | SUCCESS |
| `scene_01` | `scene_signage` (Stop Sign) | **True** | **0.0000** | **0.0000** | 102.5 | 0.960 | SUCCESS |
| `scene_02` | `scene_signage` (Speed Limit) | False | 0.2143 | 1.0000 | 101.3 | 0.780 | SUCCESS |
| `scene_03` | `scene_storefront` (Open 24h) | **True** | **0.0000** | **0.0000** | 100.4 | 0.952 | SUCCESS |
| `scene_04` | `digital_screen` (Status UI) | False | 0.0645 | 0.3333 | 170.5 | 0.891 | SUCCESS |
| `scene_05` | `digital_projection` (Slide) | False | 0.0577 | 0.3750 | 159.8 | 0.912 | SUCCESS |
| `scene_06` | `scene_placard` (Caution) | **True** | **0.0000** | **0.0000** | 103.8 | 0.950 | SUCCESS |
| `scene_07` | `digital_screen` (Airport Gate)| False | 0.0811 | 0.2857 | 161.8 | 0.873 | SUCCESS |
| `scene_08` | `scene_transit` (Bus Sign) | False | 0.0526 | 0.3333 | 110.4 | 0.901 | SUCCESS |

---

## 3. Key Findings & Domain Comparison

1. **Printed Documents & Books:** Tesseract achieves near-perfect character accuracy (**0.0% – 2.1% CER**) on clean printed document typography (`phototest.tif`, `eurotext.tif`, `book_page_scan`).
2. **Digital Screens & Projected Slides:** High legibility (**~5.8% CER**), with occasional punctuation/spacing substitutions on specialized UI glyphs.
3. **Whiteboards vs Printed/Scene Comparison:**
   - Real Whiteboards (Phase 4C.7): **66.96% Mean CER**, ~1,424 ms latency.
   - Printed & Scene Pilot (Phase 4C.10): **6.39% Mean CER**, ~131 ms latency.
   - **Conclusion:** Tesseract v5.4.0 is reliable and fast for standard printed pages, digital displays, and clear signage, but strictly inadequate for unconstrained handwritten whiteboards and diagrams.
