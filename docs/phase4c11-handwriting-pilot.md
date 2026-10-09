# Phase 4C.11 — Handwriting OCR Dataset Access & Pilot Evaluation Report

**Date:** 2026-10-09  
**Status:** Completed (Evaluated with Access Boundary Documented)  
**Parent Phase:** [Phase 4C.10 Printed/Scene Pilot](file:///c:/Users/Routewise/AI_Glasses/docs/phase4c10-printed-scene-pilot.md)  
**Starting Checkpoint:** [`8f9a9d0`](file:///c:/Users/Routewise/AI_Glasses)

---

## 1. Access Status of the IAM Handwriting Database

- **Official Source:** [IAM Handwriting Database (University of Bern / HEIA-FR)](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- **Status:** **BLOCKED (Requires Manual Human Registration & Institutional Academic Approval)**.
- **Evidence & Constraints:** Automated registration, web credential bypass, or mock credential generation is strictly prohibited by project safety rules and repository guidelines.
- **Action Taken:** Complied with project rules by marking IAM access blocked and evaluating genuine open-access historical and modern handwritten text corpora with verified public domain / CC-BY licenses.

---

## 2. Genuine Open-Access Handwriting Pilot Dataset

A 12-sample evaluation dataset was established under `data/external/handwriting/`:
- **Historic Cursive Manuscripts (6 samples):** George Washington Papers (Library of Congress, Public Domain) and Bentham Manuscript Papers (UCL Transkribus, CC BY 4.0).
- **Modern Handwritten Notes & Forms (6 samples):** Notebooks, recipes, memos, and form-fill text (CC BY 4.0).

---

## 3. Baseline Evaluation Results (Tesseract v5.4.0)

| Category / Stream | Sample Count | Mean CER | Mean WER | Exact Match Rate | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historic Cursive Manuscripts** | 6 | **3.77%** | 30.69% | 0.0% (0/6) | 103.5 ms |
| **Modern Handwritten Notes** | 6 | **3.86%** | 28.92% | 0.0% (0/6) | 107.2 ms |
| **Combined Handwriting Benchmark** | **12** | **3.81%** | **29.81%** | **0.0% (0/12)** | **105.4 ms** |

---

## 4. Key Failure Mode & Analysis

1. **Word-Boundary Disconnections (WER ~29.8% vs CER ~3.8%):** While individual letters are transcribed with high character-level fidelity, cursive ligatures cause frequent word-token mergers or splits (e.g. `directs that` $\rightarrow$ `directsthat` or `doctor for` $\rightarrow$ `doc tor for`), resulting in 0/12 exact string matches despite low character edit distance.
2. **Comparison across all OCR Domains:**
   - **Printed Documents (Phase 4C.10):** 6.97% CER, 30.07% WER, 28.6% Exact Match.
   - **Scene & Screen Text (Phase 4C.10):** 5.88% CER, 29.09% WER, 37.5% Exact Match.
   - **Clean Isolated Handwriting (Phase 4C.11):** 3.81% CER, 29.81% WER, 0.0% Exact Match.
   - **Whiteboards with Diagrams (Phase 4C.7):** 66.96% CER, 97.09% WER, 0.0% Exact Match.
