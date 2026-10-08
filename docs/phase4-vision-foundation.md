# Phase 4: Camera & Vision Foundation

## 1. Objective
Establish a stable, standardized vision preprocessing pipeline bridging the frozen Phase 3 hardware abstraction (`CameraHAL`) with future higher-level AI backend tasks (OCR, Scene Analysis, Object Detection). 

## 2. Architecture
```
[ Frozen Phase 3 Camera HAL ]
            │
            ▼
      CameraFrame
            │
            ▼
    [ Frame Validation ]
            │
            ▼
  [ Vision Preprocessing ]
      ├── Grayscale
      ├── ROI Extraction
      └── Resize
            │
            ▼
     [ Normalization ]
            │
            ▼
     [ Quality Checks ]
            │
            ▼
       VisionFrame
            │
    ┌───────┼──────────┐
    ▼       ▼          ▼
   OCR   Detection  Scene AI
  (Future) (Future) (Future)
```

## 3. Strict Phase 3 Boundary
The Phase 4 vision layer is implemented strictly as a consumer of the `CameraFrame` and `CameraHAL` API. Absolutely no modifications were made to the frozen Phase 2 DSP or Phase 3 hardware abstraction layer.

## 4. Preprocessing Implementation
The `VisionPipeline` class safely handles incoming binary data buffers:
- **Validation:** Prevents Null/zero-dimension buffer cascades.
- **Grayscale / Resize / ROI:** Encapsulated wrappers exposing abstract dimension control safely without tightly coupling to third-party bindings right away.
- **Normalization:** Scales sensor byte thresholds mapping to floating-point gradients [0.0 - 1.0].
- **Quality Checks:** Simple deterministic metric analysis (e.g. Rejecting purely black frames).

## 5. Output Contract
The `VisionFrame` standardizes metadata formats allowing backend routines to asynchronously process spatial data consistently.

## 6. Physical Hardware & Toolchain Status
- **Physical OV3660 Hardware Validation:** NOT COMPLETED (Board unavailable).
- **ESP-IDF Compilation:** NOT VERIFIED (Toolchain unavailable).
- **Validation Scope:** SOFTWARE/HOST VALIDATION (PASS) using deterministic `CameraFrame` byte mocks.
