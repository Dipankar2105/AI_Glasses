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

## 4. Preprocessing Implementation (Phase 4B)
The `VisionPipeline` class safely handles pure NumPy (`np.ndarray`) image operations without reliance on heavy AI or external frameworks (e.g., `cv2` is avoided for pure determinism). 
- **Validation:** Defends against missing data, invalid dimensions, unsupported channels, and incorrect data types natively.
- **Resize:** Deterministic Nearest-Neighbor interpolation, avoiding float drift. Preserves exact identical objects when dimensions match.
- **RGB/Grayscale Conversion:** Utilizes pure ITU-R BT.601 math (`Y = 0.299R + 0.587G + 0.114B`), converting dimensions exactly (H, W, 3) ↔ (H, W) seamlessly.
- **Normalization:** Divides pixel vectors mathematically scaling `uint8` limits strictly up to deterministic `float32` [0.0 - 1.0] arrays safely.
- **Contrast Normalization:** Executes absolute Min-Max contrast stretching per-channel for RGB arrays preventing singular color washout, avoiding `NaN`/`Inf` singularities cleanly.
- **Quality Checks:** Fully implemented `ImageQualityAnalyzer` yielding Brightness, Contrast, Sharpness (via fast NumPy array Laplacian slicing), Noise Estimation, and Dynamic Range natively. **NOTE: This component strictly measures image degradation indicators; it does not restore, deblur, or fix damaged images. It is read-only and explicitly deterministic.**

## 5. Output Contract
The `VisionFrame` standardizes metadata formats allowing backend routines to asynchronously process spatial data consistently.

## 6. Physical Hardware & Toolchain Status
- **Physical OV3660 Hardware Validation:** NOT COMPLETED (Board unavailable).
- **ESP-IDF Compilation:** NOT VERIFIED (Toolchain unavailable).
- **Validation Scope:** SOFTWARE/HOST VALIDATION (PASS) using deterministic `CameraFrame` byte mocks.
