"""Unit and integration tests for non-OCR Computer Vision Detectors & Scene Analyzers."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ai.detectors import (
    SalientRegionObjectDetector,
    ModelGatedObjectDetector,
    HeuristicSceneAnalyzer,
)
from backend.vision.frame import DetectionInput, SceneAnalysisInput
from backend.ai.errors import AIVisionError


def test_salient_detector_detects_high_contrast_box():
    # 200x200 black image with a white 60x60 square in the center (x: 70..130, y: 70..130)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[70:130, 70:130, :] = 255

    detector = SalientRegionObjectDetector()
    inp = DetectionInput(
        image=img,
        width=200,
        height=200,
        channels=3,
        normalization=(0, 255),
        timestamp=1000,
        seq_num=1
    )
    res = detector.process(inp)

    assert len(res.detections) >= 1
    det = res.detections[0]
    assert det.label == "salient_region"
    assert det.confidence > 0.5

    # Check bounding box bounds are around (0.35, 0.35, 0.30, 0.30)
    assert 0.30 <= det.box.x <= 0.40
    assert 0.30 <= det.box.y <= 0.40
    assert 0.25 <= det.box.width <= 0.35
    assert 0.25 <= det.box.height <= 0.35


def test_salient_detector_uniform_image_yields_zero_detections():
    # Uniform 100x100 gray image
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    detector = SalientRegionObjectDetector()
    inp = DetectionInput(
        image=img,
        width=100,
        height=100,
        channels=3,
        normalization=(0, 255),
        timestamp=1000,
        seq_num=1
    )
    res = detector.process(inp)

    assert len(res.detections) == 0


def test_salient_detector_invalid_inputs():
    detector = SalientRegionObjectDetector()
    with pytest.raises(AIVisionError):
        detector.process(DetectionInput(
            image=np.zeros((10, 10)),
            width=0,
            height=0,
            channels=1,
            normalization=(0, 255),
            timestamp=0,
            seq_num=0
        ))


def test_model_gated_detector_honest_reporting():
    # When weights file does not exist, reports unavailability or falls back
    detector = ModelGatedObjectDetector(model_path="non_existent_model.onnx", fallback_to_contours=False)
    assert detector.is_model_available is False

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    inp = DetectionInput(
        image=img,
        width=100,
        height=100,
        channels=3,
        normalization=(0, 255),
        timestamp=1000,
        seq_num=1
    )
    res = detector.process(inp)
    assert len(res.detections) == 0
    assert res.metadata.get("model_status") == "UNAVAILABLE"


def test_heuristic_scene_analyzer_dark_scene():
    # Dark image (value 10)
    dark_img = np.full((100, 100, 3), 10, dtype=np.uint8)
    analyzer = HeuristicSceneAnalyzer()
    inp = SceneAnalysisInput(
        image=dark_img,
        width=100,
        height=100,
        channels=3,
        timestamp=1000,
        seq_num=1
    )
    res = analyzer.process(inp)

    assert "low-light" in res.description
    assert "severe_darkness" in res.hazards
    assert res.confidence == 0.90


def test_heuristic_scene_analyzer_bright_glare_scene():
    # Glare image (value 250)
    bright_img = np.full((100, 100, 3), 250, dtype=np.uint8)
    analyzer = HeuristicSceneAnalyzer()
    inp = SceneAnalysisInput(
        image=bright_img,
        width=100,
        height=100,
        channels=3,
        timestamp=1000,
        seq_num=1
    )
    res = analyzer.process(inp)

    assert "high-light" in res.description
    assert "severe_glare" in res.hazards


def test_heuristic_scene_analyzer_color_tone():
    # Red-tinted warm image (R=200, G=100, B=50)
    warm_img = np.zeros((100, 100, 3), dtype=np.uint8)
    warm_img[:, :, 0] = 200
    warm_img[:, :, 1] = 100
    warm_img[:, :, 2] = 50

    analyzer = HeuristicSceneAnalyzer()
    inp = SceneAnalysisInput(
        image=warm_img,
        width=100,
        height=100,
        channels=3,
        timestamp=1000,
        seq_num=1
    )
    res = analyzer.process(inp)

    assert "warm" in res.description
    assert res.metadata["color_temp"] == "warm"
