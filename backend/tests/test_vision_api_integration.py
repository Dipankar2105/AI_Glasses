import os
import sys
import time
import base64
import asyncio
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.config.settings import AppSettings
from backend.app import create_app
from backend.services.vision_service import VisionService, reset_vision_service
from backend.power.policy import PowerThermalPolicyManager, reset_power_policy_manager
from backend.power.contracts import BatteryTelemetry, ThermalTelemetry, ThermalStatus, BatteryStatus

def make_test_image_b64(width: int = 64, height: int = 64, color: tuple = (100, 150, 200), fmt: str = ".jpg") -> str:
    img = np.full((height, width, 3), color, dtype=np.uint8)
    success, encoded = cv2.imencode(fmt, img)
    assert success
    return base64.b64encode(encoded).decode("utf-8")


@pytest.fixture(autouse=True)
def cleanup_singletons():
    reset_vision_service()
    reset_power_policy_manager()
    yield
    reset_vision_service()
    reset_power_policy_manager()


def test_vision_api_valid_image_processing():
    """Valid base64 encoded image produces structured detections and scene description."""
    settings = AppSettings(
        app_name="NextSight Test",
        environment="test",
        is_strict_telemetry_required=False
    )
    app = create_app(settings)
    client = TestClient(app)

    b64_img = make_test_image_b64(128, 128)
    resp = client.post("/api/v1/vision/process", json={"image_base64": b64_img, "seq_num": 10})
    assert resp.status_code == 200
    data = resp.json()

    assert data["success"] is True
    assert data["seq_num"] == 10
    assert data["timestamp"] > 0
    assert len(data["detections"]) > 0
    assert data["scene"] is not None
    assert "DEFERRED" in data["ocr_notice"]
    assert data["latency_ms"] >= 0.0
    assert "X-Request-ID" in resp.headers


def test_vision_api_explicit_mock_frame():
    """Explicit use_mock_frame=True succeeds without providing image_base64."""
    settings = AppSettings(environment="test", is_strict_telemetry_required=False)
    app = create_app(settings)
    client = TestClient(app)

    resp = client.post("/api/v1/vision/process", json={"use_mock_frame": True, "seq_num": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["seq_num"] == 5
    assert len(data["detections"]) > 0


def test_vision_api_missing_payload_rejected():
    """Missing image_base64 when use_mock_frame=False is rejected with HTTP 400."""
    settings = AppSettings(environment="test", is_strict_telemetry_required=False)
    app = create_app(settings)
    client = TestClient(app)

    resp = client.post("/api/v1/vision/process", json={"seq_num": 1, "use_mock_frame": False})
    assert resp.status_code == 400
    assert "Missing image payload" in resp.json()["detail"]


def test_vision_api_empty_base64_string_rejected():
    """Empty or whitespace base64 string is rejected with HTTP 400."""
    settings = AppSettings(environment="test", is_strict_telemetry_required=False)
    app = create_app(settings)
    client = TestClient(app)

    resp = client.post("/api/v1/vision/process", json={"image_base64": "   ", "seq_num": 1})
    assert resp.status_code == 400
    assert "Malformed image payload" in resp.json()["detail"]


def test_vision_api_corrupt_base64_rejected():
    """Corrupted base64 bytes that cannot be decoded as an image are rejected with HTTP 400."""
    settings = AppSettings(environment="test", is_strict_telemetry_required=False)
    app = create_app(settings)
    client = TestClient(app)

    # Valid base64 encoding of random non-image bytes
    raw_garbage = base64.b64encode(b"random non image binary bytes text 12345").decode("utf-8")
    resp = client.post("/api/v1/vision/process", json={"image_base64": raw_garbage, "seq_num": 1})
    assert resp.status_code == 400
    assert "Malformed image payload" in resp.json()["detail"]


def test_vision_api_oversized_dimensions_rejected():
    """Image dimensions exceeding 4096 are rejected with HTTP 400."""
    settings = AppSettings(environment="test", is_strict_telemetry_required=False)
    app = create_app(settings)
    client = TestClient(app)

    # Create dummy 4097x1 image header/payload
    img = np.zeros((1, 4097, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", img)
    b64_str = base64.b64encode(encoded).decode("utf-8")

    resp = client.post("/api/v1/vision/process", json={"image_base64": b64_str, "seq_num": 1})
    assert resp.status_code == 400
    assert "exceed maximum permitted limit" in resp.json()["detail"]


def test_vision_api_request_correlation_id_propagation():
    """Custom X-Request-ID header is preserved and returned in response headers and body."""
    settings = AppSettings(environment="test", is_strict_telemetry_required=False)
    app = create_app(settings)
    client = TestClient(app)

    custom_id = "test-corr-id-vision-999"
    resp = client.post(
        "/api/v1/vision/process",
        json={"use_mock_frame": True},
        headers={"X-Request-ID": custom_id}
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == custom_id
    assert resp.json().get("request_id") == custom_id


def test_vision_api_strict_power_policy_fails_closed_without_telemetry():
    """In production mode with strict telemetry required, missing telemetry fails closed."""
    settings = AppSettings(environment="production", is_strict_telemetry_required=True)
    app = create_app(settings)
    client = TestClient(app)

    resp = client.post("/api/v1/vision/process", json={"use_mock_frame": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert len(data["errors"]) > 0
    assert data["errors"][0]["error"] == "BLOCKED_BY_POWER_POLICY"


def test_vision_api_strict_power_policy_authorized_with_fresh_telemetry():
    """In production mode, valid fresh telemetry allows vision execution."""
    settings = AppSettings(environment="production", is_strict_telemetry_required=True)
    app = create_app(settings)
    client = TestClient(app)

    now = time.time()
    # Inject valid telemetry into app power manager
    app.state.power_manager.update_battery_telemetry(BatteryTelemetry(
        voltage_volts=4.0,
        percentage=85.0,
        is_charging=False,
        timestamp=now
    ), current_time=now)
    app.state.power_manager.update_thermal_telemetry(ThermalTelemetry(
        temperature_celsius=38.0,
        timestamp=now
    ), current_time=now)

    resp = client.post("/api/v1/vision/process", json={"use_mock_frame": True, "seq_num": 77})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["seq_num"] == 77
    assert len(data["errors"]) == 0


def test_vision_api_strict_power_policy_rejects_critical_battery():
    """In production mode, critical battery SoC (<= 5%) blocks vision processing."""
    settings = AppSettings(environment="production", is_strict_telemetry_required=True)
    app = create_app(settings)
    client = TestClient(app)

    now = time.time()
    app.state.power_manager.update_battery_telemetry(BatteryTelemetry(
        voltage_volts=3.3,
        percentage=3.0,
        is_charging=False,
        timestamp=now
    ), current_time=now)
    app.state.power_manager.update_thermal_telemetry(ThermalTelemetry(
        temperature_celsius=35.0,
        timestamp=now
    ), current_time=now)

    resp = client.post("/api/v1/vision/process", json={"use_mock_frame": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert any(err.get("error") == "BLOCKED_BY_POWER_POLICY" for err in data["errors"])


def test_vision_api_strict_power_policy_rejects_critical_temperature():
    """In production mode, emergency high temperature (>= 70°C) blocks vision processing."""
    settings = AppSettings(environment="production", is_strict_telemetry_required=True)
    app = create_app(settings)
    client = TestClient(app)

    now = time.time()
    app.state.power_manager.update_battery_telemetry(BatteryTelemetry(
        voltage_volts=4.1,
        percentage=90.0,
        is_charging=False,
        timestamp=now
    ), current_time=now)
    app.state.power_manager.update_thermal_telemetry(ThermalTelemetry(
        temperature_celsius=75.0,
        timestamp=now
    ), current_time=now)

    resp = client.post("/api/v1/vision/process", json={"use_mock_frame": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert any(err.get("error") == "BLOCKED_BY_POWER_POLICY" for err in data["errors"])

