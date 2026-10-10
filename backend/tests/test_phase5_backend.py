import os
import sys
import base64
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.config.settings import AppSettings, get_settings, reset_settings
from backend.app import create_app
from backend.services.vision_service import VisionService, reset_vision_service
from backend.ai.registry import AIEngineRegistry
from backend.ai.mock_engines import MockObjectDetector, MockOCREngine, MockSceneAnalyzer

@pytest.fixture
def test_app():
    settings = AppSettings(
        app_name="NextSight Test Backend",
        version="0.5.0-test",
        environment="test",
        host="127.0.0.1",
        port=8000,
        log_level="DEBUG",
        request_timeout_seconds=5.0
    )
    app = create_app(settings)
    return app

@pytest.fixture
def client(test_app):
    with TestClient(test_app) as client:
        yield client


# --- 1. Configuration Tests ---

def test_settings_validation_valid():
    s = AppSettings(port=9000, log_level="warning", environment="production")
    assert s.port == 9000
    assert s.log_level == "WARNING"
    assert s.environment == "production"

def test_settings_validation_invalid_port():
    with pytest.raises(Exception):
        AppSettings(port=70000)

def test_settings_validation_invalid_log_level():
    with pytest.raises(Exception):
        AppSettings(log_level="SUPER_DEBUG")

def test_settings_validation_invalid_environment():
    with pytest.raises(Exception):
        AppSettings(environment="invalid_env")

def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("NEXTSIGHT_PORT", "8080")
    monkeypatch.setenv("NEXTSIGHT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("NEXTSIGHT_ENV", "test")
    monkeypatch.setenv("NEXTSIGHT_TIMEOUT_SEC", "10.5")

    s = AppSettings.from_env()
    assert s.port == 8080
    assert s.log_level == "DEBUG"
    assert s.environment == "test"
    assert s.request_timeout_seconds == 10.5


# --- 2. Endpoint Tests ---

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "NextSight Test Backend"
    assert data["version"] == "0.5.0-test"
    assert data["environment"] == "test"
    assert "uptime_seconds" in data
    assert data["pid"] > 0
    assert "X-Request-ID" in response.headers

def test_readiness_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "READY"
    components = data["components"]
    assert components["vision_pipeline"] == "READY"
    assert components["object_detector"] == "READY"
    assert components["scene_analyzer"] == "READY"
    assert components["ocr_engine"] == "DEFERRED_PHASE_5"
    assert components["hardware_camera"] == "UNAVAILABLE"

def test_capabilities_endpoint(client):
    response = client.get("/api/v1/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "implemented" in data
    assert "fastapi_backend_server" in data["implemented"]
    assert "vision_orchestrator" in data["implemented"]
    
    assert "unavailable" in data
    assert "physical_camera_sensor" in data["unavailable"]
    assert "physical_i2s_microphone" in data["unavailable"]
    
    assert "deferred" in data
    assert any("ocr" in s.lower() for s in data["deferred"])
    assert any("llm" in s.lower() for s in data["implemented"])
    
    assert data["hardware_status"]["camera"] == "UNAVAILABLE"
    assert "MOCKABLE" in data["hardware_status"]["hal_driver"]


# --- 3. Vision Processing Endpoint Tests ---

def test_vision_process_with_synthetic_frame(client):
    payload = {"use_mock_frame": True, "seq_num": 42}
    response = client.post("/api/v1/vision/process", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["seq_num"] == 42
    assert len(data["detections"]) > 0
    assert data["detections"][0]["label"] == "mock_object"
    assert data["scene"]["description"] == "A mock scene"
    assert "DEFERRED" in data["ocr_notice"]
    assert data["latency_ms"] > 0.0

def test_vision_process_with_valid_base64_image(client):
    # Create small valid test JPEG
    dummy = np.full((50, 50, 3), 200, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", dummy)
    b64_str = base64.b64encode(encoded).decode("utf-8")

    payload = {"image_base64": b64_str, "seq_num": 100}
    response = client.post("/api/v1/vision/process", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["seq_num"] == 100
    assert len(data["detections"]) > 0

def test_vision_process_malformed_image_payload(client):
    payload = {"image_base64": "not_a_valid_base64_image_bytes", "seq_num": 1}
    response = client.post("/api/v1/vision/process", json=payload)
    assert response.status_code == 400
    assert "Malformed image payload" in response.json()["detail"]


# --- 4. Middleware & Header Tests ---

def test_request_correlation_id_generated(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 10

def test_request_correlation_id_preserved(client):
    custom_id = "test-custom-correlation-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id


# --- 5. Dependency Injection & Service Layer Tests ---

def test_vision_service_readiness():
    registry = AIEngineRegistry()
    registry.register_detector("mock", MockObjectDetector())
    registry.register_ocr("mock", MockOCREngine())
    registry.register_scene_analyzer("mock", MockSceneAnalyzer())

    service = VisionService(registry=registry)
    readiness = service.check_readiness()
    assert readiness["vision_pipeline"] == "READY"
    assert readiness["object_detector"] == "READY"
    assert readiness["scene_analyzer"] == "READY"
    assert readiness["ocr_engine"] == "DEFERRED_PHASE_5"
