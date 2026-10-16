import os
import sys
import pytest
import numpy as np
import httpx
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.providers.cloud_ocr import CloudVisionOCRAdapter
from backend.vision.frame import OCRInput
from backend.ai.errors import AIVisionError, AIVisionErrorStatus


def test_cloud_ocr_opt_in_safety():
    # Case 1: No key, opt_in=False
    adapter1 = CloudVisionOCRAdapter(provider="openai", opt_in=False)
    assert adapter1.is_available() is False
    status1 = adapter1.get_status_info()
    assert status1["opt_in_enabled"] is False
    assert status1["credentials_present"] is False
    assert status1["status"] == "CREDENTIALS_MISSING"

    # Case 2: Key present, but opt_in=False -> Still unavailable
    adapter2 = CloudVisionOCRAdapter(provider="openai", api_key="sk-test-key-12345", opt_in=False)
    assert adapter2.is_available() is False
    status2 = adapter2.get_status_info()
    assert status2["opt_in_enabled"] is False
    assert status2["credentials_present"] is True
    assert status2["status"] == "OPT_IN_REQUIRED"

    # Case 3: Both key and opt_in=True -> Ready
    adapter3 = CloudVisionOCRAdapter(provider="openai", api_key="sk-test-key-12345", opt_in=True)
    assert adapter3.is_available() is True
    assert adapter3.get_status_info()["status"] == "READY"


def test_cloud_ocr_process_rejects_unconfigured():
    adapter = CloudVisionOCRAdapter(provider="openai", opt_in=False)
    dummy_img = np.full((100, 100, 3), 200, dtype=np.uint8)
    inp = OCRInput(image=dummy_img, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)

    with pytest.raises(AIVisionError) as exc_info:
        adapter.process(inp)
    assert exc_info.value.status == AIVisionErrorStatus.ENGINE_UNAVAILABLE
    assert "unavailable" in str(exc_info.value).lower()


def test_cloud_ocr_openai_mock_response():
    adapter = CloudVisionOCRAdapter(
        provider="openai",
        api_key="sk-test-mock-key",
        model_name="gpt-4o",
        opt_in=True
    )
    dummy_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    inp = OCRInput(image=dummy_img, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)

    mock_resp_data = {
        "choices": [
            {
                "message": {
                    "content": "Line 1: Transcribed Text\nLine 2: Mathematics x + y = z"
                }
            }
        ]
    }

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_resp_data
        mock_post.return_value = mock_resp

        res = adapter.process(inp)
        assert "Line 1: Transcribed Text" in res.full_text
        assert res.metadata["engine"] == "CloudVision_openai"
        assert res.metadata["model"] == "gpt-4o"
        assert res.metadata["num_lines"] == 2


def test_cloud_ocr_gemini_mock_response():
    adapter = CloudVisionOCRAdapter(
        provider="gemini",
        api_key="ai-za-test-gemini-key",
        model_name="gemini-1.5-flash",
        opt_in=True
    )
    dummy_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    inp = OCRInput(image=dummy_img, width=100, height=100, channels=3, numerical_range=(0, 255), timestamp=1, seq_num=1)

    mock_resp_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Extracted formula: E = mc^2"}
                    ]
                }
            }
        ]
    }

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_resp_data
        mock_post.return_value = mock_resp

        res = adapter.process(inp)
        assert "Extracted formula: E = mc^2" in res.full_text
        assert res.metadata["engine"] == "CloudVision_gemini"
        assert res.metadata["model"] == "gemini-1.5-flash"
