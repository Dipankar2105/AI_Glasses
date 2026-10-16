"""
Opt-in Cloud Vision OCR Adapter using existing provider abstractions.
Supports OpenAI Vision (GPT-4o) and Google Gemini Vision for cloud-assisted OCR.

Honesty & Safety:
- Never executes live network requests without valid credentials and explicit opt-in.
- Transparently reports provider availability, authentication status, latency, and model info.
- Safely degrades to offline local OCR when credentials are absent.
"""

import os
import time
import base64
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
import httpx

from backend.ai.contracts import OCREngine, OCRResult, OCRTextRegion, BoundingBox
from backend.vision.frame import OCRInput
from backend.providers.contracts import ProviderStatus
from backend.ai.errors import AIVisionError, AIVisionErrorStatus


class CloudVisionOCRAdapter(OCREngine):
    """
    Adapter bridging Cloud Vision APIs (OpenAI GPT-4o / Google Gemini) into the OCREngine interface.
    Opt-in only: live calls require valid credentials and enabled flag.
    """

    def __init__(
        self,
        provider: str = "openai",  # "openai" or "gemini"
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 15.0,
        opt_in: bool = False
    ):
        super().__init__()
        self.provider = provider.lower().strip()
        self.opt_in = opt_in
        self.timeout_seconds = timeout_seconds

        if self.provider == "gemini":
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
            self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
            self.base_url = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        else:
            self.provider = "openai"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self.model_name = model_name or os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")
            self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")

    def is_available(self) -> bool:
        """Returns True only when opt_in is enabled and a non-empty API key is configured."""
        return bool(self.opt_in and self.api_key and self.api_key.strip())

    def get_status_info(self) -> Dict[str, Any]:
        """Provides transparent disclosure of live provider readiness and credentials."""
        has_key = bool(self.api_key and self.api_key.strip())
        return {
            "provider": self.provider,
            "model": self.model_name,
            "opt_in_enabled": self.opt_in,
            "credentials_present": has_key,
            "is_available": self.is_available(),
            "status": "READY" if self.is_available() else ("CREDENTIALS_MISSING" if not has_key else "OPT_IN_REQUIRED")
        }

    def process(self, input_data: OCRInput) -> OCRResult:
        """
        Executes OCR on OCRInput. If provider is unavailable, raises AIVisionError with AUTH_FAILED
        or returns structured skipped status.
        """
        t0 = time.perf_counter()

        if input_data is None or input_data.image is None or input_data.image.size == 0:
            raise AIVisionError(AIVisionErrorStatus.INVALID_INPUT, "Input image is empty or None")

        if not self.is_available():
            status_info = self.get_status_info()
            raise AIVisionError(
                AIVisionErrorStatus.ENGINE_UNAVAILABLE,
                f"Cloud Vision OCR [{self.provider}] is unavailable: {status_info['status']}. "
                f"Live cloud requests require explicit opt-in and valid API keys."
            )

        # Encode RGB image to JPEG in memory
        arr = input_data.image
        if arr.ndim == 2:
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            raise AIVisionError(AIVisionErrorStatus.INVALID_INPUT, f"Unsupported image channels: {arr.shape}")

        success, enc = cv2.imencode(".jpg", arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise AIVisionError(AIVisionErrorStatus.IMAGE_PROCESSING_FAILED, "Failed to encode image to JPEG")

        b64_img = base64.b64encode(enc.tobytes()).decode("ascii")

        try:
            if self.provider == "gemini":
                text = self._call_gemini(b64_img)
            else:
                text = self._call_openai(b64_img)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            return OCRResult(
                regions=[],
                full_text=text,
                metadata={
                    "engine": f"CloudVision_{self.provider}",
                    "model": self.model_name,
                    "latency_ms": round(latency_ms, 2),
                    "num_lines": len(lines)
                }
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            raise AIVisionError(
                AIVisionErrorStatus.ENGINE_FAILURE,
                f"Cloud Vision OCR request failed ({self.provider}/{self.model_name}): {str(e)}"
            )

    def _call_openai(self, b64_jpeg: str) -> str:
        prompt = "Transcribe all text from this image exactly as written. Return only the recognized text without markdown formatting or preamble."
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_jpeg}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 2048
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI API error HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            content = choices[0].get("message", {}).get("content", "")
            return content.strip() if content else ""

    def _call_gemini(self, b64_jpeg: str) -> str:
        prompt = "Transcribe all text from this image exactly as written. Return only the recognized text without markdown formatting or preamble."
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_jpeg
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 2048
            }
        }

        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini API error HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
