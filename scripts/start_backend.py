#!/usr/bin/env python3
"""NextSight Smart Glasses — Developer Backend Launcher.

Usage:
  python scripts/start_backend.py
  python scripts/start_backend.py --host 0.0.0.0 --port 8000 --reload
  python scripts/start_backend.py --offline  # Starts with mock AI providers
"""

import sys
import os
import argparse
import uvicorn

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config.settings import get_settings


def main():
    parser = argparse.ArgumentParser(description="NextSight Smart Glasses Backend Server Launcher")
    parser.add_argument("--host", default=None, help="Server bind host (default from settings or 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Server port (default from settings or 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    parser.add_argument("--offline", action="store_true", help="Force mock/offline providers (no API keys needed)")
    parser.add_argument("--log-level", default=None, help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    args = parser.parse_args()

    if args.offline:
        os.environ["NEXTSIGHT_STT_PROVIDER"] = "mock"
        os.environ["NEXTSIGHT_LLM_PROVIDER"] = "mock"
        os.environ["NEXTSIGHT_TTS_PROVIDER"] = "mock"

    cfg = get_settings()

    bind_host = args.host or cfg.host
    bind_port = args.port or cfg.port
    log_lvl = (args.log_level or cfg.log_level).lower()

    print("=" * 70)
    print(" NEXTSIGHT SMART GLASSES — PYTHON AI BACKEND")
    print("=" * 70)
    print(f" App Name:     {cfg.app_name} v{cfg.version}")
    print(f" Environment:  {cfg.environment}")
    print(f" Host/Port:    http://{bind_host}:{bind_port}")
    print(f" Mode:         {'OFFLINE / MOCK' if args.offline or cfg.stt_provider == 'mock' else 'CONFIGURED'}")
    print(f" Providers:    STT={cfg.stt_provider} | LLM={cfg.llm_provider} | TTS={cfg.tts_provider}")
    print(f" Docs UI:      http://{bind_host}:{bind_port}/docs")
    print(f" Health Check: http://{bind_host}:{bind_port}/health")
    print("=" * 70 + "\n")

    uvicorn.run(
        "backend.app:app",
        host=bind_host,
        port=bind_port,
        reload=args.reload,
        log_level=log_lvl
    )


if __name__ == "__main__":
    main()
