import os
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

class AppSettings(BaseModel):
    """
    Centralized configuration settings for NextSight Python AI Backend.
    Reads environment variables with explicit validation and fail-fast behavior.
    """
    app_name: str = "NextSight AI Backend"
    version: str = "0.5.0"
    environment: str = Field(default="development", description="Environment: development, test, production")
    host: str = Field(default="127.0.0.1", description="Server bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server bind port")
    log_level: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")
    request_timeout_seconds: float = Field(default=15.0, gt=0.0, description="Max execution timeout for requests")
    
    # AI Subsystem Configuration & Status Flags
    enable_object_detection: bool = Field(default=True, description="Enable local object detection")
    enable_scene_analysis: bool = Field(default=True, description="Enable local scene analysis")
    
    # Honest Capability Status Declarations
    ocr_status: str = Field(default="DEFERRED", description="OCR status: ON_HOLD / DEFERRED in Phase 5")
    llm_reasoning_status: str = Field(default="DEFERRED", description="LLM Reasoning status in Phase 5")
    tts_status: str = Field(default="DEFERRED", description="TTS Speech synthesis status in Phase 5")
    hardware_camera_status: str = Field(default="UNAVAILABLE", description="Physical camera status")
    hardware_audio_status: str = Field(default="UNAVAILABLE", description="Physical microphone status")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log_level '{v}'. Must be one of {sorted(valid_levels)}")
        return v_upper

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid_envs = {"development", "test", "staging", "production"}
        v_lower = v.lower()
        if v_lower not in valid_envs:
            raise ValueError(f"Invalid environment '{v}'. Must be one of {sorted(valid_envs)}")
        return v_lower

    @classmethod
    def from_env(cls) -> "AppSettings":
        """Loads configuration from environment variables with NEXTSIGHT_ prefix."""
        env_host = os.getenv("NEXTSIGHT_HOST", "127.0.0.1")
        env_port_str = os.getenv("NEXTSIGHT_PORT", "8000")
        try:
            env_port = int(env_port_str)
        except ValueError:
            raise ValueError(f"NEXTSIGHT_PORT must be an integer, got '{env_port_str}'")

        env_log_level = os.getenv("NEXTSIGHT_LOG_LEVEL", "INFO")
        env_name = os.getenv("NEXTSIGHT_ENV", "development")
        env_timeout_str = os.getenv("NEXTSIGHT_TIMEOUT_SEC", "15.0")
        try:
            env_timeout = float(env_timeout_str)
        except ValueError:
            raise ValueError(f"NEXTSIGHT_TIMEOUT_SEC must be a float, got '{env_timeout_str}'")

        return cls(
            host=env_host,
            port=env_port,
            log_level=env_log_level,
            environment=env_name,
            request_timeout_seconds=env_timeout
        )

_settings_instance: Optional[AppSettings] = None

def get_settings() -> AppSettings:
    """Singleton getter for application settings."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AppSettings.from_env()
    return _settings_instance

def reset_settings(new_settings: Optional[AppSettings] = None) -> None:
    """Resets or overrides settings singleton (primarily for testing)."""
    global _settings_instance
    _settings_instance = new_settings
