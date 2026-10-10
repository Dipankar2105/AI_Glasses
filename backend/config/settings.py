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
    
    # Power / Thermal Telemetry Safety Gate
    require_verified_telemetry: Optional[bool] = Field(
        default=None,
        description="Enforce verified telemetry before high-power vision. Defaults to True in production/staging, False in dev/test."
    )

    # Honest Capability Status Declarations
    ocr_status: str = Field(default="DEFERRED", description="OCR status: ON_HOLD / DEFERRED in Phase 5")
    llm_reasoning_status: str = Field(default="DEFERRED", description="LLM Reasoning status in Phase 5")
    tts_status: str = Field(default="DEFERRED", description="TTS Speech synthesis status in Phase 5")
    hardware_camera_status: str = Field(default="UNAVAILABLE", description="Physical camera status")
    hardware_audio_status: str = Field(default="UNAVAILABLE", description="Physical microphone status")

    # AI Provider Configuration (STT, TTS, LLM)
    stt_provider: str = Field(default="null", description="STT Provider: null, mock, whisper")
    stt_model: str = Field(default="whisper-1", description="STT Model name")
    stt_base_url: Optional[str] = Field(default=None, description="STT API base URL")

    tts_provider: str = Field(default="null", description="TTS Provider: null, mock, api, openai, elevenlabs")
    tts_model: str = Field(default="tts-1", description="TTS Model name")
    tts_voice: str = Field(default="alloy", description="TTS Voice ID")
    tts_base_url: Optional[str] = Field(default=None, description="TTS API base URL")

    llm_provider: str = Field(default="null", description="LLM Provider: null, mock, openai, gemini, claude, api")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM Model name")
    llm_base_url: Optional[str] = Field(default=None, description="LLM API base URL")

    # Optional API Keys (Never log or leak in errors)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API Key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API Key")
    tts_api_key: Optional[str] = Field(default=None, description="Dedicated TTS API Key")

    @property
    def is_strict_telemetry_required(self) -> bool:
        """Determines whether strict telemetry gating is enforced."""
        if self.require_verified_telemetry is not None:
            return self.require_verified_telemetry
        return self.environment in ("production", "staging")


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

        env_strict_telemetry = os.getenv("NEXTSIGHT_REQUIRE_VERIFIED_TELEMETRY", None)
        strict_bool = None
        if env_strict_telemetry is not None:
            strict_bool = env_strict_telemetry.strip().lower() in ("1", "true", "yes", "on")

        # Provider configurations
        stt_prov = os.getenv("NEXTSIGHT_STT_PROVIDER", "null")
        stt_mod = os.getenv("NEXTSIGHT_STT_MODEL", "whisper-1")
        stt_url = os.getenv("NEXTSIGHT_STT_BASE_URL", None)

        tts_prov = os.getenv("NEXTSIGHT_TTS_PROVIDER", "null")
        tts_mod = os.getenv("NEXTSIGHT_TTS_MODEL", "tts-1")
        tts_vc = os.getenv("NEXTSIGHT_TTS_VOICE", "alloy")
        tts_url = os.getenv("NEXTSIGHT_TTS_BASE_URL", None)

        llm_prov = os.getenv("NEXTSIGHT_LLM_PROVIDER", "null")
        llm_mod = os.getenv("NEXTSIGHT_LLM_MODEL", "gpt-4o-mini")
        llm_url = os.getenv("NEXTSIGHT_LLM_BASE_URL", None)

        # Credentials
        oai_key = os.getenv("OPENAI_API_KEY", None)
        gem_key = os.getenv("GEMINI_API_KEY", None)
        ant_key = os.getenv("ANTHROPIC_API_KEY", None)
        tts_k = os.getenv("TTS_API_KEY", None)

        return cls(
            host=env_host,
            port=env_port,
            log_level=env_log_level,
            environment=env_name,
            request_timeout_seconds=env_timeout,
            require_verified_telemetry=strict_bool,
            stt_provider=stt_prov,
            stt_model=stt_mod,
            stt_base_url=stt_url,
            tts_provider=tts_prov,
            tts_model=tts_mod,
            tts_voice=tts_vc,
            tts_base_url=tts_url,
            llm_provider=llm_prov,
            llm_model=llm_mod,
            llm_base_url=llm_url,
            openai_api_key=oai_key,
            gemini_api_key=gem_key,
            anthropic_api_key=ant_key,
            tts_api_key=tts_k,
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
