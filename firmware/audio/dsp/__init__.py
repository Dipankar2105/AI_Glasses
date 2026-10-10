"""Audio DSP package for NextSight AI Smart Glasses."""

from .buffer import AudioBuffer, DSPStage
from .filters import DCBlocker, HighPassFilter
from .dynamics import Gain, Limiter
from .aec import NLMSAEC
from .aec_dtd import NLMSAECWithDTD, GeigelDTD
from .noise_suppression import SpectralNoiseSuppression
from .vad import VAD
from .agc import AGC
from .integrated_pipeline import (
    AudioFrame,
    AudioProcessingResult,
    AudioProcessingMetrics,
    AudioPipelineConfig,
    IntegratedAudioPipeline,
    AudioStreamAdapter,
)
from .pipeline import DSPPipeline
from . import metrics

__all__ = [
    "AudioBuffer",
    "DSPStage",
    "DCBlocker",
    "HighPassFilter",
    "Gain",
    "Limiter",
    "NLMSAEC",
    "NLMSAECWithDTD",
    "GeigelDTD",
    "SpectralNoiseSuppression",
    "VAD",
    "AGC",
    "AudioFrame",
    "AudioProcessingResult",
    "AudioProcessingMetrics",
    "AudioPipelineConfig",
    "IntegratedAudioPipeline",
    "AudioStreamAdapter",
    "DSPPipeline",
    "metrics",
]
