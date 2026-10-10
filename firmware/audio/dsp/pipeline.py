from .buffer import AudioBuffer, DSPStage
from .integrated_pipeline import (
    AudioFrame,
    AudioProcessingResult,
    AudioProcessingMetrics,
    AudioPipelineConfig,
    IntegratedAudioPipeline,
    AudioStreamAdapter,
)

class DSPPipeline:
    """A chain of DSP stages executed sequentially."""
    def __init__(self):
        self.stages = []

    def add_stage(self, stage: DSPStage):
        self.stages.append(stage)

    def process(self, buffer: AudioBuffer) -> AudioBuffer:
        current_buffer = buffer.copy()
        for stage in self.stages:
            current_buffer = stage.process(current_buffer)
        return current_buffer

__all__ = [
    "DSPPipeline",
    "AudioFrame",
    "AudioProcessingResult",
    "AudioProcessingMetrics",
    "AudioPipelineConfig",
    "IntegratedAudioPipeline",
    "AudioStreamAdapter",
]
