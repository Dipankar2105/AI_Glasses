"""NextSight Integrated Audio DSP Pipeline.

Encapsulates the complete Phase 2 / Phase 3 software audio processing chain:
1. Input Normalization and Malformed Data Validation (16 kHz, 16-bit signed PCM mono)
2. DC Removal Filter (DCBlocker)
3. High-Pass Filter (HighPassFilter @ 80 Hz)
4. Acoustic Echo Cancellation with Double-Talk Detection (NLMSAECWithDTD, if reference frame provided)
5. Spectral Noise Suppression (SpectralNoiseSuppression)
6. Voice Activity Detection (VAD)
7. Automatic Gain Control (AGC)
8. Peak Limiter (Limiter @ +/-32767.0)
9. Output Serialization and Structured Metrics Collection
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import collections
import math
import struct
import time

from .buffer import AudioBuffer, DSPStage
from .filters import DCBlocker, HighPassFilter
from .aec_dtd import NLMSAECWithDTD
from .noise_suppression import SpectralNoiseSuppression
from .vad import VAD
from .agc import AGC
from .dynamics import Limiter
from . import metrics


@dataclass
class AudioProcessingMetrics:
    """Structured metrics produced by the integrated audio pipeline."""
    sample_rate: int = 16000
    sample_count: int = 0
    frame_duration_ms: float = 0.0
    processing_time_ms: float = 0.0
    realtime_factor: float = 0.0
    input_rms: float = 0.0
    output_rms: float = 0.0
    input_peak: float = 0.0
    output_peak: float = 0.0
    input_dc_offset: float = 0.0
    output_dc_offset: float = 0.0
    clipping_detected: bool = False
    clipping_percentage: float = 0.0
    snr_estimate_db: Optional[float] = None
    vad_speech_active: bool = False
    vad_active_frames: int = 0
    vad_total_frames: int = 0
    aec_status: str = "BYPASS_NO_REFERENCE"
    dt_detected_frames: int = 0
    stages_executed: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class AudioPipelineConfig:
    """Configuration parameters for the integrated DSP pipeline."""
    sample_rate: int = 16000
    dc_blocker_enabled: bool = True
    dc_blocker_r: float = 0.995
    high_pass_enabled: bool = True
    high_pass_cutoff: float = 80.0
    aec_enabled: bool = True
    aec_filter_length: int = 256
    aec_step_size: float = 0.5
    aec_regularization: float = 1e6
    dtd_threshold: float = 0.5
    dtd_hold_samples: int = 800
    noise_suppression_enabled: bool = True
    ns_frame_size: int = 256
    ns_overlap_factor: int = 2
    ns_noise_estimation_frames: int = 10
    ns_alpha: float = 2.0
    ns_spectral_floor: float = 0.05
    vad_enabled: bool = True
    vad_frame_duration_ms: int = 10
    vad_energy_threshold: float = 500.0
    vad_attack_frames: int = 1
    vad_hangover_frames: int = 15
    agc_enabled: bool = True
    agc_target_rms: float = 4000.0
    agc_max_gain: float = 10.0
    agc_min_gain: float = 0.1
    agc_attack_time: float = 0.05
    agc_release_time: float = 0.2
    agc_noise_threshold: float = 100.0
    limiter_enabled: bool = True
    limiter_threshold: float = 32767.0
    max_queue_size: int = 50


class IntegratedAudioPipeline:
    """
    Deterministic integrated software audio pipeline.
    
    Accepts 16 kHz 16-bit signed PCM mono audio in bytes, list, or AudioBuffer formats,
    chains all validated DSP stages in frozen architecture order, and outputs processed
    PCM data alongside structured signal-quality and host performance metrics.
    """

    def __init__(self, config: Optional[AudioPipelineConfig] = None):
        self.config = config or AudioPipelineConfig()
        self._init_stages()
        self._in_queue = collections.deque(maxlen=self.config.max_queue_size)
        self._out_queue = collections.deque(maxlen=self.config.max_queue_size)

    def _init_stages(self) -> None:
        """Instantiate DSP stages with current configuration."""
        self.dc_blocker = DCBlocker(
            r=self.config.dc_blocker_r,
            enabled=self.config.dc_blocker_enabled
        )
        self.high_pass = HighPassFilter(
            cutoff_freq=self.config.high_pass_cutoff,
            sample_rate=float(self.config.sample_rate),
            enabled=self.config.high_pass_enabled
        )
        self.aec = NLMSAECWithDTD(
            filter_length=self.config.aec_filter_length,
            step_size=self.config.aec_step_size,
            regularization=self.config.aec_regularization,
            enabled=self.config.aec_enabled,
            dtd_threshold=self.config.dtd_threshold,
            dtd_hold=self.config.dtd_hold_samples
        )
        self.noise_suppressor = SpectralNoiseSuppression(
            frame_size=self.config.ns_frame_size,
            overlap_factor=self.config.ns_overlap_factor,
            noise_estimation_frames=self.config.ns_noise_estimation_frames,
            alpha=self.config.ns_alpha,
            spectral_floor=self.config.ns_spectral_floor,
            enabled=self.config.noise_suppression_enabled
        )
        self.vad = VAD(
            sample_rate=self.config.sample_rate,
            frame_duration_ms=self.config.vad_frame_duration_ms,
            energy_threshold=self.config.vad_energy_threshold,
            attack_frames=self.config.vad_attack_frames,
            hangover_frames=self.config.vad_hangover_frames,
            enabled=self.config.vad_enabled
        )
        self.agc = AGC(
            target_rms=self.config.agc_target_rms,
            max_gain=self.config.agc_max_gain,
            min_gain=self.config.agc_min_gain,
            attack_time=self.config.agc_attack_time,
            release_time=self.config.agc_release_time,
            noise_threshold=self.config.agc_noise_threshold,
            sample_rate=float(self.config.sample_rate),
            enabled=self.config.agc_enabled
        )
        self.limiter = Limiter(
            threshold=self.config.limiter_threshold,
            enabled=self.config.limiter_enabled
        )

    def reset(self) -> None:
        """Reset internal filter states, histories, adaptation weights, and queues."""
        self._init_stages()
        self._in_queue.clear()
        self._out_queue.clear()

    @staticmethod
    def bytes_to_samples(raw_bytes: bytes) -> List[float]:
        """Convert 16-bit signed little-endian PCM bytes to float samples."""
        if not isinstance(raw_bytes, (bytes, bytearray)):
            raise TypeError(f"Expected bytes or bytearray, got {type(raw_bytes).__name__}")
        if len(raw_bytes) % 2 != 0:
            raise ValueError(f"Malformed PCM byte buffer: length {len(raw_bytes)} is not a multiple of 2 (16-bit).")
        if len(raw_bytes) == 0:
            return []
        count = len(raw_bytes) // 2
        unpacked = struct.unpack(f"<{count}h", raw_bytes)
        return [float(s) for s in unpacked]

    @staticmethod
    def samples_to_bytes(samples: List[float], threshold: float = 32767.0) -> bytes:
        """Convert float samples to clamped 16-bit signed little-endian PCM bytes."""
        if not samples:
            return b""
        clamped = []
        for s in samples:
            if math.isnan(s) or math.isinf(s):
                clamped.append(0)
            elif s > threshold:
                clamped.append(int(threshold))
            elif s < -threshold - 1:
                clamped.append(int(-threshold - 1))
            else:
                clamped.append(int(round(s)))
        return struct.pack(f"<{len(clamped)}h", *clamped)

    def _validate_samples(self, samples: List[float]) -> Tuple[List[float], bool, Optional[str]]:
        """Validate sample buffer for NaN/Inf and unbounded values."""
        has_nan_inf = False
        sanitized = []
        for s in samples:
            if not isinstance(s, (int, float)):
                return [], False, f"Invalid sample type: {type(s).__name__}"
            if math.isnan(s) or math.isinf(s):
                has_nan_inf = True
                sanitized.append(0.0)
            else:
                sanitized.append(float(s))
        return sanitized, has_nan_inf, None

    def process_buffer(
        self,
        mic_buffer: AudioBuffer,
        ref_buffer: Optional[AudioBuffer] = None
    ) -> Tuple[AudioBuffer, AudioProcessingMetrics]:
        """
        Process an AudioBuffer through the full integrated DSP pipeline.
        
        Order:
        1. DCBlocker
        2. HighPassFilter
        3. AEC + DTD (if ref_buffer provided and AEC enabled)
        4. SpectralNoiseSuppression
        5. VAD
        6. AGC
        7. Limiter
        """
        t0 = time.perf_counter()
        stages_executed = []
        
        # 1. Validation
        if mic_buffer is None:
            raise ValueError("mic_buffer cannot be None")
        
        raw_samples = list(mic_buffer.data)
        sample_count = len(raw_samples)
        frame_dur_ms = (sample_count / self.config.sample_rate) * 1000.0 if self.config.sample_rate > 0 else 0.0
        
        if sample_count == 0:
            m = AudioProcessingMetrics(
                sample_rate=self.config.sample_rate,
                sample_count=0,
                frame_duration_ms=0.0,
                processing_time_ms=0.0,
                realtime_factor=0.0,
                stages_executed=[],
                aec_status="BYPASS_NO_REFERENCE"
            )
            return AudioBuffer([], self.config.sample_rate), m

        sanitized, had_nan_inf, val_err = self._validate_samples(raw_samples)
        if val_err:
            raise ValueError(val_err)

        in_buf = AudioBuffer(sanitized, self.config.sample_rate)
        in_rms = metrics.calculate_rms(in_buf)
        in_peak = metrics.calculate_peak(in_buf)
        in_dc = metrics.calculate_dc_offset(in_buf)
        
        current_buf = in_buf.copy()

        # 2. DC Blocker
        if self.dc_blocker.enabled:
            current_buf = self.dc_blocker.process(current_buf)
            stages_executed.append("DCBlocker")

        # 3. High Pass Filter (80 Hz)
        if self.high_pass.enabled:
            current_buf = self.high_pass.process(current_buf)
            stages_executed.append("HighPassFilter")

        # 4. AEC + DTD (Requires playback reference)
        aec_status = "BYPASS_NO_REFERENCE"
        dt_frames = 0
        if self.aec.enabled and ref_buffer is not None and len(ref_buffer.data) > 0:
            ref_sanitized, _, _ = self._validate_samples(list(ref_buffer.data))
            ref_buf_clean = AudioBuffer(ref_sanitized, self.config.sample_rate)
            current_buf = self.aec.process_aec(current_buf, ref_buf_clean)
            aec_status = "ACTIVE"
            dt_frames = self.aec.dt_detected_frames
            stages_executed.append("NLMSAECWithDTD")
        elif not self.aec.enabled:
            aec_status = "DISABLED"

        # 5. Noise Suppression
        if self.noise_suppressor.enabled:
            current_buf = self.noise_suppressor.process(current_buf)
            stages_executed.append("SpectralNoiseSuppression")

        # 6. VAD (Voice Activity Detection)
        vad_active = False
        vad_active_count = 0
        vad_total_count = 0
        if self.vad.enabled:
            self.vad.process(current_buf.copy())
            vad_total_count = len(self.vad.history)
            vad_active_count = sum(1 for h in self.vad.history if h.get("speech_active", False))
            vad_active = vad_active_count > 0 or self.vad.speech_active
            stages_executed.append("VAD")

        # 7. AGC (Automatic Gain Control)
        if self.agc.enabled:
            current_buf = self.agc.process(current_buf)
            stages_executed.append("AGC")

        # 8. Limiter (Ceiling protection)
        if self.limiter.enabled:
            current_buf = self.limiter.process(current_buf)
            stages_executed.append("Limiter")

        t1 = time.perf_counter()
        proc_time_ms = (t1 - t0) * 1000.0
        rt_factor = proc_time_ms / frame_dur_ms if frame_dur_ms > 0 else 0.0

        out_rms = metrics.calculate_rms(current_buf)
        out_peak = metrics.calculate_peak(current_buf)
        out_dc = metrics.calculate_dc_offset(current_buf)
        clip_pct = metrics.calculate_clipping_percentage(current_buf, self.config.limiter_threshold)
        clip_detected = clip_pct > 0.0 or out_peak >= self.config.limiter_threshold

        snr_est = None
        if in_rms > 0 and out_rms > 0:
            snr_est = round(20.0 * math.log10(max(out_rms, 1e-6) / max(in_rms, 1e-6)), 2)

        pipeline_metrics = AudioProcessingMetrics(
            sample_rate=self.config.sample_rate,
            sample_count=sample_count,
            frame_duration_ms=round(frame_dur_ms, 2),
            processing_time_ms=round(proc_time_ms, 4),
            realtime_factor=round(rt_factor, 4),
            input_rms=round(in_rms, 2),
            output_rms=round(out_rms, 2),
            input_peak=round(in_peak, 2),
            output_peak=round(out_peak, 2),
            input_dc_offset=round(in_dc, 2),
            output_dc_offset=round(out_dc, 2),
            clipping_detected=clip_detected,
            clipping_percentage=round(clip_pct, 2),
            snr_estimate_db=snr_est,
            vad_speech_active=vad_active,
            vad_active_frames=vad_active_count,
            vad_total_frames=vad_total_count,
            aec_status=aec_status,
            dt_detected_frames=dt_frames,
            stages_executed=stages_executed,
            error="NaN/Inf sanitized" if had_nan_inf else None
        )

        return current_buf, pipeline_metrics

    def process_pcm_bytes(
        self,
        mic_bytes: bytes,
        ref_bytes: Optional[bytes] = None
    ) -> Tuple[bytes, AudioProcessingMetrics]:
        """
        Convenience wrapper accepting and returning raw 16-bit signed PCM bytes.
        """
        mic_samples = self.bytes_to_samples(mic_bytes)
        ref_samples = self.bytes_to_samples(ref_bytes) if ref_bytes is not None else None

        mic_buf = AudioBuffer(mic_samples, self.config.sample_rate)
        ref_buf = AudioBuffer(ref_samples, self.config.sample_rate) if ref_samples is not None else None

        out_buf, m = self.process_buffer(mic_buf, ref_buf)
        out_bytes = self.samples_to_bytes(out_buf.data, self.config.limiter_threshold)
        return out_bytes, m

    def process_pcm_samples(
        self,
        mic_samples: List[float],
        ref_samples: Optional[List[float]] = None
    ) -> Tuple[List[float], AudioProcessingMetrics]:
        """
        Convenience wrapper accepting and returning raw float sample lists.
        """
        mic_buf = AudioBuffer(mic_samples, self.config.sample_rate)
        ref_buf = AudioBuffer(ref_samples, self.config.sample_rate) if ref_samples is not None else None
        out_buf, m = self.process_buffer(mic_buf, ref_buf)
        return out_buf.data, m

    def push_capture_chunk(self, chunk: bytes) -> bool:
        """Push a raw PCM chunk into the bounded input queue."""
        if len(self._in_queue) >= self.config.max_queue_size:
            return False  # Queue overflow prevented
        self._in_queue.append(chunk)
        return True

    def process_queued_chunks(self) -> int:
        """Process all queued chunks in FIFO order into the bounded output queue."""
        processed_count = 0
        while self._in_queue and len(self._out_queue) < self.config.max_queue_size:
            chunk = self._in_queue.popleft()
            out_bytes, _ = self.process_pcm_bytes(chunk)
            self._out_queue.append(out_bytes)
            processed_count += 1
        return processed_count

    def pop_processed_chunk(self) -> Optional[bytes]:
        """Pop a processed PCM chunk from the bounded output queue."""
        if not self._out_queue:
            return None
        return self._out_queue.popleft()
