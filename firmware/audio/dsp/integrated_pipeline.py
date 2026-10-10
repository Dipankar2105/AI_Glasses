"""NextSight Integrated Audio DSP Pipeline.

Encapsulates the complete Phase 2 / Phase 3 software audio processing chain:
1. Frame Ingestion and Malformed Data Validation (16 kHz, 16-bit signed PCM mono)
2. DC Removal Filter (DCBlocker)
3. High-Pass Filter (HighPassFilter @ 80 Hz)
4. Acoustic Echo Cancellation with Double-Talk Detection (NLMSAECWithDTD, if reference frame provided)
5. Spectral Noise Suppression (SpectralNoiseSuppression)
6. Voice Activity Detection (VAD)
7. Automatic Gain Control (AGC)
8. Peak Limiter (Limiter @ +/-32767.0)
9. Output Serialization, Per-Stage Performance Profiling, and Structured Metrics Collection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
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
class AudioFrame:
    """Lightweight, strongly-typed representation of an audio frame."""
    pcm_data: List[float]
    sample_rate: int = 16000
    channels: int = 1
    sample_format: str = "int16"
    timestamp_ms: Optional[float] = None
    seq_num: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.pcm_data)

    @property
    def duration_ms(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return (len(self.pcm_data) / self.sample_rate) * 1000.0

    def to_bytes(self, threshold: float = 32767.0) -> bytes:
        """Convert float samples to clamped 16-bit signed little-endian PCM bytes."""
        if not self.pcm_data:
            return b""
        clamped = []
        for s in self.pcm_data:
            if math.isnan(s) or math.isinf(s):
                clamped.append(0)
            elif s > threshold:
                clamped.append(int(threshold))
            elif s < -threshold - 1:
                clamped.append(int(-threshold - 1))
            else:
                clamped.append(int(round(s)))
        return struct.pack(f"<{len(clamped)}h", *clamped)

    @classmethod
    def from_bytes(
        cls,
        raw_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        timestamp_ms: Optional[float] = None,
        seq_num: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AudioFrame":
        """Construct an AudioFrame from raw 16-bit signed little-endian PCM bytes."""
        if not isinstance(raw_bytes, (bytes, bytearray)):
            raise TypeError(f"Expected bytes or bytearray, got {type(raw_bytes).__name__}")
        if len(raw_bytes) % 2 != 0:
            raise ValueError(f"Malformed PCM byte buffer: length {len(raw_bytes)} is not a multiple of 2 (16-bit).")
        count = len(raw_bytes) // 2
        unpacked = struct.unpack(f"<{count}h", raw_bytes) if count > 0 else ()
        return cls(
            pcm_data=[float(s) for s in unpacked],
            sample_rate=sample_rate,
            channels=channels,
            sample_format="int16",
            timestamp_ms=timestamp_ms,
            seq_num=seq_num,
            metadata=metadata or {}
        )


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
class AudioProcessingResult:
    """Complete result returned after processing an audio frame."""
    accepted: bool = True
    rejection_reason: Optional[str] = None
    output_frame: Optional[AudioFrame] = None
    output_bytes: Optional[bytes] = None
    metrics: AudioProcessingMetrics = field(default_factory=AudioProcessingMetrics)
    per_stage_timings_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class AudioPipelineConfig:
    """Configuration parameters for the integrated DSP pipeline."""
    sample_rate: int = 16000
    frame_size_samples: int = 256  # 256 samples = 16ms @ 16kHz
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
    
    Accepts 16 kHz 16-bit signed PCM mono audio in bytes, AudioFrame, list, or AudioBuffer formats,
    chains all validated DSP stages in frozen architecture order, and outputs processed
    PCM data alongside structured signal-quality, per-stage profiling, and host performance metrics.
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

    def process_frame(
        self,
        frame: AudioFrame,
        ref_frame: Optional[AudioFrame] = None
    ) -> AudioProcessingResult:
        """
        Primary interface: Process an AudioFrame through the integrated DSP pipeline.
        Returns a structured AudioProcessingResult with explicit acceptance/rejection
        and per-stage performance timings.
        """
        # 1. Validation & Pre-checks
        if frame is None:
            return AudioProcessingResult(
                accepted=False,
                rejection_reason="AudioFrame is None",
                output_frame=None,
                output_bytes=None,
                metrics=AudioProcessingMetrics(error="AudioFrame is None")
            )

        if frame.sample_rate != self.config.sample_rate:
            return AudioProcessingResult(
                accepted=False,
                rejection_reason=f"Unsupported sample rate: got {frame.sample_rate}, expected {self.config.sample_rate}",
                output_frame=None,
                output_bytes=None,
                metrics=AudioProcessingMetrics(
                    sample_rate=frame.sample_rate,
                    error=f"Unsupported sample rate: {frame.sample_rate}"
                )
            )

        if frame.channels != 1:
            return AudioProcessingResult(
                accepted=False,
                rejection_reason=f"Unsupported channel count: got {frame.channels}, expected 1 (mono)",
                output_frame=None,
                output_bytes=None,
                metrics=AudioProcessingMetrics(error=f"Unsupported channels: {frame.channels}")
            )

        if len(frame.pcm_data) == 0:
            return AudioProcessingResult(
                accepted=False,
                rejection_reason="Empty frame provided",
                output_frame=None,
                output_bytes=None,
                metrics=AudioProcessingMetrics(
                    sample_rate=self.config.sample_rate,
                    sample_count=0,
                    error="Empty frame"
                )
            )

        # 2. Run DSP stages with per-stage timing
        t0_total = time.perf_counter()
        per_stage_timings: Dict[str, float] = {}
        stages_executed: List[str] = []

        sanitized, had_nan_inf, val_err = self._validate_samples(frame.pcm_data)
        if val_err:
            return AudioProcessingResult(
                accepted=False,
                rejection_reason=val_err,
                output_frame=None,
                output_bytes=None,
                metrics=AudioProcessingMetrics(error=val_err)
            )

        in_buf = AudioBuffer(sanitized, self.config.sample_rate)
        in_rms = metrics.calculate_rms(in_buf)
        in_peak = metrics.calculate_peak(in_buf)
        in_dc = metrics.calculate_dc_offset(in_buf)
        current_buf = in_buf.copy()

        # Stage 1: DC Blocker
        if self.dc_blocker.enabled:
            ts0 = time.perf_counter()
            current_buf = self.dc_blocker.process(current_buf)
            per_stage_timings["DCBlocker"] = round((time.perf_counter() - ts0) * 1000.0, 4)
            stages_executed.append("DCBlocker")

        # Stage 2: High Pass Filter (80 Hz)
        if self.high_pass.enabled:
            ts0 = time.perf_counter()
            current_buf = self.high_pass.process(current_buf)
            per_stage_timings["HighPassFilter"] = round((time.perf_counter() - ts0) * 1000.0, 4)
            stages_executed.append("HighPassFilter")

        # Stage 3: AEC + DTD (Conditional)
        aec_status = "BYPASS_NO_REFERENCE"
        dt_frames = 0
        if self.aec.enabled and ref_frame is not None and len(ref_frame.pcm_data) > 0:
            ts0 = time.perf_counter()
            ref_sanitized, _, _ = self._validate_samples(ref_frame.pcm_data)
            ref_buf_clean = AudioBuffer(ref_sanitized, self.config.sample_rate)
            current_buf = self.aec.process_aec(current_buf, ref_buf_clean)
            per_stage_timings["NLMSAECWithDTD"] = round((time.perf_counter() - ts0) * 1000.0, 4)
            aec_status = "ACTIVE"
            dt_frames = self.aec.dt_detected_frames
            stages_executed.append("NLMSAECWithDTD")
        elif not self.aec.enabled:
            aec_status = "DISABLED"

        # Stage 4: Noise Suppression
        if self.noise_suppressor.enabled:
            ts0 = time.perf_counter()
            current_buf = self.noise_suppressor.process(current_buf)
            per_stage_timings["SpectralNoiseSuppression"] = round((time.perf_counter() - ts0) * 1000.0, 4)
            stages_executed.append("SpectralNoiseSuppression")

        # Stage 5: VAD
        vad_active = False
        vad_active_count = 0
        vad_total_count = 0
        if self.vad.enabled:
            ts0 = time.perf_counter()
            self.vad.process(current_buf.copy())
            per_stage_timings["VAD"] = round((time.perf_counter() - ts0) * 1000.0, 4)
            vad_total_count = len(self.vad.history)
            vad_active_count = sum(1 for h in self.vad.history if h.get("speech_active", False))
            vad_active = vad_active_count > 0 or self.vad.speech_active
            stages_executed.append("VAD")

        # Stage 6: AGC
        if self.agc.enabled:
            ts0 = time.perf_counter()
            current_buf = self.agc.process(current_buf)
            per_stage_timings["AGC"] = round((time.perf_counter() - ts0) * 1000.0, 4)
            stages_executed.append("AGC")

        # Stage 7: Limiter
        if self.limiter.enabled:
            ts0 = time.perf_counter()
            current_buf = self.limiter.process(current_buf)
            per_stage_timings["Limiter"] = round((time.perf_counter() - ts0) * 1000.0, 4)
            stages_executed.append("Limiter")

        t1_total = time.perf_counter()
        proc_time_ms = (t1_total - t0_total) * 1000.0
        frame_dur_ms = frame.duration_ms
        rt_factor = proc_time_ms / frame_dur_ms if frame_dur_ms > 0 else 0.0

        out_rms = metrics.calculate_rms(current_buf)
        out_peak = metrics.calculate_peak(current_buf)
        out_dc = metrics.calculate_dc_offset(current_buf)
        clip_pct = metrics.calculate_clipping_percentage(current_buf, self.config.limiter_threshold)
        clip_detected = clip_pct > 0.0 or out_peak >= self.config.limiter_threshold

        snr_est = None
        if in_rms > 0 and out_rms > 0:
            snr_est = round(20.0 * math.log10(max(out_rms, 1e-6) / max(in_rms, 1e-6)), 2)

        out_frame = AudioFrame(
            pcm_data=list(current_buf.data),
            sample_rate=frame.sample_rate,
            channels=1,
            sample_format="int16",
            timestamp_ms=frame.timestamp_ms,
            seq_num=frame.seq_num,
            metadata={**frame.metadata, "processed": True}
        )
        out_bytes = out_frame.to_bytes(self.config.limiter_threshold)

        pipeline_metrics = AudioProcessingMetrics(
            sample_rate=self.config.sample_rate,
            sample_count=len(sanitized),
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

        return AudioProcessingResult(
            accepted=True,
            rejection_reason=None,
            output_frame=out_frame,
            output_bytes=out_bytes,
            metrics=pipeline_metrics,
            per_stage_timings_ms=per_stage_timings
        )

    def process_buffer(
        self,
        mic_buffer: AudioBuffer,
        ref_buffer: Optional[AudioBuffer] = None
    ) -> Tuple[AudioBuffer, AudioProcessingMetrics]:
        """
        Direct AudioBuffer interface for backwards compatibility.
        """
        frame = AudioFrame(pcm_data=list(mic_buffer.data), sample_rate=mic_buffer.sample_rate)
        ref_frame = AudioFrame(pcm_data=list(ref_buffer.data), sample_rate=ref_buffer.sample_rate) if ref_buffer else None
        res = self.process_frame(frame, ref_frame)
        out_data = res.output_frame.pcm_data if res.output_frame else []
        return AudioBuffer(out_data, mic_buffer.sample_rate), res.metrics

    def process_pcm_bytes(
        self,
        mic_bytes: bytes,
        ref_bytes: Optional[bytes] = None
    ) -> Tuple[bytes, AudioProcessingMetrics]:
        """
        Convenience wrapper accepting and returning raw 16-bit signed PCM bytes.
        """
        frame = AudioFrame.from_bytes(mic_bytes, self.config.sample_rate)
        ref_frame = AudioFrame.from_bytes(ref_bytes, self.config.sample_rate) if ref_bytes is not None else None
        res = self.process_frame(frame, ref_frame)
        out_bytes = res.output_bytes or b""
        return out_bytes, res.metrics

    def process_pcm_samples(
        self,
        mic_samples: List[float],
        ref_samples: Optional[List[float]] = None
    ) -> Tuple[List[float], AudioProcessingMetrics]:
        """
        Convenience wrapper accepting and returning raw float sample lists.
        """
        frame = AudioFrame(pcm_data=list(mic_samples), sample_rate=self.config.sample_rate)
        ref_frame = AudioFrame(pcm_data=list(ref_samples), sample_rate=self.config.sample_rate) if ref_samples else None
        res = self.process_frame(frame, ref_frame)
        out_samples = res.output_frame.pcm_data if res.output_frame else []
        return out_samples, res.metrics

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


class AudioStreamAdapter:
    """
    Stream-level adapter that slices incoming continuous PCM byte streams into
    fixed-size frames, manages sequence continuity, buffers partial trailing data,
    and returns a stream of AudioProcessingResults.
    """

    def __init__(self, pipeline: Optional[IntegratedAudioPipeline] = None, frame_size: int = 256):
        self.pipeline = pipeline or IntegratedAudioPipeline()
        self.frame_size = frame_size  # samples per frame (256 @ 16kHz = 16ms)
        self.bytes_per_sample = 2
        self.frame_bytes_len = self.frame_size * self.bytes_per_sample
        
        self._partial_bytes = bytearray()
        self._seq_counter = 0
        
        # Stream statistics
        self.accepted_frames = 0
        self.processed_frames = 0
        self.rejected_frames = 0
        self.dropped_frames = 0

    def reset(self) -> None:
        """Reset the streaming buffer, sequence counter, and internal DSP pipeline."""
        self._partial_bytes.clear()
        self._seq_counter = 0
        self.accepted_frames = 0
        self.processed_frames = 0
        self.rejected_frames = 0
        self.dropped_frames = 0
        self.pipeline.reset()

    def push_raw_stream(
        self,
        raw_bytes: bytes,
        timestamp_base_ms: float = 0.0
    ) -> List[AudioProcessingResult]:
        """
        Append raw PCM bytes to stream, slice into fixed frame_size blocks,
        process through pipeline, and preserve any trailing partial bytes.
        """
        if not isinstance(raw_bytes, (bytes, bytearray)):
            raise TypeError(f"Expected bytes or bytearray, got {type(raw_bytes).__name__}")

        self._partial_bytes.extend(raw_bytes)
        results: List[AudioProcessingResult] = []

        while len(self._partial_bytes) >= self.frame_bytes_len:
            frame_chunk = bytes(self._partial_bytes[:self.frame_bytes_len])
            del self._partial_bytes[:self.frame_bytes_len]

            t_ms = timestamp_base_ms + (self._seq_counter * (self.frame_size / self.pipeline.config.sample_rate) * 1000.0)
            
            frame = AudioFrame.from_bytes(
                frame_chunk,
                sample_rate=self.pipeline.config.sample_rate,
                timestamp_ms=t_ms,
                seq_num=self._seq_counter
            )
            self._seq_counter += 1
            self.accepted_frames += 1

            res = self.pipeline.process_frame(frame)
            if res.accepted:
                self.processed_frames += 1
            else:
                self.rejected_frames += 1
            results.append(res)

        return results

    @property
    def buffered_bytes_count(self) -> int:
        """Return the number of unaligned partial bytes currently waiting in the buffer."""
        return len(self._partial_bytes)
