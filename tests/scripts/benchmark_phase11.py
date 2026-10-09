"""Reproducible performance benchmark script for Phase 11."""

import os
import sys
import time
import json
import statistics
from typing import Dict, List, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.protocol.contracts import PacketType
from backend.protocol.framing import ProtocolFraming
from backend.motion.processor import MotionProcessor
from backend.motion.contracts import IMUSample
from backend.motion.touch import TouchProcessor
from backend.motion.contracts import TouchSample
from backend.power.policy import PowerThermalPolicyManager
from backend.power.contracts import BatteryTelemetry, ThermalTelemetry
from backend.conversation.session import SessionManager
from backend.reliability.orchestrator import SystemIntegrationOrchestrator


def compute_metrics(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute standard latency statistics in milliseconds."""
    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)
    return {
        "iterations": n,
        "cold_start_ms": round(sorted_lats[0], 4) if n > 0 else 0.0,
        "mean_ms": round(statistics.mean(sorted_lats), 4),
        "median_ms": round(statistics.median(sorted_lats), 4),
        "p95_ms": round(sorted_lats[int(n * 0.95)], 4) if n > 0 else 0.0,
        "p99_ms": round(sorted_lats[int(n * 0.99)], 4) if n > 0 else 0.0,
        "min_ms": round(sorted_lats[0], 4),
        "max_ms": round(sorted_lats[-1], 4),
    }


def benchmark_protocol_framing(iterations: int = 1000) -> Dict[str, Any]:
    payload = b'{"device_id":"nextsight-01","battery_pct":85.0,"temp_c":34.5}'
    latencies = []

    for i in range(iterations):
        t0 = time.perf_counter()
        encoded = ProtocolFraming.encode_message(
            packet_type=PacketType.TELEMETRY,
            payload=payload,
            sequence_number=i,
            timestamp_ms=i * 10,
        )
        decoded = ProtocolFraming.decode_message(encoded)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    stats = compute_metrics(latencies)
    stats["throughput_ops_per_sec"] = round(iterations / (sum(latencies) / 1000.0), 1)
    return stats


def benchmark_motion_processing(iterations: int = 1000) -> Dict[str, Any]:
    processor = MotionProcessor()
    latencies = []

    for i in range(iterations):
        sample = IMUSample(
            timestamp=i * 0.01,
            ax=0.01,
            ay=0.02,
            az=1.0,
            gx=0.5,
            gy=55.0,
            gz=0.1,
        )
        t0 = time.perf_counter()
        event = processor.process_sample(sample)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    stats = compute_metrics(latencies)
    stats["throughput_samples_per_sec"] = round(iterations / (sum(latencies) / 1000.0), 1)
    return stats


def benchmark_power_policy(iterations: int = 1000) -> Dict[str, Any]:
    manager = PowerThermalPolicyManager()
    latencies = []

    for i in range(iterations):
        bat = BatteryTelemetry(voltage_volts=3.8, percentage=75.0, timestamp=float(i))
        therm = ThermalTelemetry(temperature_celsius=35.0, timestamp=float(i))
        t0 = time.perf_counter()
        manager.update_battery_telemetry(bat, current_time=float(i))
        manager.update_thermal_telemetry(therm, current_time=float(i))
        snapshot = manager.get_snapshot(current_time=float(i))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    stats = compute_metrics(latencies)
    stats["throughput_cycles_per_sec"] = round(iterations / (sum(latencies) / 1000.0), 1)
    return stats


def benchmark_conversation_session(iterations: int = 1000) -> Dict[str, Any]:
    from backend.conversation.models import MessageRole
    sm = SessionManager()
    latencies = []

    for i in range(iterations):
        sess_id = f"session-{i % 10}"
        t0 = time.perf_counter()
        sm.add_message(session_id=sess_id, role=MessageRole.USER, content="Hello assistant")
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    stats = compute_metrics(latencies)
    stats["throughput_ops_per_sec"] = round(iterations / (sum(latencies) / 1000.0), 1)
    return stats


def benchmark_end_to_end_orchestrator(iterations: int = 1000) -> Dict[str, Any]:
    orchestrator = SystemIntegrationOrchestrator()
    payload = json.dumps({"gesture": "TAP", "confidence": 1.0}).encode("utf-8")
    packet = ProtocolFraming.encode_message(
        packet_type=PacketType.TOUCH_EVENT,
        payload=payload,
        sequence_number=1,
        timestamp_ms=1000,
    )
    latencies = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        result = orchestrator.process_incoming_device_packet(packet)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    stats = compute_metrics(latencies)
    stats["throughput_events_per_sec"] = round(iterations / (sum(latencies) / 1000.0), 1)
    return stats


def run_all_benchmarks() -> Dict[str, Any]:
    results = {
        "timestamp": time.time(),
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "benchmarks": {
            "protocol_framing": benchmark_protocol_framing(),
            "motion_processing": benchmark_motion_processing(),
            "power_policy": benchmark_power_policy(),
            "conversation_session": benchmark_conversation_session(),
            "end_to_end_orchestrator": benchmark_end_to_end_orchestrator(),
        },
    }

    out_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../results/phase11-performance-reliability.json")
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Benchmark results successfully written to: {out_path}")
    return results


if __name__ == "__main__":
    res = run_all_benchmarks()
    print(json.dumps(res, indent=2))
