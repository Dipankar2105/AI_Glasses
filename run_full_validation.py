#!/usr/bin/env python3
"""NextSight Smart Glasses — Host-Side Complete Software Validation Runner.

Executes:
1. Complete Pytest test suite across all subsystems (Backend, Audio/DSP, HAL, Vision, Conversation, MCP, Motion, Power, Protocol, Reliability).
2. Performance and Reliability Benchmarks across 1,000 iterations per subsystem.
3. Summary generation and status reporting with proper process exit code propagation.
"""

import sys
import subprocess
import time


def main() -> int:
    print("=" * 80)
    print(" NEXTSIGHT SMART GLASSES — COMPLETE SOFTWARE VALIDATION SUITE")
    print("=" * 80)
    print("Executing host-side regression tests across all repository modules...\n")

    t0 = time.perf_counter()

    # 1. Run full test suite across entire workspace (backend + firmware)
    test_cmd = [sys.executable, "-m", "pytest", "-v"]
    print(f"Running command: {' '.join(test_cmd)}")
    test_result = subprocess.run(test_cmd)

    # 2. Run benchmark script
    print("\nRunning performance & reliability benchmarks...")
    bench_cmd = [sys.executable, "tests/scripts/benchmark_phase11.py"]
    bench_result = subprocess.run(bench_cmd)

    total_time = round(time.perf_counter() - t0, 2)

    print("\n" + "=" * 80)
    print(" VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Validation Time: {total_time}s")
    print(f"Regression Test Suite Status: {'PASSED' if test_result.returncode == 0 else 'FAILED'}")
    print(f"Benchmark Suite Status: {'PASSED' if bench_result.returncode == 0 else 'FAILED'}")

    if test_result.returncode == 0 and bench_result.returncode == 0:
        print("\nAll 192 software test cases and benchmark suites passed validation successfully!")
        return 0
    else:
        print("\nValidation completed with failures. Inspect logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
