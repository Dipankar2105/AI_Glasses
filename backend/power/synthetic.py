"""Deterministic synthetic trace generators for power and thermal management testing."""

from typing import List
from backend.power.contracts import BatteryTelemetry, ThermalTelemetry


def generate_battery_discharge_trace(
    start_pct: float = 100.0,
    end_pct: float = 0.0,
    steps: int = 100,
    step_duration_s: float = 1.0,
    start_time: float = 1000.0,
) -> List[BatteryTelemetry]:
    """Generate deterministic linear battery discharge telemetry sequence."""
    traces = []
    pct_step = (end_pct - start_pct) / max(1, steps - 1)

    for i in range(steps):
        pct = max(0.0, min(100.0, start_pct + (i * pct_step)))
        # Approximate 1S LiPo voltage curve: 3.3V (0%) to 4.2V (100%)
        voltage = 3.3 + (pct / 100.0) * 0.9
        ts = start_time + (i * step_duration_s)
        traces.append(
            BatteryTelemetry(
                voltage_volts=round(voltage, 3),
                percentage=round(pct, 1),
                is_charging=False,
                timestamp=ts,
            )
        )
    return traces


def generate_thermal_ramp_trace(
    start_temp_c: float = 25.0,
    peak_temp_c: float = 75.0,
    cooling_temp_c: float = 35.0,
    ramp_up_steps: int = 20,
    cooling_steps: int = 20,
    step_duration_s: float = 0.5,
    start_time: float = 1000.0,
) -> List[ThermalTelemetry]:
    """Generate thermal heating and subsequent cooling profile."""
    traces = []
    current_time = start_time

    # Ramp up
    up_step = (peak_temp_c - start_temp_c) / max(1, ramp_up_steps)
    for i in range(ramp_up_steps):
        temp = start_temp_c + (i * up_step)
        traces.append(
            ThermalTelemetry(
                temperature_celsius=round(temp, 2),
                timestamp=current_time,
            )
        )
        current_time += step_duration_s

    # Cooling down
    down_step = (peak_temp_c - cooling_temp_c) / max(1, cooling_steps)
    for i in range(cooling_steps):
        temp = peak_temp_c - (i * down_step)
        traces.append(
            ThermalTelemetry(
                temperature_celsius=round(temp, 2),
                timestamp=current_time,
            )
        )
        current_time += step_duration_s

    return traces
