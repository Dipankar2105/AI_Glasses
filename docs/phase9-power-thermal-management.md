# NextSight — Phase 9: Power and Thermal Management Foundation

**Status:** Completed (Software-Only Modeled & Simulated)  
**Date:** 2026-10-09  
**Hardware Status:** Unavailable (Modeled estimates & synthetic telemetry only; no physical hardware measurements claimed)

---

## 1. Overview & Objectives

Phase 9 implements the software foundation for battery power modeling, state-based power policies, thermal throttling safeguards, and resource-aware workload scheduling for the NextSight smart glasses platform (target architecture: Seeed Studio XIAO ESP32-S3 Sense + OV3660 + MAX98357A + MPU-6050 + 1S LiPo).

### Core Goals:
1. Establish typed data contracts for battery and thermal telemetry.
2. Implement component-level energy estimation models based on official hardware specifications.
3. Build a deterministic policy manager with threshold hysteresis to prevent rapid cycling.
4. Implement workload prioritization (gating non-essential vision and background tasks during low battery or high thermal conditions while preserving safety-critical functions).
5. Enforce staleness and invalid telemetry safeguards.

---

## 2. Power Modeling & Modeled Budget

> **NOTE:** All values are engineering estimates derived from component datasheets at 3.3V supply rail. Real hardware bench testing must update these estimates once prototypes are available.

### Peripheral Power Consumption Breakdown

| Peripheral | Component | Standby Current (mA) | Active Current (mA) | Voltage (V) | Active Power (mW) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Compute Core** | ESP32-S3 Dual-Core 240MHz | 2.0 (Light sleep) | 45.0 | 3.3 | 148.5 mW |
| **Wi-Fi Radio** | 802.11 b/g/n (Tx/Rx avg) | 0.0 (Power save) | 180.0 | 3.3 | 594.0 mW |
| **Camera** | OV3660 Sensor | 0.1 (Standby) | 70.0 | 3.3 | 231.0 mW |
| **Microphone** | I2S MEMS Mic | 0.01 | 1.5 | 3.3 | 4.95 mW |
| **Audio Speaker** | MAX98357A + Speaker | 0.01 (Shutdown) | 120.0 | 3.3 | 396.0 mW |
| **IMU** | MPU-6050 (6-DOF) | 0.005 (Sleep) | 3.8 | 3.3 | 12.54 mW |
| **Storage** | MicroSD Card | 0.2 | 45.0 | 3.3 | 148.5 mW |

### Operating State Power Estimates & Modeled Battery Runtime

Assuming a nominal **500 mAh / 3.7V (1850 mWh)** single-cell LiPo battery with 90% power conversion efficiency:

| Operating State | Active Subsystems | Modeled Power (mW) | Modeled Continuous Runtime (Hours) |
| :--- | :--- | :--- | :--- |
| **IDLE** | ESP32 Light Sleep, IMU Active, Radios Standby | ~20.2 mW | ~82.4 hours |
| **LISTENING** | ESP32 Active, Wi-Fi Active, Mic Active, IMU | ~760.0 mW | ~2.19 hours |
| **CAPTURE** | ESP32 Active, Wi-Fi Active, Camera Active, IMU | ~991.0 mW | ~1.68 hours |
| **PROCESSING** | ESP32 Active, Wi-Fi Active, IMU | ~755.0 mW | ~2.21 hours |
| **SPEAKING** | ESP32 Active, Wi-Fi Active, Speaker Active, IMU | ~1151.0 mW | ~1.45 hours |
| **LOW_POWER** | ESP32 Deep/Light Sleep Duty Cycle, Low-rate IMU | ~13.6 mW | ~122.4 hours |
| **CRITICAL_SHUTDOWN**| All Subsystems Power-Gated | ~0.0 mW | Standby / Safe Cutoff |

---

## 3. Thresholds & Hysteresis State Machines

To prevent high-frequency flapping between states caused by battery internal resistance voltage drop or thermal sensor jitter:

### Battery Thresholds
- **Low Battery Warning Entry:** $\le 15\%$ ($3.60\text{V}$) $\to$ Forces `LOW_POWER` state request, disables vision capture & background sync.
- **Low Battery Warning Recovery:** $> 20\%$ ($3.70\text{V}$) $\to$ Restores normal operations.
- **Critical Battery Shutdown Entry:** $\le 5\%$ ($3.40\text{V}$) $\to$ Emergency transition to `CRITICAL_SHUTDOWN`.
- **Critical Battery Recovery:** $> 10\%$ ($3.55\text{V}$).

### Thermal Thresholds (ESP32-S3 Internal Die / Enclosure Sensor)
- **Warm State:** $\ge 45^\circ\text{C}$ entry, $< 40^\circ\text{C}$ recovery.
- **Thermal Throttling (HOT_THROTTLED):** $\ge 55^\circ\text{C}$ entry, $< 48^\circ\text{C}$ recovery $\to$ Reduces vision capture frame rate / gates non-urgent cloud queries.
- **Critical Thermal Emergency (CRITICAL):** $\ge 70^\circ\text{C}$ entry, $< 60^\circ\text{C}$ recovery $\to$ Immediate peripheral power-gating and shutdown.

### Telemetry Staleness & Validation
- Battery telemetry timeout: $30.0\text{s}$ (marked `STALE` if unrefreshed).
- Thermal telemetry timeout: $15.0\text{s}$ (marked `STALE` if unrefreshed).

---

## 4. Workload Priority Matrix

The system evaluates workload permissions via `can_execute_workload(priority)`:

| Workload Priority | Normal State | Low Battery ($<15\%$) | Thermal Throttled ($>55^\circ\text{C}$) | Critical Emergency |
| :--- | :--- | :--- | :--- | :--- |
| **SAFETY_CRITICAL** | Allowed | Allowed | Allowed | Allowed |
| **TIME_SENSITIVE_AUDIO** | Allowed | Allowed | Allowed | Denied |
| **SENSOR_PROCESSING** | Allowed | Allowed | Allowed | Denied |
| **VISION_CAPTURE** | Allowed | **Denied** | **Denied** | Denied |
| **BACKGROUND_SYNC** | Allowed | **Denied** | **Denied** | Denied |

---

## 5. Test Suite & Evidence

Test Suite: [`backend/tests/test_phase9_power_thermal.py`](file:///c:/Users/Routewise/AI_Glasses/backend/tests/test_phase9_power_thermal.py) & [`backend/tests/test_phase9.py`](file:///c:/Users/Routewise/AI_Glasses/backend/tests/test_phase9.py)
- Total Tests: 10
- Passing: 10 (100%)
- Failures: 0
- Execution Time: 0.05s

---

## 6. What Must Be Validated on Real Hardware
1. **Coulomb Counting / Fuel Gauge Calibration:** Real discharge curves of 1S LiPo under high Wi-Fi burst currents.
2. **Thermal Dissipation:** Glasses temple enclosure thermal rise during continuous OV3660 camera streaming and audio playback.
3. **Power-Gating Transistor Leakage:** Standby current through P-channel MOSFET switches gating OV3660 and MAX98357A.
