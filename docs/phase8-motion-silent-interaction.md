# Phase 8 — Motion and Silent Interaction Foundation

**Status:** Complete  
**Decision:** **ACCEPT**  
**Date:** October 9, 2026  
**Environment:** Python 3.14.6, Windows 11 (AMD64)  
**Repository Starting Checkpoint:** `72ba46d`  
**Evaluator:** AI Glasses Engineering Team  

---

## 1. Executive Summary & Objective

In **Phase 8**, we designed, implemented, and thoroughly tested a hardware-independent **Motion and Silent Interaction Subsystem** for NextSight smart glasses.

### Key Capabilities Built:
1. **Typed Sensor Contracts & Coordinate Frames:** Defined standard 6-DOF IMU (`IMUSample`) and capacitive touch (`TouchSample`) contracts with strict numerical bounds and monotonic timestamp validation.
2. **Deterministic Motion Processor:** Implemented real-time smoothing, gravity compensation, stationary vs. moving classification, and bi-phasic oscillation gesture detection (Head Nod, Head Shake, Head Tilt).
3. **Capacitive Touch State Machine:** Created a robust touch state machine with glitch debouncing ($30\text{ms}$), single tap ($< 350\text{ms}$), double tap ($< 300\text{ms}$ inter-tap gap), and long press ($\ge 600\text{ms}$).
4. **Decoupled Interaction Dispatcher:** Safe mapping of recognized gestures to application intents (`CONFIRM`, `DISMISS_OR_CANCEL`, `TRIGGER_SCENE_ANALYSIS`, `START_LISTENING`) with cooldown suppression and service-availability checks.
5. **Hardware Constraints Preserved:**
   - Physical sensors were unavailable; verified entirely on deterministic synthetic traces.
   - OCR remains explicitly **ON HOLD**; frozen Phase 4B vision code (`backend/vision/`) remains untouched.

---

## 2. Sensor Coordinate Frame & Contracts

```
                +Z (Upward - Yaw Axis)
                 ^
                 |
                 |      +Y (Forward - Pitch Axis)
                 |     /
                 |    /
                 |   /
                 +---------------> +X (Right Temple - Roll Axis)
```

| Sensor Signal | Units | Nominal Range | Typical Sampling Rate | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Accelerometer $(a_x, a_y, a_z)$** | $g$ | $\pm 16.0g$ | $50\text{ Hz}$ | $1g \approx 9.80665\text{ m/s}^2$. Total magnitude at rest $\approx 1.0g$. |
| **Gyroscope $(\omega_x, \omega_y, \omega_z)$** | $\text{deg/s}$ | $\pm 2000.0\text{ deg/s}$ | $50\text{ Hz}$ | $\omega_x = \text{roll rate}$, $\omega_y = \text{pitch rate}$, $\omega_z = \text{yaw rate}$. |
| **Capacitive Touch** | boolean | `{0, 1}` | $100\text{ Hz}$ | High when temple frame electrode is contacted. |

---

## 3. Gesture & Motion Detection Algorithms

### A. Low-Pass Exponential Smoothing
$$y_k = \alpha \cdot x_k + (1 - \alpha) \cdot y_{k-1} \quad (\alpha = 0.40)$$
Filters out high-frequency micro-tremors and gait vibration.

### B. Stationary vs. Moving Classification
- Dynamic acceleration: $a_{\text{dyn}} = |\|\vec{a}\| - 1.0g| < 0.08g$
- Angular velocity: $\|\vec{\omega}\| < 15.0\text{ deg/s}$
- If both conditions hold over the recent window, the head is classified as `STATIONARY`.

### C. Bi-Phasic Gesture Patterns
- **Head Nod:** Pitch angular velocity $\omega_y$ crossing positive peak ($> +40\text{ deg/s}$) followed by negative peak ($< -40\text{ deg/s}$) with temporal peak separation $\Delta t \in [0.10s, 0.65s]$.
- **Head Shake:** Yaw angular velocity $\omega_z$ crossing positive peak ($> +45\text{ deg/s}$) followed by negative peak ($< -45\text{ deg/s}$) with peak separation $\Delta t \in [0.10s, 0.65s]$.
- **Head Tilt:** Sustained roll rate $\omega_x > 60\text{ deg/s}$ for $> 0.35s$.

---

## 4. Capacitive Touch State Machine

```
      +--------------------+
      |        IDLE        |
      +---------+----------+
                | press (t0)
                v
      +--------------------+   hold >= 600ms   +-----------------------+
      |      PRESSED       | ----------------> | Emits: LONG_PRESS     |
      +---------+----------+                   +-----------------------+
                | release (< 350ms)
                v
      +-----------------------------+
      |   WAITING_FOR_DOUBLE_TAP    |
      +----+-------------------+----+
           |                   |
           | gap > 300ms       | press (t <= 300ms)
           v                   v
+--------------------+   +-----------------------+
| Emits: TAP         |   | Emits: DOUBLE_TAP     |
+--------------------+   +-----------------------+
```

- **Debounce:** Ignores capacitive glitches $< 30\text{ms}$.
- **Duplicate Suppression:** Second tap release resets directly to `IDLE` without firing a trailing single tap.

---

## 5. Interaction Dispatcher Matrix

| Input Gesture | Mapped Intent | Target Service Action | Failure / Fallback Behavior |
| :--- | :--- | :--- | :--- |
| **Touch: Single Tap** | `CONFIRM` | `InteractionState: CONFIRMATION_SIGNAL_EMITTED` | Safe default |
| **Touch: Double Tap** | `TRIGGER_SCENE_ANALYSIS` | `VisionService: SCENE_ANALYSIS_DISPATCHED` | Aborts if vision pipeline not ready |
| **Touch: Long Press** | `DISMISS_OR_CANCEL` | `InteractionState: DISMISS_SIGNAL_EMITTED` | Dismisses active prompt |
| **Head: Nod** | `CONFIRM` | `InteractionState: CONFIRMATION_SIGNAL_EMITTED` | Safe confirmation |
| **Head: Shake** | `DISMISS_OR_CANCEL` | `InteractionState: DISMISS_SIGNAL_EMITTED` | Dismisses active prompt |
| **Head: Tilt** | `START_LISTENING` | `SessionManager: LISTENING_SESSION_INITIALIZED` | Creates conversation session |

---

## 6. Automated Verification & Regression Results

### Focused Test Suite (`backend/tests/test_phase8_motion_interaction.py`):
16 / 16 passed (100%):
- IMU sample numeric bounds and timestamp monotonicity validation.
- Stationary noise suppression ($0$ false gestures on Gaussian noise).
- Slow head movement suppression ($0$ false gestures on $< 15\text{ deg/s}$ sweeps).
- Intentional nod and shake recognition ($> 0.5$ confidence).
- Cooldown suppression of rapid repeated triggers.
- Touch glitch debouncing ($10\text{ms}$ spikes ignored).
- Single tap, double tap, and long press resolution.
- Dispatcher intent routing and service availability checking.

### Complete Backend Regression Suite:
```
================== 146 passed, 1 warning in 60.85s ==================
```

---

## 7. Physical Hardware Calibration Roadmap

When physical MPU-6050 and capacitive sensor hardware become available in Phase 10:
1. **Mounting Angle Offset:** Calibrate static pitch/roll biases resulting from glasses frame temple curvature.
2. **Capacitive Threshold Tuning:** Measure raw touch ADC thresholds across varying skin contact impedances (dry skin, sweat, gloves).
3. **Gait Filtering:** Record walking/running traces to verify gait step frequency ($\sim 1.8\text{ Hz}$) notch filtering.
