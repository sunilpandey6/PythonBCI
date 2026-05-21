# PythonBCI

A real-time, 16-channel EEG BCI processing backend for OpenBCI + Unity VR experiments, built on the Lab Streaming Layer (LSL) protocol.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Configuration (BCIConfig)](#configuration-bciconfig)
4. [Unity ↔ Python Protocol](#unity--python-protocol)
   - [Input: Unity Marker Format](#input-unity-marker-format)
   - [Output: JSON Message Schema](#output-json-message-schema)
   - [Code Reference Table](#code-reference-table)
5. [State Machine](#state-machine)
   - [Full Flow Diagram](#full-flow-diagram)
   - [Action → State Transitions](#action--state-transitions)
6. [Signal Processing Pipelines](#signal-processing-pipelines)
   - [Global EEG Preprocessing](#global-eeg-preprocessing)
   - [SSVEP Detection (ssvep.py)](#ssvep-detection-ssveppy)
   - [Object Classification (classifiers.py)](#object-classification-classifierspy)
7. [Threading Model](#threading-model)
8. [Installation & Setup](#installation--setup)
9. [Operation & CLI Arguments](#operation--cli-arguments)
10. [Unity Integration Guide](#unity-integration-guide)
11. [Monitoring Output](#monitoring-output)
12. [Troubleshooting](#troubleshooting)

---

## Overview

PythonBCI is a standalone Python daemon that bridges a **16-channel OpenBCI Cyton** EEG headset with a **Unity** VR experiment over a local network using LSL.

It handles two primary BCI paradigms simultaneously:
1. **SSVEP Flicker Detection**: Continuous real-time identification of a configurable target frequency using FBCCA (Filter-Bank Canonical Correlation Analysis). The target frequency can be updated live from the Unity Settings UI.
2. **Motor Imagery / Active State Classification**: A three-model ML system (ACTIVE, IMAGERY, MIXED) classifying user intent using a time-domain SVM pipeline. Predictions are emitted only when a configurable agreement and confidence threshold is met across a rolling window of epochs.

| Feature | Details |
|---|---|
| **EEG Hardware** | OpenBCI Cyton (16-ch) |
| **Communication** | Lab Streaming Layer (pylsl) |
| **SSVEP Detection** | FBCCA (Filter-Bank CCA), continuous sliding window |
| **Object Classification** | Vectorizer + StandardScaler + SVM (scikit-learn) |
| **Output Format** | JSON over LSL string stream |

---

## Project Structure

```text
PythonBCI/
├── main.py                  # CLI Entrypoint: parses arguments, starts BCIBackend
├── print_results.py         # Prints BCIBackend LSL stream output to the terminal
├── bci/                     # Modular Layered Architecture package
│   ├── domain/              # Domain Layer: Pure logic, no IO, no threading
│   │   ├── config.py        # Centralized BCIConfig dataclass
│   │   ├── codes.py         # BCICode IntEnum
│   │   ├── state_machine.py # State machine states and transitions logic
│   │   └── buffers.py       # Pure EEG and training epoch buffer structures
│   ├── signal/              # Signal Processing Layer
│   │   ├── preprocessing.py # Global EEG filtering functions
│   │   └── ssvep.py         # FBCCA SSVEPDetector and FlickerResult logic
│   ├── ml/                  # Machine Learning Layer
│   │   ├── classifiers.py   # Active, Imagery, and Mixed classifier wrappers
│   │   └── voting.py        # Prediction accumulator and stable voting logic
│   └── infrastructure/      # Infrastructure Layer: External systems and IO
│       ├── lsl_io.py        # pylsl StreamInlet, StreamOutlet, resolve_byprop setup
│       └── protocol.py      # BCI JSON message building and parsing protocol
└── README.md                # This documentation
```

---

## Configuration (BCIConfig)

`BCIConfig` is a `@dataclass` in `bci/domain/config.py` and is the **single source of truth** for all configurable parameters. Defaults live here; `argparse` reads from them so there is no duplication.

```python
@dataclass
class BCIConfig:
    target_freq: float = 15.0              # SSVEP stimulus frequency (Hz)
    sfreq: float = 250.0                   # EEG sampling rate (Hz)
    epoch_duration: float = 1.0            # Sliding window length (s)
    n_train_epochs: int = 30               # Ring buffer size per class
    detection_threshold: float = 0.4       # Min FBCCA score for FLICKER_DETECTED
    resolve_timeout: float = 10.0          # LSL stream resolve timeout (s)
    predict_accumulation_time: float = 3.0 # Prediction window length (s)
    predict_agreement_threshold: float = 0.75  # Min fraction of buffer agreeing
    predict_confidence_threshold: float = 0.7  # Min avg SVM confidence
    eeg_stream_name: Optional[str] = None  # Override EEG stream name (None = auto)
    marker_stream_name: Optional[str] = None   # Override Marker stream name
```

**Usage — programmatic (e.g. from a test script):**
```python
from bci.domain.config import BCIConfig
from bci.application.backend import BCIBackend

config = BCIConfig(target_freq=10.0, detection_threshold=0.35)
backend = BCIBackend(config)
backend.start()
```

**Usage — CLI (argparse reads defaults from BCIConfig so there is no duplication):**
```bash
python main.py --target-freq 10.0 --detection-threshold 0.35
```

> Prediction thresholds (`predict_agreement_threshold`, `predict_confidence_threshold`) and `predict_accumulation_time` are not exposed as CLI arguments — modify them directly in `BCIConfig` if you need to tune them.

---

## Unity ↔ Python Protocol

### Input: Unity Marker Format

Unity sends CSV log entries over an LSL string stream (type: `"Markers"`). Each sample is a single comma-separated string with exactly **six fields**:

`Time,Experiment,Phase,Event,Detail,Action`

| Field | Role | Example |
|---|---|---|
| **Time** | Timestamp (ignored by Python) | `2026-05-05 15:08:38.284` |
| **Experiment** | Experiment label (ignored) | `BCI` |
| **Phase** | Scene/phase label (ignored) | `TrainBCI` |
| **Event** | **Pass-through** — forwarded verbatim to output | `Flicker` |
| **Detail** | **Pass-through** — forwarded verbatim to output | `Door_Single` |
| **Action** | **State machine driver** — strictly controls Python logic | `Flicker_Start` |

> **Critical Rule**: Python **never** modifies or interprets `Event` or `Detail`. They are stored and echoed back verbatim in every outgoing message so Unity can match responses to its own logged events.

### Output: JSON Message Schema

Every packet Python sends to Unity is a single JSON string pushed over the `BCIBackend` LSL outlet (type: `"BCIResult"`). The schema is strictly typed:

**Flicker result:**
```json
{
  "Code":   100,
  "Event":  "Flicker",
  "Detail": "Door_Single",
  "Remark": {
    "Detected_Frequency": 15.0,
    "Confidence_Score":   0.93,
    "SSVEP_Present":      true,
    "FBCCA_Score":        0.93
  }
}
```

**Prediction result:**
```json
{
  "Code":   302,
  "Event":  "Predict Door Imagery",
  "Detail": "eye closed",
  "Remark": {
    "Model":               "IMAGERY",
    "Prediction":          "OBJ1",
    "Confidence":          0.96,
    "Agreement":           0.87,
    "Imagery_Prediction":  "None",
    "Imagery_Confidence":  0.0
  }
}
```

| Field | Owner | Description |
|---|---|---|
| `Code` | Python | One of the integer `BCICode`s (see below). |
| `Event` | Unity | Echoed back verbatim from the Unity marker log. |
| `Detail` | Unity | Echoed back verbatim from the Unity marker log. |
| `Remark` | Python | All ML / signal-processing findings, scores, and diagnostics. |

### Code Reference Table

| Code | Constant | Trigger Condition |
|------|----------|---------|
| **100** | `FLICKER_DETECTED` | SSVEP present at the target frequency |
| **101** | `FLICKER_NOT_DETECTED` | SSVEP absent (pushed on `Flicker_End`) |
| **201** | `ACTIVE_OBJ1_TRAIN_COMPLETE` | Active training epoch buffer full for Door 1 |
| **202** | `ACTIVE_OBJ2_TRAIN_COMPLETE` | Active training epoch buffer full for Door 2 |
| **203** | `IMAGERY_OBJ1_TRAIN_COMPLETE` | Imagery training epoch buffer full for Door 1 |
| **204** | `IMAGERY_OBJ2_TRAIN_COMPLETE` | Imagery training epoch buffer full for Door 2 |
| **300** | `ACTIVE_OBJ1_PREDICT` | ACTIVE model predicts Door 1 |
| **301** | `ACTIVE_OBJ2_PREDICT` | ACTIVE model predicts Door 2 |
| **302** | `IMAGERY_OBJ1_PREDICT` | IMAGERY model predicts Door 1 |
| **303** | `IMAGERY_OBJ2_PREDICT` | IMAGERY model predicts Door 2 |

---

## State Machine

The system logic is driven by the `Action` string (field 6) parsed from the Unity marker log.

### Full Flow Diagram

```text
              ┌──────────┐
              │   IDLE   │◄──────────────────────────────────────────────┐
              └────┬─────┘                                               │
                   │ (Action received from Unity via LSL)                │
    ┌──────────────┼────────────────────────────────┐                    │
    │              │                                │                    │
    ▼              ▼                                ▼                    │
SSVEP_TEST   TRAIN_ACTIVE_OBJ1/2           TRAIN_IMAGERY_OBJ1/2         │
    │         (sliding ring buffer)          (sliding ring buffer)       │
    │              │                                │                    │
    │   Continuous │                                │                    │
    │   FBCCA on   │  *_End marker received         │                    │
    │   sliding    │  → Transition to IDLE          │                    │
    │   windows    │                                │                    │
    │              │                                │                    │
    │  Flicker_End │          Train_End             │                    │
    └──────────────┴────────────────────────────────┘                    │
           │                      │                                      │
    _finalize_flicker()     _attempt_training()                          │
    Max-confidence           ACTIVE.fit()                                │
    pushed once              IMAGERY.fit()                               │
                             MIXED.fit()                                 │
                                  │                                      │
                 ┌────────────────┼────────────────┐                     │
                 ▼                ▼                ▼                     │
          PREDICT_ACTIVE  PREDICT_IMAGERY  PREDICT_MIXED                 │
                 │                │                │                     │
                 └────────────────┴────────────────┘                     │
                    Accumulate predictions until agreement               │
                    + confidence thresholds met → push 300-303           │
                    Predict_End ─────────────────────────────────────────┘
```

### Action → State Transitions

| Unity `Action` string | Resulting State | Notes |
|---|---|---|
| `Flicker_Start` | `SSVEP_TEST` | Enables continuous FBCCA sliding-window detection. Buffer is NOT cleared. |
| `Flicker_End` | `IDLE` | Finalises the flicker by pushing the max-confidence aggregated result. |
| `Training_Active_Door1_Start` | `TRAIN_ACTIVE_OBJ1` | Starts filling the active sliding buffer for class 0. |
| `Training_Active_Door1_End` | `IDLE` | Stops buffering Door 1 active data. |
| `Training_Active_Door1_Flicker_Start` | `TRAIN_ACTIVE_OBJ1` | Starts filling active sliding buffer for class 0 (flicker variant). |
| `Training_Active_Door1_Flicker_End` | `IDLE` | Stops buffering Door 1 active flicker data. |
| `Active_Training_Door2_Start` | `TRAIN_ACTIVE_OBJ2` | Starts filling the active sliding buffer for class 1. |
| `Active_Training_Door2_End` | `IDLE` | Stops buffering Door 2 active data. |
| `Training_Imagery_Door1_Start` | `TRAIN_IMAGERY_OBJ1` | Starts filling the imagery sliding buffer for class 0. |
| `Training_Imagery_Door1_End` | `IDLE` | Stops buffering Door 1 imagery data. |
| `Image_Training_Door2_Start` | `TRAIN_IMAGERY_OBJ2` | Starts filling the imagery sliding buffer for class 1 (older variant). |
| `Image_Training_Door2_End` | `IDLE` | Stops buffering Door 2 imagery data (older variant). |
| `Training_Imagery_Door2_Start` | `TRAIN_IMAGERY_OBJ2` | Starts filling the imagery sliding buffer for class 1 (newer unified name). |
| `Training_Imagery_Door2_End` | `IDLE` | Stops buffering Door 2 imagery data (newer unified name). |
| `Train_End` | `IDLE` | Triggers `_attempt_training()` — fits ACTIVE, IMAGERY and MIXED models. |
| `Predict_Start_Active` | `PREDICT_ACTIVE` | Begins accumulating ACTIVE model predictions. |
| `Predict_Start_Imagery` | `PREDICT_IMAGERY` | Begins accumulating IMAGERY model predictions. |
| `Start_predict` | `PREDICT_MIXED` | Begins accumulating MIXED model predictions. |
| `Predict_End` | `IDLE` | Stops prediction. |
| `Set_Target_Frequency` | — | Updates `target_freq` and `SSVEPDetector.target_freq` live, no state change. |
| `Eye_Closed` | — | Sets `is_eye_closed = True` and pauses epoch processing. Bypasses state transitions. |
| `Eye_Opened` | — | Sets `is_eye_closed = False` and resumes epoch processing. Bypasses state transitions. |


---

## Signal Processing Pipelines

### Global EEG Preprocessing

Applied to **every** incoming EEG epoch across **all** states before any method-specific processing:

```text
Raw 16-ch epoch (n_channels × n_samples)
    │
    ▼ Eye Status Check (Bypasses and ignores epoch if is_eye_closed is True)
    │
    ▼ 50 Hz & 100 Hz Notch filters (Multi-harmonic powerline filtering, Q=30, scipy.signal.iirnotch)
    │
    ▼ Dynamic Bandpass filter (4th-order Butterworth, SOS zero-phase, cutoffs vary by state)
    │
    └─► Pre-processed epoch → SSVEPDetector or Classifier
```

#### Eye-Status Gating
If the Unity action `"Eye_Closed"` is received, the preprocessor immediately returns `None` and the current epoch is skipped (early exit in the logic thread, preventing it from reaching the training buffers or classifier prediction accumulators). Once `"Eye_Opened"` is received, normal preprocessing resumes.

#### Multi-Harmonic Notch Filter
To combat powerline noise and its harmonics, the system applies two narrow notch filters sequentially at:
- **50 Hz** (fundamental)
- **100 Hz** (second harmonic)

Both filters are applied dynamically only if their target frequency is below the Nyquist frequency (`sfreq / 2.0`).

#### Dynamic State-Dependent Bandpass Filter
A 4th-order Butterworth bandpass filter is dynamically customized to isolate state-specific frequency bands and optimize SNR:
- **`SSVEP_TEST`**: **1.0 to 60.0 Hz** to capture early harmonics.
- **Active States** (`TRAIN_ACTIVE_OBJ1`, `TRAIN_ACTIVE_OBJ2`, `PREDICT_ACTIVE`): **3.0 to 60.0 Hz** to isolate focused intent and suppress slow eye-movement drifts.
- **Imagery/Mixed States** (`TRAIN_IMAGERY_OBJ1`, `TRAIN_IMAGERY_OBJ2`, `PREDICT_IMAGERY`, `PREDICT_MIXED`): **1.0 to 100.0 Hz** to capture deep high-gamma frequency waves, letting the 100 Hz notch handle powerline harmonics.
- **Fallback / IDLE**: **1.0 to 90.0 Hz**.

### SSVEP Detection (`bci/signal/ssvep.py`)

`SSVEPDetector` uses **FBCCA** (Filter-Bank CCA) to score each sliding epoch against the target frequency. Detection runs **continuously** in the `SSVEP_TEST` state — one score is computed every `step_size` seconds (default: 0.2 s). The EEG buffer is never cleared when entering this state.

**Detection Logic:**
1. On `Flicker_Start`: `SSVEP_TEST` state is entered. `_flicker_results` list is cleared.
2. Every 0.2 s: FBCCA is run on the current 1.0-second sliding window. The `FlickerResult` is appended to `_flicker_results`.
3. On `Flicker_End`: `_finalize_flicker()` takes the **maximum confidence** across all accumulated windows, decides `FLICKER_DETECTED (100)` vs `FLICKER_NOT_DETECTED (101)`, and pushes a **single aggregated result** to Unity.

**FBCCA Filter-Bank parameters (defaults):**

| Parameter | Value |
|---|---|
| `fbcca_num_bands` | 5 |
| `fbcca_a` | 1.25 |
| `fbcca_b` | 0.25 |
| `fbcca_band_width` | 8.0 Hz |
| `n_harmonics` | 3 |
| `occipital_channels` | [6, 7] (O1, O2) |

If `max(fbcca_score) >= detection_threshold` (default `0.4`), the result is `FLICKER_DETECTED (100)`. Otherwise, `FLICKER_NOT_DETECTED (101)`.

### Object Classification (`bci/ml/classifiers.py`)

Three independent classifiers are trained when `Train_End` is received. Each uses a **Vectorizer → StandardScaler → LinearSVC** time-domain pipeline.

| Classifier | Training Data | Use Case |
|---|---|---|
| `ActiveClassifier` | Active attention epochs (Door 1 vs Door 2) | `PREDICT_ACTIVE` state |
| `ImageryClassifier` | Motor imagery epochs (Door 1 vs Door 2) | `PREDICT_IMAGERY` state |
| `MixedClassifier` | Combined active + imagery epochs | `PREDICT_MIXED` state |

Training epoch buffers are **ring buffers** (`deque(maxlen=n_train_epochs)`), so they naturally contain the most recent `n_train_epochs` samples at all times.

**Stable Prediction (accumulation buffer):**

Predictions are not emitted immediately. A rolling `_predict_buffer` accumulates `(pred, conf)` pairs. A result is only pushed to Unity when:
- **Agreement** (fraction of buffer agreeing on the same class) ≥ `predict_agreement_threshold` (default: 0.75)
- **Average Confidence** across the buffer ≥ `predict_confidence_threshold` (default: 0.7)

This prevents noisy single-window misfires from reaching Unity.

---

## Threading Model

The backend is built for zero-blocking real-time processing using three daemon threads:

```text
┌─────────────────────────────────────────────────────────────────┐
│  Thread 1  (Thread-EEG)                                         │
│  pull_sample() → deque ring buffer (maxlen=epoch_samples)       │
│  Sets _eeg_ready Event when buffer is full                      │
├─────────────────────────────────────────────────────────────────┤
│  Thread 2  (Thread-Marker)                                      │
│  pull_sample(timeout=0.0)  ← NON-BLOCKING                       │
│  Parses CSV → bundles {action, event, detail} → queue.Queue     │
├─────────────────────────────────────────────────────────────────┤
│  Thread 3  (Thread-Logic)                                       │
│  Drains marker queue → _handle_marker_payload() → state change  │
│  Waits on _eeg_ready → snapshots epoch → _process_epoch()       │
│  Builds JSON → pushes via LSL outlet                            │
└─────────────────────────────────────────────────────────────────┘
```

> **Marker atomicity**: Thread 2 bundles all three marker fields (`action`, `event`, `detail`) into a single dict before enqueuing. Thread 3 (the logic thread) is the sole owner of `_last_unity_event` / `_last_unity_detail`, and updates them only for non-flicker transitions. This prevents the `Flicker_End` event from overwriting the `Flicker_Start` metadata that `_finalize_flicker()` needs to echo back to Unity.

> **Non-blocking marker pull**: Thread 2 uses `pull_sample(timeout=0.0)` so it **never blocks the GIL** waiting for network I/O. A `time.sleep(0.001)` yields the GIL at a stable ~1 kHz polling cadence.

---

## Installation & Setup

### Requirements

| Package | Version | Purpose |
|---|---|---|
| **Python** | ≥ 3.9 | Core runtime |
| **numpy** | ≥ 1.24 | Matrix operations |
| **scipy** | ≥ 1.11 | Filtering (Notch, Butterworth) |
| **pylsl** | ≥ 1.16 | Lab Streaming Layer networking |
| **mne** | ≥ 1.0 | EEG data utilities (CSP Vectorizer used by prediction pipeline) |
| **scikit-learn** | ≥ 1.2 | SVM classifiers and preprocessing |

### Environment Setup

```bash
# 1. Navigate to the PythonBCI directory
cd /path/to/PythonBCI

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install numpy scipy pylsl mne scikit-learn
```

> **OpenBCI Hardware**: Ensure the OpenBCI GUI (or BrainFlow) is running and broadcasting a `type="EEG"` LSL stream before launching `main.py`. If no stream is detected within `--resolve-timeout` seconds, the backend falls back to generating simulated Gaussian noise so the pipeline logic can still be tested.

---

## Operation & CLI Arguments

### Command Line Execution

```bash
# Default settings: 15 Hz target, 250 Hz sampling, 1-second epochs
python main.py

# Custom: different target frequency, 2-second analysis window
python main.py --target-freq 10 --epoch-duration 2

# Verbose debug output (shows per-window FBCCA scores)
python main.py --log-level DEBUG

# Send test heartbeat messages every 2s to verify Unity connection
python main.py --test-mode
```

### Configuration Reference

| Argument | Default | Description |
|---|---|---|
| `--target-freq` | `15.0` | SSVEP stimulus frequency to detect (Hz). Can also be updated live by Unity's `Set_Target_Frequency` marker. |
| `--sfreq` | `250.0` | EEG hardware sampling rate (Hz). |
| `--epoch-duration` | `1.0` | Length of each sliding EEG analysis window (s). |
| `--n-train-epochs` | `30` | Ring buffer size per class per object during training. |
| `--detection-threshold` | `0.4` | Minimum FBCCA score for `FLICKER_DETECTED`. |
| `--resolve-timeout` | `10.0` | Seconds to search for each LSL stream before falling back to simulation. |
| `--log-level` | `INFO` | Logging verbosity (`DEBUG` / `INFO` / `WARNING` / `ERROR`). |
| `--test-mode` | `false` | Emit dummy heartbeat messages every 2 s for Unity connection testing. |

---

## Unity Integration Guide

### LSL Streams Required

| Direction | Stream Name | Stream Type | Format |
|---|---|---|---|
| Unity → Python | *(any)* | `"Markers"` | 1-channel string CSV |
| Python → Unity | `"BCIBackend"` | `"BCIResult"` | 1-channel JSON string |

### Receiving BCI Results in Unity

Your `LSLCommunicationManager` should resolve the `BCIBackend` outlet and parse each JSON message:

**Data Classes:**
```csharp
public class BCIMessage {
    public int    Code;
    public string Event;
    public string Detail;
    public BCIRemark Remark;
}

public class BCIRemark {
    // SSVEP fields
    public float  Detected_Frequency;
    public float  Confidence_Score;
    public bool   SSVEP_Present;
    public float  FBCCA_Score;
    // Prediction fields
    public string Model;
    public string Prediction;
    public float  Confidence;
    public float  Agreement;
    public string Imagery_Prediction;
    public float  Imagery_Confidence;
}
```

**Flicker Response Pattern (BB.cs / OB.cs):**
```csharp
// Unity validates the echo by matching Event + Detail to what it sent
private void HandleFlickerLSL(BCIMessage msg)
{
    if (msg.Event != lastEvent || msg.Detail != lastDetail) return;
    if (msg.Code == (int)BCICommand.FlickerDetected)
        ExecuteAction();
}
```

**Prediction Response Pattern (Test3D.cs):**
```csharp
public void HandlePredictionLSL(BCIMessage msg)
{
    if (door1 != null && msg.Code == (int)door1.doorCode)
        door1.TriggerInteraction();
    else if (door2 != null && msg.Code == (int)door2.doorCode)
        door2.TriggerInteraction();

    // Signal the end of prediction phase back to Python
    LSL_Logger.Instance?.LogEvent("Predict_End", "Prediction_Phase", "Predict_End");
}
```

---

## Monitoring Output

To monitor the outgoing BCI Result JSON stream in real time, you can run the `print_results.py` helper script:

```bash
# Watch the JSON result stream in real time
python print_results.py
```

---

## Troubleshooting

### "No EEG LSL stream found"
- Confirm the OpenBCI GUI Networking panel has **LSL → Start** active.
- List all visible streams: `python -c "from pylsl import resolve_streams; print(resolve_streams())"`
- Check firewall / VPN — LSL uses multicast UDP on port `16571`.
- *If hardware is unavailable, Python automatically generates simulated Gaussian noise so you can test the pipeline logic regardless.*

### "No Marker LSL stream found"
- Confirm the Unity project has `LSL_Logger` active and `LSLCommunicationManager` set to stream type `"Markers"`.
- Ensure Unity and Python are on the same local network subnet or loopback (`127.0.0.1`).

### SSVEP is rarely detected or throws false positives
- **Rarely detected:** Lower `--detection-threshold` (e.g. `0.35`).
- **False positives:** Raise `--detection-threshold` (e.g. `0.55`).
- Enable `--log-level DEBUG` to see per-window `FBCCA_Score` values in the terminal.
- Check `occipital_channels` indices in `SSVEPDetector` match your electrode layout (default: `[6, 7]` = O1, O2 on a standard 8-ch Cyton mapping).
- Verify the target frequency matches the Unity flicker setting. The `Set_Target_Frequency` action from the Settings UI updates it live — check for a `[Logic] Target frequency set to X Hz` log line on startup and after each settings change.

### Flicker result is `101` even with high FBCCA scores
- The result pushed on `Flicker_End` is the **maximum FBCCA score** across all sliding windows during the flicker window. If no window exceeded the threshold, `101` is sent.
- If the `Flicker_End` marker arrives before the buffer fills to `epoch_samples` (e.g. very short flicker duration), the window may be skipped. Try increasing `--epoch-duration` to `0.5` or ensuring flicker duration > epoch duration.

### Predictions are not emitted
- Verify `Train_End` was received **after** both Door 1 and Door 2 epochs were buffered.
- Check logs for `[Logic] ACTIVE/IMAGERY/MIXED training skipped — insufficient epochs`. The ring buffer needs `n_train_epochs` samples per class.
- Predictions are only emitted when `agreement >= 0.75` AND `avg_confidence >= 0.7`. If the model is uncertain, no result will be pushed. Tune `predict_agreement_threshold` and `predict_confidence_threshold` in the `BCIConfig` dataclass.

### High CPU usage
- Reduce `--sfreq` or decrease `--epoch-duration` to lower the processing rate.
- Thread 2 polls at ~1 kHz. Adjust `time.sleep(0.001)` in `_marker_ingestion_loop` to `0.005` to trade a small amount of marker latency for CPU headroom.
