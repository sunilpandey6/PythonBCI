# PythonBCI

A real-time, 16-channel EEG BCI processing backend for OpenBCI + Unity VR experiments, built on the Lab Streaming Layer (LSL) protocol.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Unity ↔ Python Protocol](#unity--python-protocol)
   - [Input: Unity Marker Format](#input-unity-marker-format)
   - [Output: JSON Message Schema](#output-json-message-schema)
   - [Code Reference Table](#code-reference-table)
4. [State Machine](#state-machine)
   - [Full Flow Diagram](#full-flow-diagram)
   - [Action → State Transitions](#action--state-transitions)
5. [Signal Processing Pipelines](#signal-processing-pipelines)
   - [Global EEG Preprocessing](#global-eeg-preprocessing)
   - [SSVEP Detection (flicker.py)](#ssvep-detection-flickerpy)
   - [Object Classification (prediction.py)](#object-classification-predictionpy)
6. [Threading Model](#threading-model)
7. [Installation & Setup](#installation--setup)
8. [Operation & CLI Arguments](#operation--cli-arguments)
9. [Unity Integration Guide](#unity-integration-guide)
10. [Testing the Connection](#testing-the-connection)
11. [Troubleshooting](#troubleshooting)

---

## Overview

PythonBCI is a standalone Python daemon that bridges a **16-channel OpenBCI Cyton** EEG headset with a **Unity** VR experiment over a local network using LSL.

It handles two primary BCI paradigms simultaneously:
1. **SSVEP Flicker Detection**: Real-time identification of target frequencies using an ensemble of FFT, Welch PSD, and Canonical Correlation Analysis (CCA).
2. **Motor Imagery / Active State Classification**: A dual machine-learning system (ACTIVE vs IMAGERY) using a Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA) pipeline to classify user intent.

| Feature | Details |
|---|---|
| **EEG Hardware** | OpenBCI Cyton (16-ch) |
| **Communication** | Lab Streaming Layer (pylsl) |
| **SSVEP Detection** | FFT + Welch PSD + CCA ensemble |
| **Object Classification** | CSP + LDA (mne + scikit-learn) |
| **Output Format** | JSON over LSL string stream |

---

## Project Structure

```text
PythonBCI/
├── main.py                  # Entry point: threading, state machine, LSL I/O
├── protocol.py              # BCICode IntEnum + message schema helpers
├── flicker.py               # SSVEPDetector: FFT + Welch PSD + CCA ensemble
├── prediction.py            # ObjectClassifier: CSP + LDA ML pipeline
├── test_unitypythontest.py  # Standalone LSL round-trip connection test
└── README.md                # This documentation
```

---

## Unity ↔ Python Protocol

### Input: Unity Marker Format

Unity sends CSV log entries over an LSL string stream (type: `"Markers"`). Each sample is a single comma-separated string with exactly **six fields**:

`Time,Experiment,Phase,Event,Detail,Action`

| Field | Role | Example |
|---|---|---|
| **Time** | Timestamp (ignored by Python) | `2026-05-05 15:08:38.284` |
| **Experiment** | Experiment label (ignored) | `BCI` |
| **Phase** | Scene/phase label (ignored) | `Demo3D` |
| **Event** | **Pass-through** — forwarded verbatim to output | `Flicker_Start` |
| **Detail** | **Pass-through** — forwarded verbatim to output | `Object: Door_Single; Hz: 15` |
| **Action** | **State machine driver** — strictly controls Python logic | `Flicker_Start` |

> **Critical Rule**: Python **never** modifies or interprets `Event` or `Detail`. They are stored as-is and echoed back in every outgoing message so Unity can match responses to its own logged events.

### Output: JSON Message Schema

Every packet Python sends to Unity is a single JSON string pushed over the `BCIBackend` LSL outlet (type: `"BCIResult"`). The schema is strictly typed:

```json
{
  "Code":   100,
  "Event":  "Flicker_Start",
  "Detail": "Object: Door_Single; Hz: 15",
  "Remark": {
    "Detected_Frequency": 15.0,
    "Confidence_Score":   0.83,
    "SSVEP_Present":      true,
    "FFT_Score":          0.74,
    "PSD_Score":          0.68,
    "CCA_Score":          0.91
  }
}
```

| Field | Owner | Description |
|---|---|---|
| `Code` | Python | One of the integer `BCICode`s (see below). |
| `Event` | Unity | Echoed back verbatim. |
| `Detail` | Unity | Echoed back verbatim. |
| `Remark` | Python | All ML / signal-processing findings, scores, and debug details. |

### Code Reference Table

| Code | Constant | Trigger Condition |
|------|----------|---------|
| **100** | `FLICKER_DETECTED` | SSVEP present at the target frequency |
| **101** | `FLICKER_NOT_DETECTED` | SSVEP absent |
| **201** | `ACTIVE_OBJ1_TRAIN_COMPLETE` | Enough ACTIVE epochs collected for Door 1 |
| **202** | `ACTIVE_OBJ2_TRAIN_COMPLETE` | Enough ACTIVE epochs collected for Door 2 |
| **203** | `IMAGERY_OBJ1_TRAIN_COMPLETE` | Enough IMAGERY epochs collected for Door 1 |
| **204** | `IMAGERY_OBJ2_TRAIN_COMPLETE` | Enough IMAGERY epochs collected for Door 2 |
| **300** | `ACTIVE_OBJ1_PREDICT` | ACTIVE model predicts Door 1 |
| **301** | `ACTIVE_OBJ2_PREDICT` | ACTIVE model predicts Door 2 |
| **303** | `IMAGERY_OBJ1_PREDICT` | IMAGERY model predicts Door 1 |
| **304** | `IMAGERY_OBJ2_PREDICT` | IMAGERY model predicts Door 2 |

---

## State Machine

The system logic is driven entirely by the `Action` string parsed from the Unity logs. 

### Full Flow Diagram

```text
              ┌──────────┐
              │   IDLE   │◄──────────────────────────────────────────┐
              └────┬─────┘                                           │
                   │ (Unity Action received via LSL)                 │
    ┌──────────────┼────────────────────────────────┐                │
    │              │                                │                │
    ▼              ▼                                ▼                │
SSVEP_TEST   TRAIN_ACTIVE_OBJ1               TRAIN_IMAGERY_OBJ1      │
    │         TRAIN_ACTIVE_OBJ2               TRAIN_IMAGERY_OBJ2     │
    │              │                                │                │
    │   detect()   │  collect n_train_epochs        │                │
    │  → push 100  │  → push 201/202              push 203/204       │
    │    or 101    │                                │                │
    │              │                                │                │
    │  Flicker_End │          Train_End             │                │
    └──────────────┴────────────────────────────────┘                │
                                │                                    │
                           _attempt_training()                       │
                           ACTIVE model fit()                        │
                           IMAGERY model fit()                       │
                                │                                    │
                           ┌────▼───────┐                            │
                           │  PREDICT   │ ── push 300/301, 303/304 ──┘
                           └────────────┘
```

### Action → State Transitions

| Unity `Action` string | Resulting State | Notes |
|---|---|---|
| `Flicker_Start` | `SSVEP_TEST` | Begins analyzing epochs for SSVEP. |
| `Flicker_End` | `IDLE` | Stops analysis. |
| `Training_Active_Door1_Start` | `TRAIN_ACTIVE_OBJ1` | Buffers ACTIVE data for class 0. |
| `Training_Active_Door2_Start` | `TRAIN_ACTIVE_OBJ2` | Buffers ACTIVE data for class 1. |
| `Training_Imagery_Door1_Start` | `TRAIN_IMAGERY_OBJ1` | Buffers IMAGERY data for class 0. |
| `Training_Imagery_Door2_Start` | `TRAIN_IMAGERY_OBJ2` | Buffers IMAGERY data for class 1. |
| `Train_End` | `IDLE` | Automatically triggers `_attempt_training()` first. |
| `Predict_Start` | `PREDICT` | Begins evaluating epochs against trained models. |
| `Predict_End` | `IDLE` | Stops prediction. |

---

## Signal Processing Pipelines

### Global EEG Preprocessing

Applied to **every** incoming EEG epoch across **all** states before any method-specific processing. This ensures the baseline signal is completely uniform before feature extraction.

```text
Raw 16-ch epoch (n_channels × n_samples)
    │
    ▼ 50 Hz Notch filter  (IIR, Q=30)
    │
    ▼ 8–30 Hz Bandpass filter  (4th-order Butterworth, SOS zero-phase)
    │
    ▼ Per-channel Z-score normalisation
    │
    └─► Pre-processed epoch → SSVEPDetector or ObjectClassifier
```

### SSVEP Detection (`flicker.py`)

`SSVEPDetector` performs its own secondary bandpass (1–40 Hz, configurable) on top of the global preprocessing, then runs three independent sub-methods. Their scores are combined using a configurable weighted ensemble:

| Sub-method | Description | Default Weight |
|---|---|---|
| **FFT** | Mean spectral amplitude at f, 2f, 3f normalised by total band power | 1.0 |
| **Welch PSD** | Log-compressed SNR at target frequency vs. neighbouring bins | 1.0 |
| **CCA** | Maximum canonical correlation against synthetic sine/cosine references | 1.5 |

If `ensemble_score >= detection_threshold` (default `0.55`), the result is `FLICKER_DETECTED (100)`. Otherwise, `FLICKER_NOT_DETECTED (101)`.

### Object Classification (`prediction.py`)

`ObjectClassifier` implements a robust prediction pipeline using `mne` and `scikit-learn`. There are two entirely independent models: `active_model` and `imagery_model`.

**Training** (triggered by `Train_End`):
```text
Training epochs (n_trials × n_channels × n_samples)
    │
    ▼ CSP.fit_transform(X, y)
      → 4 spatial components, extracts log-variance features
    │
    ▼ LDA.fit(X_features, y)
    │
    └─► Model is ready
```

**Prediction** (in `PREDICT` state):
```text
Single epoch (n_channels × n_samples)
    │
    ▼ CSP.transform(X)
    │
    ▼ LDA.predict(X_features) + LDA.predict_proba(X_features)
    │
    └─► (predicted_label, confidence)
```

During the `PREDICT` state, **both** models evaluate the incoming epoch simultaneously and emit their own respective JSON messages to Unity.

---

## Threading Model

The backend is built for zero-blocking real-time processing using three daemon threads.

```text
┌─────────────────────────────────────────────────────────────────┐
│  Thread 1  (Thread-EEG)                                         │
│  pull_sample() → deque(maxlen=epoch_samples)                    │
│  Sets _eeg_ready Event when buffer is full                      │
├─────────────────────────────────────────────────────────────────┤
│  Thread 2  (Thread-Marker)                                      │
│  pull_sample(timeout=0.0)  ← NON-BLOCKING                       │
│  Parses CSV → stores event/detail, puts action → queue.Queue   │
├─────────────────────────────────────────────────────────────────┤
│  Thread 3  (Thread-Logic)                                       │
│  Drains marker queue → calls _handle_marker() → state change   │
│  Waits on _eeg_ready → snapshots epoch → _process_epoch()      │
│  Builds JSON → pushes via LSL outlet                           │
└─────────────────────────────────────────────────────────────────┘
```

> **Why non-blocking marker pull?**
> Thread 2 uses `pull_sample(timeout=0.0)` so it **never blocks the GIL** waiting for network I/O. A `time.sleep(0.001)` yields the GIL at a stable ~1 kHz polling cadence.

---

## Installation & Setup

### Requirements

| Package | Version | Purpose |
|---|---|---|
| **Python** | ≥ 3.9 | Core runtime |
| **numpy** | ≥ 1.24 | Matrix operations |
| **scipy** | ≥ 1.11 | Filtering (Notch, Butterworth, Welch PSD) |
| **pylsl** | ≥ 1.16 | Lab Streaming Layer networking |
| **mne** | ≥ 1.0 | Common Spatial Patterns (CSP) |
| **scikit-learn** | ≥ 1.2 | Linear Discriminant Analysis (LDA) |

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

> **OpenBCI Hardware**: Ensure the OpenBCI GUI (or BrainFlow) is running and broadcasting a `type="EEG"` LSL stream before launching `main.py`. If no stream is detected, the backend gracefully falls back to generating simulated Gaussian noise for testing.

---

## Operation & CLI Arguments

### Command Line Execution

```bash
# Default settings: 10 Hz target, 250 Hz sampling, 4-second epochs
python main.py

# Custom: 15 Hz flicker, 256 Hz hardware, 3-second analysis window
python main.py --target-freq 15 --sfreq 256 --epoch-duration 3

# Collect 20 training epochs per object
python main.py --n-train-epochs 20

# Verbose debug output (shows per-method sub-scores)
python main.py --log-level DEBUG
```

### Configuration Reference

| Argument | Default | Description |
|---|---|---|
| `--target-freq` | `10.0` | SSVEP stimulus frequency to detect (Hz). |
| `--sfreq` | `250.0` | EEG hardware sampling rate (Hz). |
| `--epoch-duration` | `4.0` | Length of the EEG analysis window (s). |
| `--n-train-epochs` | `10` | Epochs required per object to complete training. |
| `--detection-threshold` | `0.55` | Minimum weighted ensemble score for SSVEP detection. |
| `--resolve-timeout` | `10.0` | Seconds to search for each LSL stream before giving up. |
| `--log-level` | `INFO` | Logging verbosity (`DEBUG` / `INFO` / `WARNING` / `ERROR`). |
| `--test-mode` | `false` | Emit dummy heartbeat messages every 2s for connection testing. |

---

## Unity Integration Guide

### LSL Streams Required

| Direction | Stream Name | Stream Type | Format |
|---|---|---|---|
| Unity → Python | *(any)* | `"Markers"` | 1-channel string CSV |
| Python → Unity | `"BCIBackend"` | `"BCIResult"` | 1-channel JSON string |

### Receiving BCI Results in Unity

Your `LSLCommunicationManager` should resolve the `BCIBackend` outlet and parse each JSON message into a struct.

**Data Classes:**
```csharp
public class BCIMessage {
    public int    Code;
    public string Event;
    public string Detail;
    public BCIRemark Remark;
}

public class BCIRemark {
    public float  Detected_Frequency;
    public float  Confidence_Score;
    public bool   SSVEP_Present;
    public string Message;
    public int    Epochs_Collected;
    public int    Target_Epochs;
    public string Object;
    public string Model;
    public string Prediction;
    public float  Confidence;
}
```

**Subscription Pattern:**
```csharp
private void OnEnable()
{
    LSLCommunicationManager.Instance.OnFlickerStateChanged  += HandleFlicker;
    LSLCommunicationManager.Instance.OnPredictionResult     += HandlePrediction;
}

private void HandleFlicker(bool detected, BCIMessage msg)
{
    // Ensure the message belongs to this specific object's event
    if (msg.Event == lastEvent && msg.Detail == lastDetail)
    {
        if (detected) ExecuteAction();
    }
}

private void HandlePrediction(int code)
{
    // 300 = ACTIVE Door 1 | 303 = IMAGERY Door 1
    if (code == 300 || code == 303) OpenDoor1();
    else OpenDoor2();
}
```

---

## Testing the Connection

`test_unitypythontest.py` simulates the full Unity → Python → Unity LSL round-trip without needing the Unity Editor open.

**Usage (Requires two terminal windows):**

```bash
# Terminal 1: Start the BCI backend
python main.py --log-level DEBUG

# Terminal 2: Run the mock Unity sender/receiver
python test_unitypythontest.py
```

The test script:
1. Creates an LSL outlet (type `"Markers"`) and sends mock Unity CSV log entries (including training loops and predictions).
2. Listens on the `"BCIResult"` outlet and prints every JSON message Python sends back.

---

## Troubleshooting

### "No EEG LSL stream found"
- Confirm the OpenBCI GUI Networking panel has **LSL → Start** active.
- List all visible streams via python: `python -c "from pylsl import resolve_streams; print(resolve_streams())"`
- Check firewall / VPN — LSL uses multicast UDP on port `16571`.
- *Note: If hardware is unavailable, Python automatically generates simulated Gaussian noise so you can test the pipeline logic regardless.*

### "No Marker LSL stream found"
- Confirm the Unity project has `LSL_Logger` active and `LSLCommunicationManager` set to stream type `"Markers"`.
- Ensure Unity and Python are on the same local network subnet or loopback (`127.0.0.1`).

### SSVEP is rarely detected or throws false positives
- **Rarely detected:** Lower `--detection-threshold` (e.g. `0.45`).
- **False positives:** Raise `--detection-threshold` (e.g. `0.65`).
- Enable `--log-level DEBUG` to view per-method sub-scores (`FFT_Score`, `PSD_Score`, `CCA_Score`) in the terminal.
- **CCA score always 0:** Check `occipital_channels` indices match your electrode layout.

### Training fails silently
- Verify `Train_End` is sent **after** both Door 1 and Door 2 epochs have been collected.
- Check that at least `n_train_epochs` epochs arrived. If fewer epochs are available, Python logs a warning and skips `model.fit()`.
- CSP requires a minimum of 2 classes × 2 epochs to compute covariance matrices.

### High CPU usage
- Reduce `sfreq` or increase `epoch_duration` to lower the analysis rate.
- Thread 2 polls at ~1 kHz. You can adjust `time.sleep(0.001)` in `_marker_ingestion_loop` to `0.005` to trade a tiny amount of latency for CPU headroom.
