# BCI Offline Machine Learning Validation Pipeline

This directory contains the tools and scripts to record and validate the BCI machine learning models offline using previously recorded session data (`.xdf` files). It replicates the data-ingestion, preprocessing, sliding-window epoching, and model-training logic of the main BCI application in a controlled simulation environment.

---

## Directory Structure

- **[run_offline_validation.py](run_offline_validation.py)**: An automated Python script that launches the XDF replayer in the background, records the simulated LSL stream, processes epochs, trains all three classifier models, prints metrics, and saves confusion matrices.
- **[offline.ipynb](offline.ipynb)**: An interactive Jupyter Notebook implementation of the validation pipeline, allowing developers to run step-by-step, inspect variables, and visualize intermediate states.
- **[debug_epochs.py](debug_epochs.py)**: A debugging utility to check extracted epoch dimensions and print marker events.
- **[create_notebook.py](../.gemini/antigravity-ide/brain/838f8343-5504-4f76-b76c-6a69f14ba42f/scratch/create_notebook.py)**: A helper script to regenerate the `offline.ipynb` file from JSON definitions to ensure parity with the python script.

---

## Machine Learning Model Variants

The pipeline trains and evaluates both Active and Imagery classification models using a stratified **80-20 train-test split**.

### 1. Active Classification Models
Three variants are trained to classify Door 1 Active, Door 2 Active, and Door 1 Active Flicker:
- **Variant 1 (Globally Normalized, All Channels)**:
  - Continuous EEG data is standardized globally (channel-by-channel subtraction of the mean and division by the standard deviation of the entire session) before extracting epochs.
  - The ML training pipeline uses a Scikit-Learn `Pipeline` composed of an MNE `Vectorizer`, a `StandardScaler`, and a linear `SVC`.
- **Variant 2 (Raw / Non-Normalized, All Channels)**:
  - Continuous EEG data is kept raw (no session-wide normalization).
  - The ML training pipeline uses `Vectorizer` and a linear `SVC` directly on raw continuous EEG.
- **Variant 3 (Raw / Selected Channels, Specific Subsets)**:
  - Continuous EEG data is kept raw.
  - The dataset is sliced to include only a specific user-defined subset of channels (e.g., specific channels like `[4, 5, 6, 7, 14, 15]`) before feeding them to a pipeline of `Vectorizer` and a linear `SVC`.

### 2. Imagery Classification Models (No Normalization)
Two variants are trained to classify Door 1 Imagery, Door 2 Imagery, and Door 1 Flicker (Class 2):
- **Imagery V1 (CSP + LDA)**:
  - Spatial patterns are extracted using Common Spatial Patterns (CSP) filtering (using MNE `CSP` with 4 components), followed by a `LinearDiscriminantAnalysis` classifier.
- **Imagery V2 (Vectorizer + SVM)**:
  - Replicates the Vectorizer + SVM structure of the Active models (without normalization).

### Classifier Classes

#### Active Classifier Classes
- **Class 0**: Door 1 Active (`Training_Active_Door1_Start` / `End` or `TAD1S` / `TAD1E`)
  - *Note: SSVEP validation check on Class 0 training epochs is disabled to keep all scheduled epochs inside the target training class.*
- **Class 1**: Door 2 Active (`Active_Training_Door2_Start` / `End` or `TAD2S` / `TAD2E`)
- **Class 2**: Door 1 Active Flicker (`Training_Active_Door1_Flicker_Start` / `End` or `TF1S` / `TF1E`)

#### Imagery Classifier Classes
- **Class 0**: Door 1 Imagery (`Training_Imagery_Door1_Start` / `End` or `TID1S` / `TID1E`)
- **Class 1**: Door 2 Imagery (`Image_Training_Door2_Start`/`TID2S`/`Training_Imagery_Door2_Start` to `Image_Training_Door2_End`/`TID2E`/`Training_Imagery_Door2_End`)
- **Class 2**: Door 1 Flicker (`Training_Active_Door1_Flicker_Start` / `End` or `TF1S` / `TF1E` - flicker trials)

---

## Setup & Ingestion Pipeline

### 1. The Replay Server (`testxdfmain.py`)
To feed data into the offline validation pipeline, you must replay a recorded session file (e.g. `s4.xdf`) over LSL.
Run the replayer from the repository root:
```bash
python3 PythonBCI/check/testxdfmain.py PythonBCI/data/s4.xdf --speed 10.0
```
- The `--speed` flag controls the playback rate (e.g. `10.0` plays the session back 10 times faster than real-time).
- The offline pipeline is **speed-agnostic**—it automatically scales timestamps using the ratio of actual samples collected to nominal sampling rates, mapping replayed events back to the 1x nominal speed domain.

---

## How to Run Validation

### Option A: Using the Automated Script (`run_offline_validation.py`)
You do not need to manually start the replayer for this script. It automatically launches the replayer process in the background at the specified speed:

1. Open `run_offline_validation.py` and modify the target file or selected channels in the `__main__` section:
   ```python
   # Run validation using alternate channels:
   run_validation("s4.xdf", speed=20.0, output_prefix="s4", selected_channels=[4, 5, 6, 7, 14, 15])
   ```
2. Execute the script:
   ```bash
   python3 PythonBCI/offline/run_offline_validation.py
   ```
3. View the printed classification reports for both Active and Imagery variants and check the saved plot:
   - Plot location: `PythonBCI/offline/confusion_matrices_<prefix>.png`

### Option B: Using the Jupyter Notebook (`offline.ipynb`)
1. In a terminal, launch the LSL replayer:
   ```bash
   python3 PythonBCI/check/testxdfmain.py PythonBCI/data/s4.xdf --speed 10.0
   ```
2. Open `offline.ipynb` in your Jupyter interface or VS Code.
3. Run the cells in order:
   - **Step 1**: Configures imports and paths (including MNE CSP and LDA).
   - **Step 2 & 3**: Instantiates the `LslOfflineRecorder` and starts listening. The recorder automatically stops once the session end marker (`MLtest_End` or `Train_End`) is captured.
   - **Step 4**: Scales timestamps, calculates global normalization parameters, extracts sliding epochs (for both Active and Imagery classes), and applies the SSVEP filter bypassed rules.
   - **Step 5**: Splits the datasets 80-20, fits all three Active SVM variants and both Imagery variants, and displays classification reports along with a 2x3 confusion matrix grid (Row 1: Active V1/V2/V3; Row 2: Imagery V1/V2).

---

## Understanding the Results

### Epoch Counts
During interval extraction, you may notice different epoch counts for each class:
- **Class 1 (Door 2)** has more epochs because its training intervals are longer (e.g., 4 seconds each). With a sliding window duration of 1.0s and step size of 0.2s, each interval yields `(4.0 - 1.0)/0.2 + 1 = 16` epochs.
- **Class 0 (Door 1)** has fewer epochs if its training intervals are shorter. For instance, intervals of ~2.1 seconds yield `(2.1 - 1.0)/0.2 + 1 = 6` epochs per interval.
- The pipeline correctly handles these differences by using `class_weight='balanced'` in the SVM model to account for class imbalances.
