import sys
import os
import time
import subprocess
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to save plots without showing window
import matplotlib.pyplot as plt
import seaborn as sns
from pylsl import StreamInlet, resolve_byprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mne.decoding import Vectorizer

# Add the parent directory of PythonBCI to the path to import components from bci
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bci.signal.preprocessing import preprocess_global
from bci.signal.ssvep import SSVEPDetector

class LslOfflineRecorder:
    def __init__(self, target_freq=15.0, detection_threshold=0.4, epoch_duration=1.0, step_size=0.2, latency_shift_s=0.75):
        self.target_freq = target_freq
        self.detection_threshold = detection_threshold
        self.epoch_duration = epoch_duration
        self.step_size = step_size
        self.latency_shift_s = latency_shift_s
        
        self.eeg_data = []
        self.eeg_timestamps = []
        self.marker_events = []
        
        self.shutdown_event = threading.Event()
        self.eeg_thread = None
        self.marker_thread = None
        
        self.sfreq = None
        self.n_channels = None
        
    def start(self):
        print("Resolving LSL streams...")
        eeg_streams = resolve_byprop("type", "EEG", timeout=10.0)
        marker_streams = resolve_byprop("type", "Markers", timeout=10.0)
        
        if not eeg_streams:
            raise RuntimeError("No EEG stream found! Please make sure testxdfmain.py is running.")
        if not marker_streams:
            raise RuntimeError("No Marker stream found! Please make sure testxdfmain.py is running.")
            
        eeg_inlet = StreamInlet(eeg_streams[0])
        marker_inlet = StreamInlet(marker_streams[0])
        
        info = eeg_inlet.info()
        self.sfreq = info.nominal_srate()
        self.n_channels = info.channel_count()
        print(f"Connected to EEG stream '{info.name()}' with {self.n_channels} channels @ {self.sfreq} Hz.")
        print(f"Connected to Marker stream '{marker_inlet.info().name()}'")
        
        self.shutdown_event.clear()
        self.eeg_data.clear()
        self.eeg_timestamps.clear()
        self.marker_events.clear()
        
        self.eeg_thread = threading.Thread(target=self._ingest_eeg, args=(eeg_inlet,), daemon=True)
        self.marker_thread = threading.Thread(target=self._ingest_markers, args=(marker_inlet,), daemon=True)
        
        self.eeg_thread.start()
        self.marker_thread.start()
        print("Ingestion threads started. Collecting data...")
        
    def stop(self):
        self.shutdown_event.set()
        if self.eeg_thread:
            self.eeg_thread.join(timeout=2.0)
        if self.marker_thread:
            self.marker_thread.join(timeout=2.0)
        print("\nIngestion threads stopped.")
        
    def _ingest_eeg(self, inlet):
        while not self.shutdown_event.is_set():
            sample, timestamp = inlet.pull_sample(timeout=0.1)
            if sample is not None:
                self.eeg_data.append(sample)
                self.eeg_timestamps.append(timestamp)
                
    def _ingest_markers(self, inlet):
        while not self.shutdown_event.is_set():
            sample, timestamp = inlet.pull_sample(timeout=0.1)
            if sample is not None:
                marker_str = sample[0]
                self.marker_events.append((timestamp, marker_str))
                if "MLtest_End" in marker_str or "Train_End" in marker_str:
                    print(f"\nCaptured session end marker: '{marker_str}' at timestamp {timestamp:.2f}")

def run_validation(xdf_file, speed=10.0, output_prefix="s4", selected_channels=None):
    if selected_channels is None:
        selected_channels = [4,5,6,7,14,15]  # Default to first 8 channels
    # Start replayer in background
    replayer_path = os.path.join(parent_dir, "check", "testxdfmain.py")
    xdf_path = os.path.join(parent_dir, "data", xdf_file)
    
    print(f"\n=======================================================")
    print(f"Starting Offline ML pipeline test for: {xdf_file}")
    print(f"=======================================================")
    
    print(f"Launching replayer for {xdf_file} at speed {speed}...")
    replayer_process = subprocess.Popen(
        [sys.executable, replayer_path, xdf_path, "--speed", str(speed)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    recorder = LslOfflineRecorder(target_freq=15.0, detection_threshold=0.4)
    
    try:
        # Give replayer process 1-2 seconds to start and register outlets
        time.sleep(2.0)
        recorder.start()
        
        # Monitor progress until session end marker is received or replayer process terminates
        while True:
            # Check if background replayer failed/exited
            ret_code = replayer_process.poll()
            if ret_code is not None:
                print(f"\nReplayer process exited with code {ret_code}.")
                # Wait 2 seconds for any trailing samples
                time.sleep(2.0)
                break
                
            has_end = any("MLtest_End" in m or "Train_End" in m for _, m in recorder.marker_events)
            print(f"\rCollected: {len(recorder.eeg_data)} EEG samples, {len(recorder.marker_events)} markers... ", end="", flush=True)
            
            if has_end:
                print("\nSession end marker detected. Stopping recorder...")
                time.sleep(2.0)
                break
                
            time.sleep(0.5)
            
    except Exception as e:
        print(f"\nError during data capture: {e}")
        replayer_process.terminate()
        raise e
    finally:
        recorder.stop()
        # Clean up replayer process
        if replayer_process.poll() is None:
            replayer_process.terminate()
            replayer_process.wait()
            
    if len(recorder.eeg_data) == 0:
        print("Error: No EEG data was collected!")
        return
        
    eeg_data_arr = np.array(recorder.eeg_data)
    eeg_timestamps_arr = np.array(recorder.eeg_timestamps)
    marker_events = recorder.marker_events
    
    print(f"\nEEG Data shape: {eeg_data_arr.shape}")
    print(f"Markers count: {len(marker_events)}")
    
    # Scale timestamps relative to first sample and adjust for LSL replay speed
    sfreq = recorder.sfreq if recorder.sfreq else 125.0
    if len(eeg_timestamps_arr) > 1:
        t0 = eeg_timestamps_arr[0]
        tN = eeg_timestamps_arr[-1]
        duration_lsl = tN - t0
        if duration_lsl > 0:
            duration_exp = len(eeg_timestamps_arr) / sfreq
            speed_factor = duration_exp / duration_lsl
            print(f"Detected LSL replay speed: {1.0/speed_factor:.2f}x. Scaling timestamps to 1x speed.")
            eeg_timestamps_arr = (eeg_timestamps_arr - t0) * speed_factor
            marker_events = [((ts - t0) * speed_factor, m_str) for ts, m_str in marker_events]
            
    # 1. Global continuous EEG standardization
    global_mean = np.mean(eeg_data_arr, axis=0)
    global_std = np.std(eeg_data_arr, axis=0)
    global_std[global_std == 0.0] = 1.0
    
    eeg_data_norm = (eeg_data_arr - global_mean) / global_std
    
    # 2. Extract training intervals from markers
    def extract_intervals(markers, start_labels, end_labels):
        intervals = []
        current_start = None
        for ts, m_str in markers:
            if any(label in m_str for label in start_labels):
                current_start = ts
            elif any(label in m_str for label in end_labels) and current_start is not None:
                intervals.append((current_start, ts))
                current_start = None

        return intervals
        
    class0_intervals = extract_intervals(marker_events, ["Training_Active_Door1_Start","TAD1S"], ["Training_Active_Door1_End","TAD1E"])
    class1_intervals = extract_intervals(marker_events, ["Active_Training_Door2_Start","TAD2S"], ["Active_Training_Door2_End","TAD2E"])
    class2_intervals = extract_intervals(marker_events, ["Training_Active_Door1_Flicker_Start","TF1S"], ["Training_Active_Door1_Flicker_End","TF1E"])
    
    print("\n--- Interval extraction summary ---")
    print(f"Class 0 (Door 1 Active) intervals: {len(class0_intervals)}")
    print(f"Class 1 (Door 2 Active) intervals: {len(class1_intervals)}")
    print(f"Class 2 (Door 1 Active Flicker) intervals: {len(class2_intervals)}")
    
    # 3. Sliding window epoch extractor
    def extract_epochs(eeg, timestamps, intervals, sfreq, epoch_duration=1.0, step_size=0.2, latency_shift_s=0.75):
        epochs = []
        n_samples = int(epoch_duration * sfreq)
        for t_start, t_end in intervals:
            t = t_start + latency_shift_s
            while t + epoch_duration <= t_end:
                idx = np.searchsorted(timestamps, t)
                if idx + n_samples <= len(timestamps):
                    epoch = eeg[idx : idx + n_samples, :].T
                    epochs.append(epoch.copy())
                t += step_size
        return epochs
        
    sfreq = recorder.sfreq if recorder.sfreq else 125.0
    
    raw_epochs_c0 = extract_epochs(eeg_data_arr, eeg_timestamps_arr, class0_intervals, sfreq)
    norm_epochs_c0 = extract_epochs(eeg_data_norm, eeg_timestamps_arr, class0_intervals, sfreq)
    
    raw_epochs_c1 = extract_epochs(eeg_data_arr, eeg_timestamps_arr, class1_intervals, sfreq)
    norm_epochs_c1 = extract_epochs(eeg_data_norm, eeg_timestamps_arr, class1_intervals, sfreq)
    
    raw_epochs_c2 = extract_epochs(eeg_data_arr, eeg_timestamps_arr, class2_intervals, sfreq)
    norm_epochs_c2 = extract_epochs(eeg_data_norm, eeg_timestamps_arr, class2_intervals, sfreq)
    
    print(f"\nExtracted initial epochs:")
    print(f"  Class 0 (Door 1 Active): {len(raw_epochs_c0)} epochs")
    print(f"  Class 1 (Door 2 Active): {len(raw_epochs_c1)} epochs")
    print(f"  Class 2 (Door 1 Active Flicker - explicit): {len(raw_epochs_c2)} epochs")
    
    # 4. Process and package preprocessed epochs (SSVEP check on Class 0/TRAIN_ACTIVE_OBJ1 is disabled)
    X_raw_final = []
    X_norm_final = []
    y_final = []
    reassigned_count = 0
    
    # Process Class 0
    for raw_ep, norm_ep in zip(raw_epochs_c0, norm_epochs_c0):
        prep_raw = preprocess_global(raw_ep, sfreq, current_state="TRAIN_ACTIVE_OBJ1")
        prep_norm = preprocess_global(norm_ep, sfreq, current_state="TRAIN_ACTIVE_OBJ1")
        
        if prep_raw is None or prep_norm is None:
            continue
            
        X_raw_final.append(prep_raw)
        X_norm_final.append(prep_norm)
        y_final.append(0)
            
    # Process Class 1
    for raw_ep, norm_ep in zip(raw_epochs_c1, norm_epochs_c1):
        prep_raw = preprocess_global(raw_ep, sfreq, current_state="TRAIN_ACTIVE_OBJ2")
        prep_norm = preprocess_global(norm_ep, sfreq, current_state="TRAIN_ACTIVE_OBJ2")
        if prep_raw is not None and prep_norm is not None:
            X_raw_final.append(prep_raw)
            X_norm_final.append(prep_norm)
            y_final.append(1)
            
    # Process Class 2
    for raw_ep, norm_ep in zip(raw_epochs_c2, norm_epochs_c2):
        prep_raw = preprocess_global(raw_ep, sfreq, current_state="TRAIN_ACTIVE_OBJ1")
        prep_norm = preprocess_global(norm_ep, sfreq, current_state="TRAIN_ACTIVE_OBJ1")
        if prep_raw is not None and prep_norm is not None:
            X_raw_final.append(prep_raw)
            X_norm_final.append(prep_norm)
            y_final.append(2)
            
    y_final = np.array(y_final)
    print(f"\n--- SSVEP Reassignment Results ---")
    print(f"Reassigned {reassigned_count} epochs from Class 0 to Class 2.")
    print(f"Final Class Counts:")
    print(f"  Class 0 (Door 1 Active): {np.sum(y_final == 0)} epochs")
    print(f"  Class 1 (Door 2 Active): {np.sum(y_final == 1)} epochs")
    print(f"  Class 2 (Door 1 Active Flicker): {np.sum(y_final == 2)} epochs")
    
    if len(y_final) < 5:
        print("Error: Too few epochs collected for machine learning split and training.")
        return
        
    # Stack epochs
    X_raw_final = np.stack(X_raw_final, axis=0)
    X_norm_final = np.stack(X_norm_final, axis=0)
    
    # 5. Split, Train, and Evaluate Models
    # Train-test split (80-20 rule, stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw_final, y_final, test_size=0.2, random_state=42, stratify=y_final
    )
    X_train_norm, X_test_norm, _, _ = train_test_split(
        X_norm_final, y_final, test_size=0.2, random_state=42, stratify=y_final
    )
    
    # Slice the raw data for selected channels (Variant 3)
    X_train_chan = X_train_raw[:, selected_channels, :]
    X_test_chan = X_test_raw[:, selected_channels, :]
    
    # Define models
    model_norm = make_pipeline(
        Vectorizer(),
        StandardScaler(),  # Features normalized per channel-time point
        SVC(kernel='linear', probability=True, class_weight='balanced')
    )
    
    model_raw = make_pipeline(
        Vectorizer(),
        SVC(kernel='linear', probability=True, class_weight='balanced')
    )
    
    model_chan = make_pipeline(
        Vectorizer(),
        SVC(kernel='linear', probability=True, class_weight='balanced')
    )
    
    model_norm.fit(X_train_norm, y_train)
    model_raw.fit(X_train_raw, y_train)
    model_chan.fit(X_train_chan, y_train)
    
    y_pred_norm = model_norm.predict(X_test_norm)
    y_pred_raw = model_raw.predict(X_test_raw)
    y_pred_chan = model_chan.predict(X_test_chan)
    
    class_names = ["Door1 Active", "Door2 Active", "Door1 Active Flicker"]
    present_classes = np.unique(y_test)
    labels_present = [class_names[c] for c in present_classes]
    
    # Print results
    print("===========================================================")
    print("      VARIANT 1: EEG DATA GLOBALLY NORMALIZED")
    print("===========================================================")
    print(classification_report(y_test, y_pred_norm, target_names=labels_present, labels=present_classes, zero_division=0))
    
    print("===========================================================")
    print("      VARIANT 2: EEG DATA RAW (NON-NORMALIZED)")
    print("===========================================================")
    print(classification_report(y_test, y_pred_raw, target_names=labels_present, labels=present_classes, zero_division=0))
    
    print("===========================================================")
    print(f"      VARIANT 3: EEG DATA RAW (SELECTED CHANNELS: {selected_channels})")
    print("===========================================================")
    print(classification_report(y_test, y_pred_chan, target_names=labels_present, labels=present_classes, zero_division=0))
    
    # Plot and save confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    cm_norm = confusion_matrix(y_test, y_pred_norm, labels=present_classes)
    sns.heatmap(cm_norm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=labels_present, yticklabels=labels_present)
    axes[0].set_title("Variant 1 (Normalized) Confusion Matrix")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    
    cm_raw = confusion_matrix(y_test, y_pred_raw, labels=present_classes)
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=labels_present, yticklabels=labels_present)
    axes[1].set_title("Variant 2 (Raw/Non-normalized) Confusion Matrix")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    
    cm_chan = confusion_matrix(y_test, y_pred_chan, labels=present_classes)
    sns.heatmap(cm_chan, annot=True, fmt='d', cmap='Greens', ax=axes[2],
                xticklabels=labels_present, yticklabels=labels_present)
    axes[2].set_title(f"Variant 3 (Raw, Chans: {selected_channels}) Matrix")
    axes[2].set_xlabel("Predicted label")
    axes[2].set_ylabel("True label")
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), f"confusion_matrices_{output_prefix}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved confusion matrix plot to: {plot_path}")

if __name__ == "__main__":
    # Test s4.xdf with subset of channels (e.g. alternate channels [0, 2, 4, 6, 8, 10, 12, 14])
    run_validation("s4.xdf", speed=20.0, output_prefix="s4", selected_channels=[4, 5, 6, 7, 14, 15])
    # Test s5.xdf
    # run_validation("s5.xdf", speed=20.0, output_prefix="s5")
    # run_validation("s5.xdf", speed=20.0, output_prefix="s5", selected_channels=[0, 2, 4, 6, 8, 10, 12, 14])

