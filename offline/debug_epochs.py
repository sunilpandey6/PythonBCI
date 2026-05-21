import pyxdf
import os
import numpy as np

data_dir = "/Users/sunilpandey/Documents/bci/PythonBCI/data"
for fname in ["s4.xdf", "s5.xdf"]:
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        continue
    print(f"\n=================== Debugging {fname} ===================")
    streams, header = pyxdf.load_xdf(fpath)
    
    eeg_ts = None
    marker_ts = []
    marker_vals = []
    
    for s in streams:
        stype = s["info"]["type"][0]
        if stype == "EEG":
            eeg_ts = s["time_stamps"]
        elif stype == "Markers":
            marker_ts = s["time_stamps"]
            marker_vals = [m[0] for m in s["time_series"]]
            
    if eeg_ts is None:
        print("No EEG stream found!")
        continue
        
    print(f"EEG timestamps: count={len(eeg_ts)}, min={np.min(eeg_ts):.3f}, max={np.max(eeg_ts):.3f}, diff={np.max(eeg_ts)-np.min(eeg_ts):.3f}s")
    if len(marker_ts) > 0:
        print(f"Marker timestamps: count={len(marker_ts)}, min={np.min(marker_ts):.3f}, max={np.max(marker_ts):.3f}, diff={np.max(marker_ts)-np.min(marker_ts):.3f}s")
        
        # Let's see some markers and their timestamps
        print("First 5 markers:")
        for i in range(min(5, len(marker_ts))):
            print(f"  {marker_ts[i]:.3f}: {marker_vals[i]}")
            
        # Let's check alignment
        eeg_min, eeg_max = np.min(eeg_ts), np.max(eeg_ts)
        markers_in_eeg_range = sum(1 for t in marker_ts if eeg_min <= t <= eeg_max)
        print(f"Number of markers within EEG timestamp range: {markers_in_eeg_range} / {len(marker_ts)}")
        
        # Let's extract the training intervals using the markers
        def extract_intervals(m_ts, m_vals, start_label, end_label):
            intervals = []
            current_start = None
            for ts, m_str in zip(m_ts, m_vals):
                if start_label in m_str:
                    current_start = ts
                elif end_label in m_str and current_start is not None:
                    intervals.append((current_start, ts))
                    current_start = None
            return intervals
            
        c0 = extract_intervals(marker_ts, marker_vals, "Training_Active_Door1_Start", "Training_Active_Door1_End")
        print(f"Class 0 intervals count: {len(c0)}")
        for idx, (t_start, t_end) in enumerate(c0):
            print(f"  Interval {idx}: [{t_start:.3f}, {t_end:.3f}] (duration: {t_end-t_start:.3f}s)")
            # Let's count EEG samples in this interval
            eeg_samples_in_interval = sum(1 for t in eeg_ts if t_start <= t <= t_end)
            print(f"    EEG samples in interval: {eeg_samples_in_interval}")
            
            # Check with shift
            latency_shift_s = 0.75
            epoch_duration = 1.0
            t = t_start + latency_shift_s
            epochs_possible = 0
            while t + epoch_duration <= t_end:
                epochs_possible += 1
                t += 0.2
            print(f"    Possible epochs with latency shift {latency_shift_s}s and duration {epoch_duration}s: {epochs_possible}")
