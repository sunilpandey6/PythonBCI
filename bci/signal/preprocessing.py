from __future__ import annotations
import numpy as np
from scipy import signal

STATE_BANDPASS_MAP = {
    "SSVEP_TEST": (3.0, 60.0),
    "TRAIN_ACTIVE_OBJ1": (3.0, 60.0),
    "TRAIN_ACTIVE_OBJ2": (3.0, 60.0),
    "PREDICT_ACTIVE": (3.0, 60.0),
    "TRAIN_IMAGERY_OBJ1": (3.0, 90.0),
    "TRAIN_IMAGERY_OBJ2": (3.0, 90.0),
    "PREDICT_IMAGERY": (3.0, 90.0),
    "PREDICT_MIXED": (3.0, 90.0),
}

def preprocess_global(
    epoch: np.ndarray,
    sfreq: float,
    is_eye_closed: bool = False,
    current_state: str = "IDLE"
) -> np.ndarray | None:
    if is_eye_closed:
        return None
    nyq = sfreq / 2.0
    
    epoch_filt = epoch.copy()
    if 50.0 < nyq:
        b50, a50 = signal.iirnotch(50.0 / nyq, 30.0)
        epoch_filt = signal.filtfilt(b50, a50, epoch_filt, axis=-1)
    
    low_cut, high_cut = STATE_BANDPASS_MAP.get(current_state, (1.0, 90.0))
    low = max(0.1, low_cut)
    high = min(high_cut, nyq - 0.1)
    if low >= high:
        low = 1.0
        high = min(40.0, nyq - 0.1)
        
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    epoch_filt = signal.sosfiltfilt(sos, epoch_filt, axis=-1)
    
    return epoch_filt

pre_process_epoch = preprocess_global

