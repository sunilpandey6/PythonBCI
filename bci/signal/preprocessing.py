from __future__ import annotations
import numpy as np
from scipy import signal

def preprocess_global(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """Notch (50 Hz) -> Bandpass (1.0-90.0 Hz)."""
    nyq = sfreq / 2.0
    b_n, a_n = signal.iirnotch(50.0 / nyq, 30.0)
    epoch_filt = signal.filtfilt(b_n, a_n, epoch, axis=-1)
    
    sos = signal.butter(4, [1.0, min(90.0, nyq - 0.1)], btype="bandpass", fs=sfreq, output="sos")
    epoch_filt = signal.sosfiltfilt(sos, epoch_filt, axis=-1)
    
    return epoch_filt
