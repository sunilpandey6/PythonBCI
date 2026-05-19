from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

DEFAULT_SFREQ: float = 250.0

@dataclass
class BCIConfig:
    """Single source of truth for all BCIBackend configuration parameters."""
    target_freq: float = 15.0
    sfreq: float = DEFAULT_SFREQ
    epoch_duration: float = 1.0
    n_train_epochs: int = 30
    detection_threshold: float = 0.4
    resolve_timeout: float = 10.0
    predict_accumulation_time: float = 3.0
    predict_agreement_threshold: float = 0.75
    predict_confidence_threshold: float = 0.7
    eeg_stream_name: Optional[str] = None
    marker_stream_name: Optional[str] = None
