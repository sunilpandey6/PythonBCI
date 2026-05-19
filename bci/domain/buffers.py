from __future__ import annotations
from collections import deque
from typing import Optional
import numpy as np

class EEGBuffer:
    def __init__(self, max_samples: int) -> None:
        self._buffer: deque[np.ndarray] = deque(maxlen=max_samples)
        self.max_samples = max_samples

    def append(self, sample: np.ndarray) -> None:
        self._buffer.append(sample)

    def clear(self) -> None:
        self._buffer.clear()

    def is_ready(self) -> bool:
        return len(self._buffer) >= self.max_samples

    def snapshot(self) -> Optional[np.ndarray]:
        if len(self._buffer) < self.max_samples:
            return None
        return np.array(self._buffer, dtype=np.float64).T

class EpochBuffer:
    def __init__(self, max_epochs: int) -> None:
        self._buffer: deque[np.ndarray] = deque(maxlen=max_epochs)
        self.max_epochs = max_epochs

    def append(self, epoch: np.ndarray) -> None:
        self._buffer.append(epoch)

    def clear(self) -> None:
        self._buffer.clear()

    def get_all(self) -> list[np.ndarray]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)
