from __future__ import annotations
from collections import Counter, deque
from typing import Deque, Tuple, Optional, Dict, Any

class PredictionAccumulator:
    def __init__(
        self,
        maxlen: int,
        agreement_threshold: float,
        confidence_threshold: float,
    ) -> None:
        self._buffer: Deque[Tuple[int, float, str, float]] = deque(maxlen=maxlen)
        self.agreement_threshold = agreement_threshold
        self.confidence_threshold = confidence_threshold

    def append(self, pred: int, conf: float, imagery_str: str, imagery_conf: float) -> None:
        self._buffer.append((pred, conf, imagery_str, imagery_conf))

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def maxlen(self) -> int:
        return self._buffer.maxlen if self._buffer.maxlen is not None else 1

    def is_full(self) -> bool:
        return len(self._buffer) == self._buffer.maxlen

    def get_stable_prediction(self, model_name: str) -> Optional[Dict[str, Any]]:
        if len(self._buffer) < self._buffer.maxlen:
            return None

        counts = Counter([p[0] for p in self._buffer])
        most_common_pred, most_common_count = counts.most_common(1)[0]
        agreement = most_common_count / len(self._buffer)
        avg_conf = sum([p[1] for p in self._buffer]) / len(self._buffer)

        if agreement >= self.agreement_threshold and avg_conf >= self.confidence_threshold:
            last_imagery_str = self._buffer[-1][2]
            avg_imagery_conf = sum([p[3] for p in self._buffer]) / len(self._buffer)
            
            return {
                "prediction": most_common_pred,
                "confidence": avg_conf,
                "agreement": agreement,
                "imagery_prediction": last_imagery_str,
                "imagery_confidence": avg_imagery_conf,
                "model_name": model_name,
            }
        return None
