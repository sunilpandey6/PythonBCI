from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import signal as sp_signal
from scipy.linalg import svd

from bci.domain.codes import BCICode

logger = logging.getLogger(__name__)


@dataclass
class FlickerResult:
    code: BCICode
    detected_frequency: float
    confidence_score: float
    ssvep_present: bool
    fbcca_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "Detected_Frequency": self.detected_frequency,
            "Confidence_Score":   round(self.confidence_score, 4),
            "SSVEP_Present":      self.ssvep_present,
            "FBCCA_Score":        round(self.fbcca_score, 4),
        }


class SSVEPDetector:
    """FBCCA-based SSVEP detector. Channels O1/O2 are indices 6 and 7 (standard 16-ch Cyton)."""

    def __init__(
        self,
        target_freq: float,
        sfreq: float = 125.0,
        n_harmonics: int = 3,
        occipital_channels: List[int] | None = None,
        detection_threshold: float = 0.5,
        filter_order: int = 4,
        fbcca_num_bands: int = 3,
        fbcca_a: float = 1.25,
        fbcca_b: float = 0.25,
        fbcca_band_width: float = 8.0,
    ) -> None:
        self.target_freq = float(target_freq)
        self.sfreq = float(sfreq)
        self.n_harmonics = max(1, int(n_harmonics))
        self.occipital_channels: List[int] = (
            occipital_channels if occipital_channels is not None else [6, 7]
        )
        self.detection_threshold = float(detection_threshold)
        self.filter_order = int(filter_order)
        self.fbcca_num_bands = int(fbcca_num_bands)
        self.fbcca_a = float(fbcca_a)
        self.fbcca_b = float(fbcca_b)
        self.fbcca_band_width = float(fbcca_band_width)

        self._fbcca_sos_list = self._design_fbcca_filters()

        logger.info(
            "SSVEPDetector initialised | target=%.1f Hz | sfreq=%.1f Hz | harmonics=%d | threshold=%.2f",
            self.target_freq, self.sfreq, self.n_harmonics, self.detection_threshold,
        )

    def detect(self, epoch: np.ndarray) -> FlickerResult:
        if epoch.ndim != 2:
            raise ValueError(f"epoch must be 2-D (n_channels, n_samples), got shape {epoch.shape}.")

        fbcca_score = self._score_fbcca(epoch)
        ssvep_present = fbcca_score >= self.detection_threshold
        code = BCICode.FLICKER_DETECTED if ssvep_present else BCICode.FLICKER_NOT_DETECTED

        logger.debug("FBCCA=%.3f | Decision=%s", fbcca_score, code.name)

        return FlickerResult(
            code=code,
            detected_frequency=self.target_freq,
            confidence_score=fbcca_score,
            ssvep_present=ssvep_present,
            fbcca_score=fbcca_score,
        )

    def _score_fbcca(self, data: np.ndarray) -> float:
        valid_ch = [ch for ch in self.occipital_channels if ch < data.shape[0]]
        if not valid_ch:
            logger.warning("FBCCA: no valid occipital channels (n_channels=%d). Returning 0.", data.shape[0])
            return 0.0

        fbcca_score = 0.0
        Y = self._build_reference(data.shape[1]).T

        for n, sos in enumerate(self._fbcca_sos_list, start=1):
            sub_data = sp_signal.sosfiltfilt(sos, data, axis=1)
            X = sub_data[valid_ch, :].T
            try:
                rho = self._canonical_correlation(X, Y)
            except np.linalg.LinAlgError as exc:
                logger.warning("FBCCA sub-band %d failed: %s", n, exc)
                rho = 0.0
            w = (n ** -self.fbcca_a) + self.fbcca_b
            fbcca_score += w * rho
        return float(np.clip(fbcca_score, 0.0, 1.0))

    def _build_reference(self, n_samples: int) -> np.ndarray:
        t = np.arange(n_samples) / self.sfreq
        rows: List[np.ndarray] = []
        for h in range(1, self.n_harmonics + 1):
            rows.append(np.sin(2.0 * np.pi * self.target_freq * h * t))
            rows.append(np.cos(2.0 * np.pi * self.target_freq * h * t))
        return np.vstack(rows)

    @staticmethod
    def _canonical_correlation(X: np.ndarray, Y: np.ndarray) -> float:
        def _centre(M: np.ndarray) -> np.ndarray:
            return M - M.mean(axis=0)
        X = _centre(X)
        Y = _centre(Y)
        n = X.shape[0]
        Cxx = X.T @ X / (n - 1) + np.eye(X.shape[1]) * 1e-8
        Cyy = Y.T @ Y / (n - 1) + np.eye(Y.shape[1]) * 1e-8
        Cxy = X.T @ Y / (n - 1)
        Lxx = np.linalg.cholesky(Cxx)
        Lyy = np.linalg.cholesky(Cyy)
        M = np.linalg.solve(Lxx, Cxy)
        M = np.linalg.solve(Lyy, M.T).T
        _, singular_values, _ = svd(M, full_matrices=False)
        return float(singular_values[0])

    def _design_fbcca_filters(self) -> List[np.ndarray]:
        sos_list = []

        nyq = self.sfreq / 2.0
        start = 6.0

        for i in range(self.fbcca_num_bands):

            low = start + i * self.fbcca_band_width
            high = low + self.fbcca_band_width

            # keep safely below Nyquist
            if high >= nyq:
                high = nyq - 0.5

            # skip invalid bands
            if low >= high or low <= 0:
                logger.warning(
                    "Skipping invalid FBCCA band: low=%.2f high=%.2f",
                    low,
                    high,
                )
                continue

            sos = sp_signal.butter(
                self.filter_order,
                [low, high],
                btype="bandpass",
                fs=self.sfreq,
                output="sos",
            )

            sos_list.append(sos)

            logger.info(
                "FBCCA band %d: %.1f-%.1f Hz",
                i + 1,
                low,
                high,
            )

        return sos_list
