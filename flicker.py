"""
flicker.py
==========
SSVEP Flicker Detection - Ensemble Approach
-------------------------------------------
Implements ``SSVEPDetector``, which analyses a multi-channel EEG epoch and
determines whether a steady-state visual evoked potential (SSVEP) is present
at a target stimulus frequency using two independent methods:

    A) PSD  - Welch power-spectral density with neighbour-normalised SNR.
    B) CCA  - Canonical Correlation Analysis against synthetic reference signals.

A weighted ensemble combines both scores into a final binary decision and
a continuous confidence value.

Design notes
------------
* Channels are 0-indexed.  O1 → index 6, O2 → index 7 (standard 8/16-ch
  OpenBCI Cyton mapping). Override via ``occipital_channels``.
* All numpy arrays are assumed to be shape ``(n_channels, n_samples)`` at
  the public API boundary so that callers match the OpenBCI / MNE convention.
* Filtering is applied in-place on a *copy* of the incoming data, keeping the
  original epoch unmodified.

Dependencies: numpy, scipy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.linalg import svd

from protocol import BCICode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return-value container
# ---------------------------------------------------------------------------

@dataclass
class FlickerResult:
    """Structured result returned by :meth:`SSVEPDetector.detect`.

    Attributes
    ----------
    code:
        ``BCICode.FLICKER_DETECTED`` or ``BCICode.FLICKER_NOT_DETECTED``.
    detected_frequency:
        The target frequency that was tested (Hz).
    confidence_score:
        Weighted ensemble confidence in [0, 1].
    ssvep_present:
        Boolean summary of the decision.
    psd_score:
        Normalised Welch-SNR sub-score (0-1).
    cca_score:
        CCA canonical correlation coefficient (0-1).
    """
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


# ---------------------------------------------------------------------------
# SSVEPDetector
# ---------------------------------------------------------------------------

class SSVEPDetector:
    """Full-pipeline SSVEP detector using a PSD + FBCCA ensemble.

    Parameters
    ----------
    target_freq:
        Stimulus frequency to test for (Hz), e.g. 10.0 or 12.0.
    sfreq:
        EEG sampling rate (Hz), e.g. 250.
    n_harmonics:
        Number of harmonics (including fundamental) used in FBCCA.
        Default: 3  →  f, 2f, 3f.

    occipital_channels:
        List of 0-based channel indices used in FBCCA (occipital electrodes).
        Defaults to ``[6, 7]`` (O1, O2 in a standard 8-ch Cyton layout).
    detection_threshold:
        Minimum score required to classify SSVEP as *present*.
    filter_order:
        Butterworth filter order for bandpass and notch.
    fbcca_num_bands:
        Number of sub-bands for FBCCA.
    fbcca_a:
        Parameter 'a' for FBCCA sub-band weights w(n) = n^(-a) + b.
    fbcca_b:
        Parameter 'b' for FBCCA sub-band weights w(n) = n^(-a) + b.
    fbcca_band_width:
        Frequency step for FBCCA sub-bands.
    """

    def __init__(
        self,
        target_freq: float,
        sfreq: float = 250.0,
        n_harmonics: int = 3,
        occipital_channels: List[int] | None = None,
        detection_threshold: float = 0.4,
        filter_order: int = 4,
        fbcca_num_bands: int = 5,
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

        # Pre-compute FBCCA filter coefficients
        self._fbcca_sos_list = self._design_fbcca_filters()

        logger.info(
            "SSVEPDetector initialised | target=%.1f Hz | sfreq=%.1f Hz | "
            "harmonics=%d | threshold=%.2f",
            self.target_freq, self.sfreq, self.n_harmonics,
            self.detection_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, epoch: np.ndarray) -> FlickerResult:
        """Run the FBCCA detection pipeline on a single EEG epoch.

        Parameters
        ----------
        epoch:
            Raw EEG data shaped ``(n_channels, n_samples)``.

        Returns
        -------
        FlickerResult
            Structured result including scores and the decision.
        """
        if epoch.ndim != 2:
            raise ValueError(
                f"epoch must be 2-D (n_channels, n_samples), got shape {epoch.shape}."
            )

        # Apply FBCCA to the raw epoch (which is already notch-filtered globally)
        fbcca_score = self._score_fbcca(epoch)

        ssvep_present = fbcca_score >= self.detection_threshold
        code = (
            BCICode.FLICKER_DETECTED
            if ssvep_present
            else BCICode.FLICKER_NOT_DETECTED
        )

        logger.debug(
            "FBCCA=%.3f | Decision=%s",
            fbcca_score, code.name,
        )

        return FlickerResult(
            code=code,
            detected_frequency=self.target_freq,
            confidence_score=fbcca_score,
            ssvep_present=ssvep_present,
            fbcca_score=fbcca_score,
        )

    # ------------------------------------------------------------------
    # Method B - FBCCA
    # ------------------------------------------------------------------

    def _score_fbcca(self, data: np.ndarray) -> float:
        """Compute the Filter Bank Canonical Correlation Analysis (FBCCA) score.

        Decomposes the data into multiple sub-bands, computes CCA for each,
        and returns a weighted sum of the canonical correlations.

        Parameters
        ----------
        data:
            Raw or notch-filtered epoch ``(n_channels, n_samples)``.
            Should NOT be heavily bandpass-filtered beforehand to preserve harmonics.

        Returns
        -------
        float
            Weighted FBCCA score in [0, 1].
        """
        valid_ch = [
            ch for ch in self.occipital_channels if ch < data.shape[0]
        ]
        if not valid_ch:
            logger.warning(
                "FBCCA: no valid occipital channels found (n_channels=%d). "
                "Returning 0.", data.shape[0]
            )
            return 0.0

        fbcca_score = 0.0
        Y = self._build_reference(data.shape[1]).T  # (n_samples, 2*n_harmonics)

        for n, sos in enumerate(self._fbcca_sos_list, start=1):
            # Filter data for this sub-band
            sub_data = sp_signal.sosfiltfilt(sos, data, axis=1)
            X = sub_data[valid_ch, :].T  # (n_samples, n_occ_channels)

            try:
                rho = self._canonical_correlation(X, Y)
            except np.linalg.LinAlgError as exc:
                logger.warning("FBCCA sub-band %d failed: %s", n, exc)
                rho = 0.0
            
            # Weight: W(n) = n^(-a) + b
            w = (n ** -self.fbcca_a) + self.fbcca_b
            fbcca_score += w * rho

        return float(np.clip(fbcca_score, 0.0, 1.0))

    def _build_reference(self, n_samples: int) -> np.ndarray:
        """Build a reference matrix of sine/cosine pairs at each harmonic.

        Parameters
        ----------
        n_samples:
            Number of time samples.

        Returns
        -------
        np.ndarray
            Shape ``(2 * n_harmonics, n_samples)``.
        """
        t = np.arange(n_samples) / self.sfreq
        rows: List[np.ndarray] = []
        for h in range(1, self.n_harmonics + 1):
            rows.append(np.sin(2.0 * np.pi * self.target_freq * h * t))
            rows.append(np.cos(2.0 * np.pi * self.target_freq * h * t))
        return np.vstack(rows)  # (2*n_harmonics, n_samples)

    @staticmethod
    def _canonical_correlation(X: np.ndarray, Y: np.ndarray) -> float:
        """Return the largest canonical correlation coefficient between X and Y.

        Parameters
        ----------
        X:
            Matrix ``(n_samples, p)``.
        Y:
            Matrix ``(n_samples, q)``.

        Returns
        -------
        float
            Largest canonical correlation in [0, 1].
        """
        def _centre(M: np.ndarray) -> np.ndarray:
            return M - M.mean(axis=0)

        X = _centre(X)
        Y = _centre(Y)
        n = X.shape[0]

        Cxx = X.T @ X / (n - 1) + np.eye(X.shape[1]) * 1e-8
        Cyy = Y.T @ Y / (n - 1) + np.eye(Y.shape[1]) * 1e-8
        Cxy = X.T @ Y / (n - 1)

        # Whitening matrices
        Lxx = np.linalg.cholesky(Cxx)
        Lyy = np.linalg.cholesky(Cyy)

        M = np.linalg.solve(Lxx, Cxy)
        M = np.linalg.solve(Lyy, M.T).T

        _, singular_values, _ = svd(M, full_matrices=False)
        return float(singular_values[0])

    # ------------------------------------------------------------------
    # Filter design helpers
    # ------------------------------------------------------------------

    def _design_fbcca_filters(self) -> List[np.ndarray]:
        """Design Butterworth bandpass filters for each FBCCA sub-band."""
        sos_list = []
        nyq = self.sfreq / 2.0
        high = min(90.0 / nyq, 0.999)
        for i in range(1, self.fbcca_num_bands + 1):
            low = (i * self.fbcca_band_width) / nyq
            if low >= high:
                logger.warning(
                    "FBCCA sub-band %d low cutoff (%.1f Hz) exceeds high cutoff (%.1f Hz). "
                    "Using default offset.", i, low * nyq, high * nyq
                )
                low = max(1e-4, high - 0.01)
            sos = sp_signal.butter(
                self.filter_order,
                [low, high],
                btype="bandpass",
                output="sos",
            )
            sos_list.append(sos)
        return sos_list

