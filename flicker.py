"""
flicker.py
==========
SSVEP Flicker Detection - Ensemble Approach
-------------------------------------------
Implements ``SSVEPDetector``, which analyses a multi-channel EEG epoch and
determines whether a steady-state visual evoked potential (SSVEP) is present
at a target stimulus frequency using three independent methods:

    A) FFT  - spectral amplitude at f, 2f, 3f.
    B) PSD  - Welch power-spectral density with neighbour-normalised SNR.
    C) CCA  - Canonical Correlation Analysis against synthetic reference signals.

A weighted ensemble combines all three scores into a final binary decision and
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
    fft_score:
        Normalised FFT sub-score (0-1).
    psd_score:
        Normalised Welch-SNR sub-score (0-1).
    cca_score:
        CCA canonical correlation coefficient (0-1).
    """
    code: BCICode
    detected_frequency: float
    confidence_score: float
    ssvep_present: bool
    fft_score: float = 0.0
    psd_score: float = 0.0
    cca_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "Detected_Frequency": self.detected_frequency,
            "Confidence_Score":   round(self.confidence_score, 4),
            "SSVEP_Present":      self.ssvep_present,
            "FFT_Score":          round(self.fft_score, 4),
            "PSD_Score":          round(self.psd_score, 4),
            "CCA_Score":          round(self.cca_score, 4),
        }


# ---------------------------------------------------------------------------
# SSVEPDetector
# ---------------------------------------------------------------------------

class SSVEPDetector:
    """Full-pipeline SSVEP detector using an FFT + PSD + CCA ensemble.

    Parameters
    ----------
    target_freq:
        Stimulus frequency to test for (Hz), e.g. 10.0 or 12.0.
    sfreq:
        EEG sampling rate (Hz), e.g. 250.
    n_harmonics:
        Number of harmonics (including fundamental) used in FFT and CCA.
        Default: 3  →  f, 2f, 3f.
    bandpass_low:
        Low-cut frequency for the pre-processing bandpass filter (Hz).
    bandpass_high:
        High-cut frequency for the pre-processing bandpass filter (Hz).
    notch_freq:
        Power-line notch frequency (Hz). Set to ``None`` to skip.
    notch_width:
        Width of the notch band (Hz). Default: 2.0.
    occipital_channels:
        List of 0-based channel indices used in CCA (occipital electrodes).
        Defaults to ``[6, 7]`` (O1, O2 in a standard 8-ch Cyton layout).
    weights:
        Tuple ``(w_fft, w_psd, w_cca)`` controlling the ensemble blend.
        Values need not sum to 1 - they are normalised internally.
    detection_threshold:
        Minimum weighted score required to classify SSVEP as *present*.
    psd_snr_neighbors:
        Number of frequency bins on each side of the target used as the
        noise baseline for the Welch SNR estimate.
    welch_nperseg:
        Segment length (samples) passed to ``scipy.signal.welch``.
        ``None`` → library default.
    filter_order:
        Butterworth filter order for bandpass and notch.
    """

    def __init__(
        self,
        target_freq: float,
        sfreq: float = 250.0,
        n_harmonics: int = 3,
        bandpass_low: float = 1.0,
        bandpass_high: float = 40.0,
        notch_freq: float | None = 50.0,
        notch_width: float = 2.0,
        occipital_channels: List[int] | None = None,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.5),
        detection_threshold: float = 0.55,
        psd_snr_neighbors: int = 4,
        welch_nperseg: int | None = None,
        filter_order: int = 4,
    ) -> None:
        self.target_freq = float(target_freq)
        self.sfreq = float(sfreq)
        self.n_harmonics = max(1, int(n_harmonics))
        self.bandpass_low = float(bandpass_low)
        self.bandpass_high = float(bandpass_high)
        self.notch_freq = notch_freq
        self.notch_width = float(notch_width)
        self.occipital_channels: List[int] = (
            occipital_channels if occipital_channels is not None else [6, 7]
        )
        # Normalise weights so they sum to 1
        w = np.array(weights, dtype=float)
        self.weights: np.ndarray = w / w.sum()
        self.detection_threshold = float(detection_threshold)
        self.psd_snr_neighbors = int(psd_snr_neighbors)
        self.welch_nperseg = welch_nperseg
        self.filter_order = int(filter_order)

        # Pre-compute filter coefficients
        self._bp_sos = self._design_bandpass()
        self._notch_b: np.ndarray | None = None
        self._notch_a: np.ndarray | None = None
        if self.notch_freq is not None:
            self._notch_b, self._notch_a = self._design_notch()

        logger.info(
            "SSVEPDetector initialised | target=%.1f Hz | sfreq=%.1f Hz | "
            "harmonics=%d | weights=%s | threshold=%.2f",
            self.target_freq, self.sfreq, self.n_harmonics,
            np.round(self.weights, 3), self.detection_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, epoch: np.ndarray) -> FlickerResult:
        """Run the full ensemble detection pipeline on a single EEG epoch.

        Parameters
        ----------
        epoch:
            Raw EEG data shaped ``(n_channels, n_samples)``.

        Returns
        -------
        FlickerResult
            Structured result including per-method scores and the final
            ensemble decision.
        """
        if epoch.ndim != 2:
            raise ValueError(
                f"epoch must be 2-D (n_channels, n_samples), got shape {epoch.shape}."
            )

        # Step 1 - pre-processing (works on a copy)
        filtered = self._preprocess(epoch)

        # Step 2 - individual method scores
        fft_score = self._score_fft(filtered)
        psd_score = self._score_psd(filtered)
        cca_score = self._score_cca(filtered)

        # Step 3 - weighted ensemble
        scores = np.array([fft_score, psd_score, cca_score])
        ensemble_score = float(np.dot(self.weights, scores))

        ssvep_present = ensemble_score >= self.detection_threshold
        code = (
            BCICode.FLICKER_DETECTED
            if ssvep_present
            else BCICode.FLICKER_NOT_DETECTED
        )

        logger.debug(
            "FFT=%.3f | PSD=%.3f | CCA=%.3f | Ensemble=%.3f | Decision=%s",
            fft_score, psd_score, cca_score, ensemble_score, code.name,
        )

        return FlickerResult(
            code=code,
            detected_frequency=self.target_freq,
            confidence_score=ensemble_score,
            ssvep_present=ssvep_present,
            fft_score=fft_score,
            psd_score=psd_score,
            cca_score=cca_score,
        )

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, epoch: np.ndarray) -> np.ndarray:
        """Apply bandpass and notch filters to a copy of ``epoch``.

        Parameters
        ----------
        epoch:
            Shape ``(n_channels, n_samples)``.

        Returns
        -------
        np.ndarray
            Filtered copy, same shape.
        """
        data = epoch.copy().astype(np.float64)

        # Bandpass
        data = sp_signal.sosfiltfilt(self._bp_sos, data, axis=1)

        # Notch (optional)
        if self._notch_b is not None and self._notch_a is not None:
            data = sp_signal.filtfilt(self._notch_b, self._notch_a, data, axis=1)

        return data

    # ------------------------------------------------------------------
    # Method A - FFT
    # ------------------------------------------------------------------

    def _score_fft(self, data: np.ndarray) -> float:
        """Compute a normalised FFT score across harmonics.

        The score equals the mean amplitude at {f, 2f, …, n_harmonics*f}
        normalised by the mean spectral amplitude in the full band, clamped
        to [0, 1].

        Parameters
        ----------
        data:
            Filtered epoch ``(n_channels, n_samples)``.

        Returns
        -------
        float
            Score in [0, 1].
        """
        n_samples = data.shape[1]
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.sfreq)
        spectrum = np.abs(np.fft.rfft(data, axis=1))  # (n_ch, n_freqs)

        # Mean across channels
        mean_spectrum = spectrum.mean(axis=0)
        mean_total = mean_spectrum.mean() + 1e-12

        harmonic_amplitudes: List[float] = []
        for h in range(1, self.n_harmonics + 1):
            target_hz = self.target_freq * h
            idx = int(np.argmin(np.abs(freqs - target_hz)))
            harmonic_amplitudes.append(float(mean_spectrum[idx]))

        raw_score = float(np.mean(harmonic_amplitudes)) / mean_total
        return float(np.clip(raw_score / 10.0, 0.0, 1.0))  # empirical normaliser

    # ------------------------------------------------------------------
    # Method B - PSD / Welch SNR
    # ------------------------------------------------------------------

    def _score_psd(self, data: np.ndarray) -> float:
        """Compute a normalised SNR score using Welch PSD.

        SNR is defined as the power at the fundamental frequency divided by
        the mean power in the ``psd_snr_neighbors`` bins on each side.
        The SNR is log-compressed and normalised to [0, 1].

        Parameters
        ----------
        data:
            Filtered epoch ``(n_channels, n_samples)``.

        Returns
        -------
        float
            Score in [0, 1].
        """
        nperseg = self.welch_nperseg or data.shape[1] // 2

        snr_values: List[float] = []
        for ch_data in data:
            f, pxx = sp_signal.welch(
                ch_data,
                fs=self.sfreq,
                nperseg=min(nperseg, len(ch_data)),
            )
            idx = int(np.argmin(np.abs(f - self.target_freq)))
            # Guard against edge bins
            lo = max(0, idx - self.psd_snr_neighbors)
            hi = min(len(pxx) - 1, idx + self.psd_snr_neighbors)
            neighbors = np.concatenate([pxx[lo:idx], pxx[idx + 1:hi + 1]])
            noise_floor = float(neighbors.mean()) + 1e-30
            snr = float(pxx[idx]) / noise_floor
            snr_values.append(snr)

        mean_snr = float(np.mean(snr_values))
        # Log-compress: SNR of 1 → 0, typical SSVEP SNR of ~5-20 → mid-range
        log_snr = np.log10(max(mean_snr, 1.0))
        return float(np.clip(log_snr / 2.0, 0.0, 1.0))  # ~SNR 100 → score 1.0

    # ------------------------------------------------------------------
    # Method C - CCA
    # ------------------------------------------------------------------

    def _score_cca(self, data: np.ndarray) -> float:
        """Compute the maximum canonical correlation coefficient (CCA).

        Reference signals are synthetic sinusoids at the fundamental and all
        requested harmonics.  The EEG matrix is formed from the occipital
        channels defined in ``self.occipital_channels``.

        The canonical correlation is computed via the thin SVD of the
        whitened cross-covariance matrix (standard CCA formulation).

        Parameters
        ----------
        data:
            Filtered epoch ``(n_channels, n_samples)``.

        Returns
        -------
        float
            Maximum canonical correlation coefficient in [0, 1].
        """
        # Select occipital channels; clip indices to valid range
        valid_ch = [
            ch for ch in self.occipital_channels if ch < data.shape[0]
        ]
        if not valid_ch:
            logger.warning(
                "CCA: no valid occipital channels found (n_channels=%d). "
                "Returning 0.", data.shape[0]
            )
            return 0.0

        X = data[valid_ch, :].T  # (n_samples, n_occ_channels)
        Y = self._build_reference(data.shape[1]).T  # (n_samples, 2*n_harmonics)

        try:
            rho = self._canonical_correlation(X, Y)
        except np.linalg.LinAlgError as exc:
            logger.warning("CCA failed with LinAlgError: %s", exc)
            return 0.0

        return float(np.clip(rho, 0.0, 1.0))

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

    def _design_bandpass(self) -> np.ndarray:
        """Design a Butterworth bandpass filter; return SOS coefficients."""
        nyq = self.sfreq / 2.0
        low = self.bandpass_low / nyq
        high = min(self.bandpass_high / nyq, 0.999)
        sos = sp_signal.butter(
            self.filter_order,
            [low, high],
            btype="bandpass",
            output="sos",
        )
        return sos

    def _design_notch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design an IIR notch filter; return (b, a) coefficients."""
        nyq = self.sfreq / 2.0
        low = (self.notch_freq - self.notch_width / 2.0) / nyq
        high = (self.notch_freq + self.notch_width / 2.0) / nyq
        low = max(low, 1e-4)
        high = min(high, 0.9999)
        b, a = sp_signal.butter(
            self.filter_order,
            [low, high],
            btype="bandstop",
        )
        return b, a
