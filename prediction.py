from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

logger = logging.getLogger(__name__)


class ObjectClassifier:
    """Binary EEG object classifier using a CSP + LDA pipeline.

    Parameters
    ----------
    sfreq:
        EEG sampling rate (Hz).
    n_channels:
        Number of EEG channels in each incoming epoch.
    class_labels:
        Ordered ``(label_obj1, label_obj2)`` pair used internally by
        ``train()`` and ``predict()``. Defaults to ``(0, 1)``.
    """

    def __init__(
        self,
        sfreq: float = 250.0,
        n_channels: int = 16,
        class_labels: Tuple[int, int] = (0, 1),
    ) -> None:
        self.sfreq = float(sfreq)
        self.n_channels = int(n_channels)
        self.class_labels = class_labels
        self.is_trained: bool = False
        self.csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis()
        logger.info("ObjectClassifier initialized | sfreq=%.1f Hz | n_channels=%d", self.sfreq, self.n_channels)

    def train(self, data: List[np.ndarray], labels: List[int]) -> None:
        """Fit the CSP spatial filter and LDA classifier on labelled epochs.

        Parameters
        ----------
        data:
            List of EEG epochs, each shaped ``(n_channels, n_samples)``.
        labels:
            Integer class label per epoch (0 = OBJ1, 1 = OBJ2).
        """
        X = np.stack(data, axis=0)  # (n_epochs, n_channels, n_samples)
        y = np.array(labels)
        logger.info("Training on %d epochs …", len(y))
        X_features = self.csp.fit_transform(X, y)
        self.lda.fit(X_features, y)
        self.is_trained = True
        logger.info("Training complete.")

    def predict(self, epoch: np.ndarray) -> Tuple[int, float]:
        """Predict the object class for a single EEG epoch.

        Parameters
        ----------
        epoch:
            EEG data shaped ``(n_channels, n_samples)``.

        Returns
        -------
        Tuple[int, float]
            ``(predicted_label, confidence)`` where confidence is the
            maximum class probability from the LDA decision function.

        Raises
        ------
        RuntimeError
            If called before ``train()``.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        X = epoch[np.newaxis, :, :]
        X_features = self.csp.transform(X)
        pred_label = int(self.lda.predict(X_features)[0])
        confidence = float(np.max(self.lda.predict_proba(X_features)[0]))
        return pred_label, confidence

    def reset(self) -> None:
        """Revert to untrained state (call between experimental sessions)."""
        self.is_trained = False
        self.csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis()
        logger.info("ObjectClassifier reset.")
