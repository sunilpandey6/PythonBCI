from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from mne.decoding import CSP, Vectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class ImageryClassifier:
    """CSP + LDA classifier for motor imagery (Door 1 vs Door 2)."""

    def __init__(self, sfreq: float = 125.0, n_channels: int = 16, class_labels: Tuple[int, int] = (0, 1)) -> None:
        self.sfreq = float(sfreq)
        self.n_channels = int(n_channels)
        self.class_labels = class_labels
        self.is_trained: bool = False
        self.csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis()
        logger.info("ImageryClassifier initialized (CSP+LDA) | sfreq=%.1f Hz | n_channels=%d", self.sfreq, self.n_channels)

    def train(self, data: List[np.ndarray], labels: List[int]) -> None:
        X = np.stack(data, axis=0)
        y = np.array(labels)
        logger.info("Training ImageryClassifier on %d epochs …", len(y))
        X_features = self.csp.fit_transform(X, y)
        self.lda.fit(X_features, y)
        self.is_trained = True
        logger.info("ImageryClassifier training complete.")

    def predict(self, epoch: np.ndarray) -> Tuple[int, float]:
        if not self.is_trained:
            raise RuntimeError("ImageryClassifier is not trained yet.")
        X = epoch[np.newaxis, :, :]
        X_features = self.csp.transform(X)
        pred_label = int(self.lda.predict(X_features)[0])
        confidence = float(np.max(self.lda.predict_proba(X_features)[0]))
        return pred_label, confidence

    def reset(self) -> None:
        self.is_trained = False
        self.csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis()
        logger.info("ImageryClassifier reset.")


class ActiveClassifier:
    """Vectorizer + StandardScaler + SVM classifier for active attention (Door 1 vs Door 2)."""

    def __init__(self, sfreq: float = 125.0, n_channels: int = 16, class_labels: Tuple[int, int] = (0, 1)) -> None:
        self.sfreq = float(sfreq)
        self.n_channels = int(n_channels)
        self.class_labels = class_labels
        self.is_trained: bool = False
        self.pipeline = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            SVC(kernel='linear', probability=True, class_weight='balanced')
        )
        logger.info("ActiveClassifier initialized (Vectorizer+SVM) | sfreq=%.1f Hz | n_channels=%d", self.sfreq, self.n_channels)

    def train(self, data: List[np.ndarray], labels: List[int]) -> None:
        X = np.stack(data, axis=0)
        y = np.array(labels)
        logger.info("Training ActiveClassifier on %d epochs …", len(y))
        self.pipeline.fit(X, y)
        self.is_trained = True
        logger.info("ActiveClassifier training complete.")

    def predict(self, epoch: np.ndarray) -> Tuple[int, float]:
        if not self.is_trained:
            raise RuntimeError("ActiveClassifier is not trained yet.")
        X = epoch[np.newaxis, :, :]
        pred_label = int(self.pipeline.predict(X)[0])
        confidence = float(np.max(self.pipeline.predict_proba(X)[0]))
        return pred_label, confidence

    def reset(self) -> None:
        self.is_trained = False
        self.pipeline = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            SVC(kernel='linear', probability=True, class_weight='balanced')
        )
        logger.info("ActiveClassifier reset.")


class MixedClassifier:
    """Vectorizer + StandardScaler + SVM classifier trained on combined active + imagery data."""

    def __init__(self, sfreq: float = 125.0, n_channels: int = 16, class_labels: Tuple[int, int] = (0, 1)) -> None:
        self.sfreq = float(sfreq)
        self.n_channels = int(n_channels)
        self.class_labels = class_labels
        self.is_trained: bool = False
        self.pipeline = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            SVC(kernel='linear', probability=True, class_weight='balanced')
        )
        logger.info("MixedClassifier initialized (Vectorizer+SVM) | sfreq=%.1f Hz | n_channels=%d", self.sfreq, self.n_channels)

    def train(self, data: List[np.ndarray], labels: List[int]) -> None:
        X = np.stack(data, axis=0)
        y = np.array(labels)
        logger.info("Training MixedClassifier on %d epochs …", len(y))
        self.pipeline.fit(X, y)
        self.is_trained = True
        logger.info("MixedClassifier training complete.")

    def predict(self, epoch: np.ndarray) -> Tuple[int, float]:
        if not self.is_trained:
            raise RuntimeError("MixedClassifier is not trained yet.")
        X = epoch[np.newaxis, :, :]
        pred_label = int(self.pipeline.predict(X)[0])
        confidence = float(np.max(self.pipeline.predict_proba(X)[0]))
        return pred_label, confidence

    def reset(self) -> None:
        self.is_trained = False
        self.pipeline = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            SVC(kernel='linear', probability=True, class_weight='balanced')
        )
        logger.info("MixedClassifier reset.")
