from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from flicker import FlickerResult, SSVEPDetector
from prediction import ActiveClassifier, ImageryClassifier
from protocol import BCICode, build_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bci.main")

try:
    from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop  # type: ignore
    _LSL_AVAILABLE = True
except ImportError:
    _LSL_AVAILABLE = False
    logger.warning("pylsl not found. Running in SIMULATION mode - no real LSL streams.")
"""LSL stream names in unity"""
EEG_STREAM_TYPE: str = "EEG"
MARKER_STREAM_TYPE: str = "Markers"
"""LSL stream names in python"""
OUTPUT_STREAM_NAME: str = "BCIBackend"
OUTPUT_STREAM_TYPE: str = "BCIResult"
OUTPUT_STREAM_CHANNELS: int = 1
N_CHANNELS: int = 16
DEFAULT_SFREQ: float = 250.0


class BCIState:
    IDLE = "IDLE"
    SSVEP_TEST = "SSVEP_TEST"
    TRAIN_ACTIVE_OBJ1 = "TRAIN_ACTIVE_OBJ1"
    TRAIN_ACTIVE_OBJ2 = "TRAIN_ACTIVE_OBJ2"
    TRAIN_IMAGERY_OBJ1 = "TRAIN_IMAGERY_OBJ1"
    TRAIN_IMAGERY_OBJ2 = "TRAIN_IMAGERY_OBJ2"
    PREDICT_ACTIVE = "PREDICT_ACTIVE"
    PREDICT_IMAGERY = "PREDICT_IMAGERY"


class BCIBackend:
    def __init__(
        self,
        target_freq: float = 10.0,
        sfreq: float = DEFAULT_SFREQ,
        epoch_duration: float = 4.0,
        eeg_stream_name: Optional[str] = None,
        marker_stream_name: Optional[str] = None,
        n_train_epochs: int = 10,
        detection_threshold: float = 0.55,
        resolve_timeout: float = 10.0,
    ) -> None:
        self.target_freq = float(target_freq)
        self.sfreq = float(sfreq)
        self.epoch_duration = float(epoch_duration)
        self.eeg_stream_name = eeg_stream_name
        self.marker_stream_name = marker_stream_name
        self.n_train_epochs = int(n_train_epochs)
        self.detection_threshold = float(detection_threshold)
        self.resolve_timeout = float(resolve_timeout)

        self._epoch_samples: int = int(self.sfreq * self.epoch_duration)

        self._shutdown: threading.Event = threading.Event()
        self._eeg_ready: threading.Event = threading.Event()
        self._marker_queue: queue.Queue[str] = queue.Queue()

        self._eeg_buffer: Deque[np.ndarray] = deque(maxlen=self._epoch_samples)
        self._buffer_lock: threading.Lock = threading.Lock()

        self._state: str = BCIState.IDLE
        self._state_lock: threading.Lock = threading.Lock()

        self._detector = SSVEPDetector(
            target_freq=self.target_freq,
            sfreq=self.sfreq,
            detection_threshold=self.detection_threshold,
        )
        self.active_model = ActiveClassifier(sfreq=self.sfreq, n_channels=N_CHANNELS)
        self.imagery_model = ImageryClassifier(sfreq=self.sfreq, n_channels=N_CHANNELS)

        self.active_obj1_epochs: List[np.ndarray] = []
        self.active_obj2_epochs: List[np.ndarray] = []
        self.imagery_obj1_epochs: List[np.ndarray] = []
        self.imagery_obj2_epochs: List[np.ndarray] = []

        self._last_unity_event: str = ""
        self._last_unity_detail: str = ""
        self._unity_event_lock: threading.Lock = threading.Lock()

        self._eeg_inlet: Optional[object] = None
        self._marker_inlet: Optional[object] = None
        self._output_outlet: Optional[object] = None
        self._threads: List[threading.Thread] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        logger.info("BCIBackend starting …")
        self._setup_streams()
        self._setup_output_stream()

        t1 = threading.Thread(target=self._eeg_ingestion_loop, name="Thread-EEG", daemon=True)
        t2 = threading.Thread(target=self._marker_ingestion_loop, name="Thread-Marker", daemon=True)
        t3 = threading.Thread(target=self._logic_loop, name="Thread-Logic", daemon=True)
        self._threads = [t1, t2, t3]
        for t in self._threads:
            t.start()

        logger.info(
            "All threads started. State=%s | Target=%.1f Hz | Epoch=%.1f s",
            self._state, self.target_freq, self.epoch_duration,
        )

    def stop(self) -> None:
        logger.info("Shutdown requested.")
        self._shutdown.set()
        for t in self._threads:
            t.join(timeout=5.0)
        logger.info("BCIBackend stopped.")

    # ------------------------------------------------------------------
    # Thread 1 - EEG ingestion
    # ------------------------------------------------------------------

    def _eeg_ingestion_loop(self) -> None:
        logger.info("[Thread-EEG] started.")
        while not self._shutdown.is_set():
            if not _LSL_AVAILABLE or self._eeg_inlet is None:
                sample = np.random.randn(N_CHANNELS).astype(np.float32)
                time.sleep(1.0 / self.sfreq)
            else:
                sample_list, _ts = self._eeg_inlet.pull_sample(timeout=1.0)
                if sample_list is None:
                    continue
                sample = np.array(sample_list, dtype=np.float32)

            with self._buffer_lock:
                self._eeg_buffer.append(sample)
                if len(self._eeg_buffer) >= self._epoch_samples:
                    self._eeg_ready.set()

        logger.info("[Thread-EEG] stopped.")

    # ------------------------------------------------------------------
    # Thread 2 - Unity Marker ingestion
    # ------------------------------------------------------------------

    def _marker_ingestion_loop(self) -> None:
        logger.info("[Thread-Marker] started.")
        while not self._shutdown.is_set():
            if not _LSL_AVAILABLE or self._marker_inlet is None:
                time.sleep(0.05)
                continue

            sample_list, _ts = self._marker_inlet.pull_sample(timeout=0.0)
            if sample_list is not None and len(sample_list) > 0:
                raw: str = str(sample_list[0])
                logger.debug("[Thread-Marker] received: %s", raw)

                # Unity log format: Time,Experiment,Phase,Event,Detail,Action
                parts = raw.split(",")

                if len(parts) >= 6:
                    event_str = parts[3].strip()
                    detail_str = parts[4].strip()
                    action_str = parts[5].strip()
                else:
                    event_str = raw
                    detail_str = ""
                    action_str = raw

                with self._unity_event_lock:
                    self._last_unity_event = event_str
                    self._last_unity_detail = detail_str

                self._marker_queue.put_nowait(action_str)

            time.sleep(0.001)

        logger.info("[Thread-Marker] stopped.")

    # ------------------------------------------------------------------
    # Thread 3 - Logic / Decision loop
    # ------------------------------------------------------------------

    def _logic_loop(self) -> None:
        logger.info("[Thread-Logic] started. State: %s", self._state)

        while not self._shutdown.is_set():
            while not self._marker_queue.empty():
                try:
                    marker = self._marker_queue.get_nowait()
                except queue.Empty:
                    break
                self._handle_marker(marker)

            current_state = self._get_state()

            if current_state == BCIState.IDLE:
                time.sleep(0.05)
                continue

            if current_state in (
                BCIState.SSVEP_TEST,
                BCIState.TRAIN_ACTIVE_OBJ1,
                BCIState.TRAIN_ACTIVE_OBJ2,
                BCIState.TRAIN_IMAGERY_OBJ1,
                BCIState.TRAIN_IMAGERY_OBJ2,
                BCIState.PREDICT_ACTIVE,
                BCIState.PREDICT_IMAGERY,
            ):
                if not self._eeg_ready.wait(timeout=1.0):
                    continue
                self._eeg_ready.clear()

                epoch = self._snapshot_epoch()
                if epoch is None:
                    continue

                self._process_epoch(epoch, current_state)

        logger.info("[Thread-Logic] stopped.")

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _handle_marker(self, action: str) -> None:
        logger.info("[Logic] Action received: '%s'", action)

        transitions = {
            "Flicker_Start": BCIState.SSVEP_TEST,
            "Flicker_End": BCIState.IDLE,
            "Training_Active_Door1_Start": BCIState.TRAIN_ACTIVE_OBJ1,
            "Training_Active_Door2_Start": BCIState.TRAIN_ACTIVE_OBJ2,
            "Training_Imagery_Door1_Start": BCIState.TRAIN_IMAGERY_OBJ1,
            "Training_Imagery_Door2_Start": BCIState.TRAIN_IMAGERY_OBJ2,
            "Predict_Start_Active": BCIState.PREDICT_ACTIVE,
            "Predict_Start_Imagery": BCIState.PREDICT_IMAGERY,
            "Predict_End": BCIState.IDLE,
        }

        if action in transitions:
            self._set_state(transitions[action])
            return

        if action == "Train_End":
            self._attempt_training()
            self._set_state(BCIState.IDLE)
            return

        logger.debug("[Logic] Unrecognised action: '%s' - ignored.", action)

    def _preprocess_global(self, epoch: np.ndarray) -> np.ndarray:
        """Notch (50 Hz) → Bandpass (8–30 Hz) → Z-score normalisation per channel."""
        from scipy import signal
        nyq = self.sfreq / 2.0
        b_n, a_n = signal.iirnotch(50.0 / nyq, 30.0)
        epoch_filt = signal.filtfilt(b_n, a_n, epoch, axis=-1)
        sos = signal.butter(4, [8.0, 30.0], btype="bandpass", fs=self.sfreq, output="sos")
        epoch_filt = signal.sosfiltfilt(sos, epoch_filt, axis=-1)
        mean = np.mean(epoch_filt, axis=-1, keepdims=True)
        std = np.std(epoch_filt, axis=-1, keepdims=True)
        std[std < 1e-6] = 1.0
        return (epoch_filt - mean) / std

    def _process_epoch(self, epoch: np.ndarray, state: str) -> None:
        epoch = self._preprocess_global(epoch)
        unity_event, unity_detail = self._read_unity_event()

        if state == BCIState.SSVEP_TEST:
            result = self._detector.detect(epoch)
            self._push_flicker_result(result, unity_event, unity_detail)

        elif state == BCIState.TRAIN_ACTIVE_OBJ1:
            self.active_obj1_epochs.append(epoch.copy())
            n = len(self.active_obj1_epochs)
            logger.info("[Logic] TRAIN_ACTIVE_OBJ1 epoch %d/%d.", n, self.n_train_epochs)
            if n >= self.n_train_epochs:
                self._push_message(BCICode.ACTIVE_OBJ1_TRAIN_COMPLETE, unity_event, unity_detail,
                                   remark={"Epochs_Collected": n, "Target_Epochs": self.n_train_epochs, "Object": "OBJ1"})
                self._set_state(BCIState.IDLE)

        elif state == BCIState.TRAIN_ACTIVE_OBJ2:
            self.active_obj2_epochs.append(epoch.copy())
            n = len(self.active_obj2_epochs)
            logger.info("[Logic] TRAIN_ACTIVE_OBJ2 epoch %d/%d.", n, self.n_train_epochs)
            if n >= self.n_train_epochs:
                self._push_message(BCICode.ACTIVE_OBJ2_TRAIN_COMPLETE, unity_event, unity_detail,
                                   remark={"Epochs_Collected": n, "Target_Epochs": self.n_train_epochs, "Object": "OBJ2"})
                self._set_state(BCIState.IDLE)

        elif state == BCIState.TRAIN_IMAGERY_OBJ1:
            self.imagery_obj1_epochs.append(epoch.copy())
            n = len(self.imagery_obj1_epochs)
            logger.info("[Logic] TRAIN_IMAGERY_OBJ1 epoch %d/%d.", n, self.n_train_epochs)
            if n >= self.n_train_epochs:
                self._push_message(BCICode.IMAGERY_OBJ1_TRAIN_COMPLETE, unity_event, unity_detail,
                                   remark={"Epochs_Collected": n, "Target_Epochs": self.n_train_epochs, "Object": "OBJ1"})
                self._set_state(BCIState.IDLE)

        elif state == BCIState.TRAIN_IMAGERY_OBJ2:
            self.imagery_obj2_epochs.append(epoch.copy())
            n = len(self.imagery_obj2_epochs)
            logger.info("[Logic] TRAIN_IMAGERY_OBJ2 epoch %d/%d.", n, self.n_train_epochs)
            if n >= self.n_train_epochs:
                self._push_message(BCICode.IMAGERY_OBJ2_TRAIN_COMPLETE, unity_event, unity_detail,
                                   remark={"Epochs_Collected": n, "Target_Epochs": self.n_train_epochs, "Object": "OBJ2"})
                self._set_state(BCIState.IDLE)

        elif state == BCIState.PREDICT_ACTIVE:
            if self.active_model.is_trained:
                pred, conf = self.active_model.predict(epoch)
                code = BCICode.ACTIVE_OBJ1_PREDICT if pred == 0 else BCICode.ACTIVE_OBJ2_PREDICT
                self._push_message(code, unity_event, unity_detail,
                                   remark={"Model": "ACTIVE", "Prediction": "OBJ1" if pred == 0 else "OBJ2", "Confidence": conf})
            else:
                logger.warning("[Logic] ACTIVE model not trained.")

        elif state == BCIState.PREDICT_IMAGERY:
            if self.imagery_model.is_trained:
                pred, conf = self.imagery_model.predict(epoch)
                code = BCICode.IMAGERY_OBJ1_PREDICT if pred == 0 else BCICode.IMAGERY_OBJ2_PREDICT
                self._push_message(code, unity_event, unity_detail,
                                   remark={"Model": "IMAGERY", "Prediction": "OBJ1" if pred == 0 else "OBJ2", "Confidence": conf})
            else:
                logger.warning("[Logic] IMAGERY model not trained.")

    def _attempt_training(self) -> None:
        if self.active_obj1_epochs and self.active_obj2_epochs:
            X_a = self.active_obj1_epochs + self.active_obj2_epochs
            y_a = [0] * len(self.active_obj1_epochs) + [1] * len(self.active_obj2_epochs)
            self.active_model.train(X_a, y_a)
        else:
            logger.warning("[Logic] ACTIVE training skipped - missing epochs.")

        if self.imagery_obj1_epochs and self.imagery_obj2_epochs:
            X_i = self.imagery_obj1_epochs + self.imagery_obj2_epochs
            y_i = [0] * len(self.imagery_obj1_epochs) + [1] * len(self.imagery_obj2_epochs)
            self.imagery_model.train(X_i, y_i)
        else:
            logger.warning("[Logic] IMAGERY training skipped - missing epochs.")

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def _set_state(self, new_state: str) -> None:
        with self._state_lock:
            logger.info("[Logic] State: %s → %s", self._state, new_state)
            self._state = new_state

    def _get_state(self) -> str:
        with self._state_lock:
            return self._state

    # ------------------------------------------------------------------
    # Epoch snapshot
    # ------------------------------------------------------------------

    def _snapshot_epoch(self) -> Optional[np.ndarray]:
        with self._buffer_lock:
            if len(self._eeg_buffer) < self._epoch_samples:
                return None
            epoch = np.array(self._eeg_buffer, dtype=np.float64).T
        return epoch

    # ------------------------------------------------------------------
    # LSL output helpers
    # ------------------------------------------------------------------

    def _push_flicker_result(self, result: FlickerResult, unity_event: str = "", unity_detail: str = "") -> None:
        msg = build_message(code=result.code, event=unity_event, detail=unity_detail, remark=result.to_dict())
        self._push_raw(msg)
        logger.info("[Output] %s", msg)

    def _push_message(self, code: BCICode, unity_event: str = "", unity_detail: str = "", remark: object = "") -> None:
        msg = build_message(code=code, event=unity_event, detail=unity_detail, remark=remark)
        self._push_raw(msg)
        logger.info("[Output] %s", msg)

    def _push_raw(self, json_str: str) -> None:
        if self._output_outlet is None or not _LSL_AVAILABLE:
            return
        self._output_outlet.push_sample([json_str])

    def _read_unity_event(self) -> tuple[str, str]:
        with self._unity_event_lock:
            return self._last_unity_event, self._last_unity_detail

    def send_test_message(self) -> None:
        self._push_message(
            code=BCICode.FLICKER_DETECTED,
            unity_event="Test_Connection",
            unity_detail="Connection_Check",
            remark={"Detected_Frequency": 12.0, "Confidence_Score": 0.99, "SSVEP_Present": True, "Message": "Test message from Python backend."},
        )
        logger.info("[Test] Test message sent.")

    # ------------------------------------------------------------------
    # LSL stream setup
    # ------------------------------------------------------------------

    def _setup_streams(self) -> None:
        if not _LSL_AVAILABLE:
            logger.warning("LSL not available - running without real streams.")
            return

        logger.info("Resolving EEG stream …")
        eeg_streams = resolve_byprop("type", EEG_STREAM_TYPE, timeout=self.resolve_timeout)
        if eeg_streams:
            self._eeg_inlet = StreamInlet(eeg_streams[0])
            logger.info("EEG inlet connected: %s", eeg_streams[0].name())
        else:
            logger.error("No EEG stream found - running in simulation.")

        logger.info("Resolving Unity Marker stream …")
        marker_streams = resolve_byprop("type", MARKER_STREAM_TYPE, timeout=self.resolve_timeout)
        if marker_streams:
            self._marker_inlet = StreamInlet(marker_streams[0])
            logger.info("Marker inlet connected: %s", marker_streams[0].name())
        else:
            logger.warning("No Marker stream found - marker ingestion disabled.")

    def _setup_output_stream(self) -> None:
        if not _LSL_AVAILABLE:
            return
        info = StreamInfo(
            name=OUTPUT_STREAM_NAME,
            type=OUTPUT_STREAM_TYPE,
            channel_count=OUTPUT_STREAM_CHANNELS,
            nominal_srate=0,
            channel_format="string",
            source_id="bci_backend_001",
        )
        self._output_outlet = StreamOutlet(info)
        logger.info("Output outlet: '%s' (type='%s')", OUTPUT_STREAM_NAME, OUTPUT_STREAM_TYPE)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time BCI Backend (OpenBCI + Unity LSL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-freq", type=float, default=10.0, metavar="HZ", help="SSVEP stimulus frequency.")
    parser.add_argument("--sfreq", type=float, default=DEFAULT_SFREQ, metavar="HZ", help="EEG sampling rate.")
    parser.add_argument("--epoch-duration", type=float, default=4.0, metavar="S", help="Analysis window length (s).")
    parser.add_argument("--n-train-epochs", type=int, default=10, help="Epochs per object during training.")
    parser.add_argument("--detection-threshold", type=float, default=0.55, help="Minimum ensemble score for SSVEP detection.")
    parser.add_argument("--resolve-timeout", type=float, default=10.0, metavar="S", help="Seconds to wait for each LSL stream.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging verbosity.")
    parser.add_argument("--test-mode", action="store_true", help="Send dummy LSL messages every 2s to test Unity connection.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    backend = BCIBackend(
        target_freq=args.target_freq,
        sfreq=args.sfreq,
        epoch_duration=args.epoch_duration,
        n_train_epochs=args.n_train_epochs,
        detection_threshold=args.detection_threshold,
        resolve_timeout=args.resolve_timeout,
    )

    backend.start()

    if args.test_mode:
        logger.info("Test mode enabled - sending dummy LSL messages every 2 seconds.")

    logger.info("Press Ctrl-C to stop.")

    try:
        while True:
            if args.test_mode:
                backend.send_test_message()
                time.sleep(2.0)
            else:
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Ctrl-C received.")
    finally:
        backend.stop()


if __name__ == "__main__":
    main()
