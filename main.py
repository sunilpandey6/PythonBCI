from __future__ import annotations

import argparse
import logging
import queue
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from flicker import FlickerResult, SSVEPDetector
from prediction import ActiveClassifier, ImageryClassifier, MixedClassifier
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

EEG_STREAM_TYPE: str = "EEG"
MARKER_STREAM_TYPE: str = "Markers"
OUTPUT_STREAM_NAME: str = "BCIBackend"
OUTPUT_STREAM_TYPE: str = "BCIResult"
OUTPUT_STREAM_CHANNELS: int = 1
N_CHANNELS: int = 16
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


class BCIState:
    IDLE = "IDLE"
    SSVEP_TEST = "SSVEP_TEST"
    TRAIN_ACTIVE_OBJ1 = "TRAIN_ACTIVE_OBJ1"
    TRAIN_ACTIVE_OBJ2 = "TRAIN_ACTIVE_OBJ2"
    TRAIN_IMAGERY_OBJ1 = "TRAIN_IMAGERY_OBJ1"
    TRAIN_IMAGERY_OBJ2 = "TRAIN_IMAGERY_OBJ2"
    PREDICT_ACTIVE = "PREDICT_ACTIVE"
    PREDICT_IMAGERY = "PREDICT_IMAGERY"
    PREDICT_MIXED = "PREDICT_MIXED"


class BCIBackend:
    def __init__(self, config: BCIConfig = BCIConfig()) -> None:
        self.target_freq = float(config.target_freq)
        self.sfreq = float(config.sfreq)
        self.epoch_duration = float(config.epoch_duration)
        self.eeg_stream_name = config.eeg_stream_name
        self.marker_stream_name = config.marker_stream_name
        self.n_train_epochs = int(config.n_train_epochs)
        self.detection_threshold = float(config.detection_threshold)
        self.resolve_timeout = float(config.resolve_timeout)
        self.predict_accumulation_time = float(config.predict_accumulation_time)
        self.predict_agreement_threshold = float(config.predict_agreement_threshold)
        self.predict_confidence_threshold = float(config.predict_confidence_threshold)

        self.step_size = 0.2
        self.latency_shift_s = 0.5
        self._last_process_time = 0.0
        self._ignore_samples = 0

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
        self.mixed_model = MixedClassifier(sfreq=self.sfreq, n_channels=N_CHANNELS)

        self.active_obj1_epochs: Deque[np.ndarray] = deque(maxlen=self.n_train_epochs)
        self.active_obj2_epochs: Deque[np.ndarray] = deque(maxlen=self.n_train_epochs)
        self.imagery_obj1_epochs: Deque[np.ndarray] = deque(maxlen=self.n_train_epochs)
        self.imagery_obj2_epochs: Deque[np.ndarray] = deque(maxlen=self.n_train_epochs)
        self._flicker_results: List[FlickerResult] = []
        
        self._predict_buffer: Deque[Tuple[int, float, str, float]] = deque(
            maxlen=max(1, int(self.predict_accumulation_time / self.step_size))
        )

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
                if self._ignore_samples > 0:
                    self._ignore_samples -= 1
                    continue

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

                # Bundle all three fields into one atomic payload.
                # Do NOT touch _last_unity_event/_last_unity_detail here —
                # Thread 3 owns those updates to avoid the Flicker_End race.
                payload = {"action": action_str, "event": event_str, "detail": detail_str}
                self._marker_queue.put_nowait(payload)

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
                    payload = self._marker_queue.get_nowait()
                except queue.Empty:
                    break
                self._handle_marker_payload(payload)

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
                BCIState.PREDICT_MIXED,
            ):
                if not self._eeg_ready.wait(timeout=1.0):
                    continue
                self._eeg_ready.clear()

                epoch = self._snapshot_epoch()
                if epoch is None:
                    continue

                now = time.time()
                if now - self._last_process_time < self.step_size:
                    continue
                self._last_process_time = now

                self._process_epoch(epoch, current_state)

        logger.info("[Thread-Logic] stopped.")

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _handle_marker_payload(self, payload: dict) -> None:
        action = payload["action"]
        logger.info("[Logic] Action received: '%s'", action)

        if "Set_Target_Frequency" in action:
            match = re.search(r'[\d\.]+', action)
            if match:
                try:
                    freq = float(match.group())
                    self.target_freq = freq
                    self._detector.target_freq = freq
                    logger.info("[Logic] Target frequency set to %.2f Hz", freq)
                except ValueError:
                    self.target_freq = 15.0
                    self._detector.target_freq = 15.0
                    logger.warning("[Logic] Invalid frequency in '%s', defaulting to 15.0 Hz", action)
            else:
                self.target_freq = 15.0
                self._detector.target_freq = 15.0
                logger.info("[Logic] No frequency found in '%s', defaulting to 15.0 Hz", action)
            return

        transitions = {
            "Flicker_Start": BCIState.SSVEP_TEST,
            "Flicker_End": BCIState.IDLE,
            "Training_Active_Door1_Start": BCIState.TRAIN_ACTIVE_OBJ1,
            "Training_Active_Door1_End": BCIState.IDLE,
            "Training_Imagery_Door1_Start": BCIState.TRAIN_IMAGERY_OBJ1,
            "Training_Imagery_Door1_End": BCIState.IDLE,
            "Active_Training_Door2_Start": BCIState.TRAIN_ACTIVE_OBJ2,
            "Active_Training_Door2_End": BCIState.IDLE,
            "Image_Training_Door2_Start": BCIState.TRAIN_IMAGERY_OBJ2,
            "Image_Training_Door2_End": BCIState.IDLE,
            "Predict_Start_Active": BCIState.PREDICT_ACTIVE,
            "Predict_Start_Imagery": BCIState.PREDICT_IMAGERY,
            "Start_predict": BCIState.PREDICT_MIXED,
            "Predict_End": BCIState.IDLE,
        }

        if action in transitions:
            new_state = transitions[action]
            # Condition A: Flicker_End while in SSVEP_TEST.
            # Do NOT overwrite _last_unity_event/_last_unity_detail so that
            # _finalize_flicker() can echo back the correct Flicker_Start metadata.
            if self._get_state() == BCIState.SSVEP_TEST and new_state == BCIState.IDLE:
                self._finalize_flicker()
            else:
                # Condition B: All other transitions — safe to update shared metadata.
                with self._unity_event_lock:
                    self._last_unity_event = payload["event"]
                    self._last_unity_detail = payload["detail"]
            self._set_state(new_state)
            return

        if action == "Train_End":
            self._attempt_training()
            self._set_state(BCIState.IDLE)
            return

        logger.debug("[Logic] Unrecognised action: '%s' - ignored.", action)

    def _preprocess_global(self, epoch: np.ndarray) -> np.ndarray:
        """Notch (50 Hz) -> Bandpass (1.0-90.0 Hz)."""
        from scipy import signal
        nyq = self.sfreq / 2.0
        b_n, a_n = signal.iirnotch(50.0 / nyq, 30.0)
        epoch_filt = signal.filtfilt(b_n, a_n, epoch, axis=-1)
        
        sos = signal.butter(4, [1.0, min(90.0, nyq - 0.1)], btype="bandpass", fs=self.sfreq, output="sos")
        epoch_filt = signal.sosfiltfilt(sos, epoch_filt, axis=-1)
        
        return epoch_filt

    def _process_epoch(self, epoch: np.ndarray, state: str) -> None:
        epoch = self._preprocess_global(epoch)
        unity_event, unity_detail = self._read_unity_event()

        if state == BCIState.SSVEP_TEST:
            result = self._detector.detect(epoch)
            self._flicker_results.append(result)

        elif state == BCIState.TRAIN_ACTIVE_OBJ1:
            self.active_obj1_epochs.append(epoch.copy())
            n = len(self.active_obj1_epochs)
            logger.info("[Logic] TRAIN_ACTIVE_OBJ1 sliding window: %d/%d epochs buffered.", n, self.n_train_epochs)
            if n >= self.n_train_epochs:
                logger.info("[Logic] TRAIN_ACTIVE_OBJ1 buffer full - sliding window ready.")

        elif state == BCIState.TRAIN_ACTIVE_OBJ2:
            self.active_obj2_epochs.append(epoch.copy())
            n = len(self.active_obj2_epochs)
            logger.info("[Logic] TRAIN_ACTIVE_OBJ2 sliding window: %d/%d epochs buffered.", n, self.n_train_epochs)
            if n >= self.n_train_epochs:
                logger.info("[Logic] TRAIN_ACTIVE_OBJ2 buffer full - sliding window ready.")

        elif state == BCIState.TRAIN_IMAGERY_OBJ1:
            self.imagery_obj1_epochs.append(epoch.copy())
            n = len(self.imagery_obj1_epochs)
            logger.info("[Logic] TRAIN_IMAGERY_OBJ1 sliding window: %d/%d epochs buffered.", n, self.n_train_epochs)
            if n >= self.n_train_epochs:
                logger.info("[Logic] TRAIN_IMAGERY_OBJ1 buffer full - sliding window ready.")

        elif state == BCIState.TRAIN_IMAGERY_OBJ2:
            self.imagery_obj2_epochs.append(epoch.copy())
            n = len(self.imagery_obj2_epochs)
            logger.info("[Logic] TRAIN_IMAGERY_OBJ2 sliding window: %d/%d epochs buffered.", n, self.n_train_epochs)
            if n >= self.n_train_epochs:
                logger.info("[Logic] TRAIN_IMAGERY_OBJ2 buffer full - sliding window ready.")

        elif state == BCIState.PREDICT_ACTIVE:
            if self.active_model.is_trained:
                pred, conf = self.active_model.predict(epoch)
                self._accumulate_and_push(pred, conf, "ACTIVE", "None", 0.0, unity_event, unity_detail)
            else:
                logger.warning("[Logic] ACTIVE model not trained.")

        elif state == BCIState.PREDICT_IMAGERY:
            if self.imagery_model.is_trained:
                pred, conf = self.imagery_model.predict(epoch)
                self._accumulate_and_push(pred, conf, "IMAGERY", "None", 0.0, unity_event, unity_detail)
            else:
                logger.warning("[Logic] IMAGERY model not trained.")

        elif state == BCIState.PREDICT_MIXED:
            if self.mixed_model.is_trained:
                pred_mixed, conf_mixed = self.mixed_model.predict(epoch)
                
                # Get imagery predict if available
                pred_imagery_str = "None"
                conf_imagery = 0.0
                if self.imagery_model.is_trained:
                    pred_im, conf_im = self.imagery_model.predict(epoch)
                    pred_imagery_str = "OBJ1" if pred_im == 0 else "OBJ2"
                    conf_imagery = conf_im
                
                self._accumulate_and_push(pred_mixed, conf_mixed, "MIXED", pred_imagery_str, conf_imagery, unity_event, unity_detail)
            else:
                logger.warning("[Logic] MIXED model not trained.")

    def _accumulate_and_push(self, pred: int, conf: float, model_name: str, imagery_str: str, imagery_conf: float, unity_event: str, unity_detail: str) -> None:
        self._predict_buffer.append((pred, conf, imagery_str, imagery_conf))
        if len(self._predict_buffer) == self._predict_buffer.maxlen:
            counts = Counter([p[0] for p in self._predict_buffer])
            most_common_pred, most_common_count = counts.most_common(1)[0]
            agreement = most_common_count / len(self._predict_buffer)
            avg_conf = sum([p[1] for p in self._predict_buffer]) / len(self._predict_buffer)
            
            if agreement >= self.predict_agreement_threshold and avg_conf >= self.predict_confidence_threshold:
                last_imagery_str = self._predict_buffer[-1][2]
                avg_imagery_conf = sum([p[3] for p in self._predict_buffer]) / len(self._predict_buffer)
                
                code = BCICode.ACTIVE_OBJ1_PREDICT if most_common_pred == 0 else BCICode.ACTIVE_OBJ2_PREDICT
                if model_name == "IMAGERY":
                    code = BCICode.IMAGERY_OBJ1_PREDICT if most_common_pred == 0 else BCICode.IMAGERY_OBJ2_PREDICT

                self._push_message(code, unity_event, unity_detail,
                                   remark={
                                       "Model": model_name,
                                       "Prediction": "OBJ1" if most_common_pred == 0 else "OBJ2",
                                       "Confidence": avg_conf,
                                       "Agreement": agreement,
                                       "Imagery_Prediction": last_imagery_str,
                                       "Imagery_Confidence": avg_imagery_conf
                                   })
                logger.info("[Logic] Stable prediction emitted: %s (Agreement: %.2f, AvgConf: %.2f)", code.name, agreement, avg_conf)
                self._predict_buffer.clear()

    def _attempt_training(self) -> None:
        act_obj1 = list(self.active_obj1_epochs)
        act_obj2 = list(self.active_obj2_epochs)
        img_obj1 = list(self.imagery_obj1_epochs)
        img_obj2 = list(self.imagery_obj2_epochs)

        if len(act_obj1) >= self.n_train_epochs and len(act_obj2) >= self.n_train_epochs:
            X_a = act_obj1 + act_obj2
            y_a = [0] * len(act_obj1) + [1] * len(act_obj2)
            self.active_model.train(X_a, y_a)
        else:
            logger.warning("[Logic] ACTIVE training skipped - insufficient epochs (OBJ1: %d, OBJ2: %d, need: %d).",
                           len(act_obj1), len(act_obj2), self.n_train_epochs)

        if len(img_obj1) >= self.n_train_epochs and len(img_obj2) >= self.n_train_epochs:
            X_i = img_obj1 + img_obj2
            y_i = [0] * len(img_obj1) + [1] * len(img_obj2)
            self.imagery_model.train(X_i, y_i)
        else:
            logger.warning("[Logic] IMAGERY training skipped - insufficient epochs (OBJ1: %d, OBJ2: %d, need: %d).",
                           len(img_obj1), len(img_obj2), self.n_train_epochs)

        X_m_obj1 = act_obj1 + img_obj1
        X_m_obj2 = act_obj2 + img_obj2
        if len(X_m_obj1) >= self.n_train_epochs and len(X_m_obj2) >= self.n_train_epochs:
            X_m = X_m_obj1 + X_m_obj2
            y_m = [0] * len(X_m_obj1) + [1] * len(X_m_obj2)
            self.mixed_model.train(X_m, y_m)
        else:
            logger.warning("[Logic] MIXED training skipped - insufficient epochs (OBJ1: %d, OBJ2: %d, need: %d).",
                           len(X_m_obj1), len(X_m_obj2), self.n_train_epochs)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def _finalize_flicker(self) -> None:
        unity_event, unity_detail = self._read_unity_event()
        if not self._flicker_results:
            logger.warning("[Logic] No SSVEP epochs were processed during flicker.")
            from flicker import FlickerResult
            from protocol import BCICode
            result = FlickerResult(
                code=BCICode.FLICKER_NOT_DETECTED,
                detected_frequency=self.target_freq,
                confidence_score=0.0,
                ssvep_present=False,
                fbcca_score=0.0
            )
            self._push_flicker_result(result, unity_event, unity_detail)
            return

        max_conf = float(max(r.confidence_score for r in self._flicker_results))
        ssvep_present = max_conf >= self._detector.detection_threshold
        
        from flicker import FlickerResult
        from protocol import BCICode
        final_result = FlickerResult(
            code=BCICode.FLICKER_DETECTED if ssvep_present else BCICode.FLICKER_NOT_DETECTED,
            detected_frequency=self.target_freq,
            confidence_score=max_conf,
            ssvep_present=ssvep_present,
            fbcca_score=float(max(r.fbcca_score for r in self._flicker_results))
        )
        self._push_flicker_result(final_result, unity_event, unity_detail)
        logger.info("[Logic] Flicker finalized. Max confidence: %.2f (Windows: %d)", max_conf, len(self._flicker_results))

    def _set_state(self, new_state: str) -> None:
        with self._state_lock:
            logger.info("[Logic] State: %s → %s", self._state, new_state)
            self._state = new_state
            
        if new_state == BCIState.SSVEP_TEST:
            with self._buffer_lock:
                self._flicker_results.clear()

        # When transitioning to a new processing state (except SSVEP which is continuous),
        # flush the buffer and ignore the next 140ms of samples to account for visual processing latency.
        if new_state in (
            BCIState.TRAIN_ACTIVE_OBJ1,
            BCIState.TRAIN_ACTIVE_OBJ2,
            BCIState.TRAIN_IMAGERY_OBJ1,
            BCIState.TRAIN_IMAGERY_OBJ2,
            BCIState.PREDICT_ACTIVE,
            BCIState.PREDICT_IMAGERY,
            BCIState.PREDICT_MIXED,
        ):
            with self._buffer_lock:
                self._eeg_buffer.clear()
                self._ignore_samples = int(self.latency_shift_s * self.sfreq)
                self._predict_buffer.clear()

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
    defaults = BCIConfig()
    parser = argparse.ArgumentParser(
        description="Real-Time BCI Backend (OpenBCI + Unity LSL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-freq",           type=float, default=defaults.target_freq,              metavar="HZ", help="SSVEP stimulus frequency.")
    parser.add_argument("--sfreq",                 type=float, default=defaults.sfreq,                   metavar="HZ", help="EEG sampling rate.")
    parser.add_argument("--epoch-duration",        type=float, default=defaults.epoch_duration,           metavar="S",  help="Analysis window length (s).")
    parser.add_argument("--n-train-epochs",        type=int,   default=defaults.n_train_epochs,                        help="Ring buffer size per class during training.")
    parser.add_argument("--detection-threshold",   type=float, default=defaults.detection_threshold,                   help="Minimum FBCCA score for FLICKER_DETECTED.")
    parser.add_argument("--resolve-timeout",       type=float, default=defaults.resolve_timeout,          metavar="S",  help="Seconds to wait for each LSL stream.")
    parser.add_argument("--log-level",             choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                                                   default="INFO",                                                      help="Logging verbosity.")
    parser.add_argument("--test-mode",             action="store_true",                                                help="Send dummy LSL messages every 2s to test Unity connection.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    config = BCIConfig(
        target_freq=args.target_freq,
        sfreq=args.sfreq,
        epoch_duration=args.epoch_duration,
        n_train_epochs=args.n_train_epochs,
        detection_threshold=args.detection_threshold,
        resolve_timeout=args.resolve_timeout,
    )

    backend = BCIBackend(config)
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
