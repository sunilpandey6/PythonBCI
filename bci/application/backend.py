from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import List, Optional

import numpy as np

from bci.domain.config import BCIConfig
from bci.domain.codes import BCICode
from bci.domain.state_machine import BCIStateMachine, BCIState
from bci.domain.buffers import EEGBuffer, EpochBuffer
from bci.signal.preprocessing import preprocess_global
from bci.signal.ssvep import SSVEPDetector, FlickerResult
from bci.ml.classifiers import ActiveClassifier, ImageryClassifier, MixedClassifier
from bci.ml.voting import PredictionAccumulator
from bci.infrastructure.lsl_io import LslManager
from bci.infrastructure.protocol import build_message

logger = logging.getLogger("bci.main")

N_CHANNELS: int = 16


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
        self.latency_shift_s = 0.75
        self._last_process_time = 0.0
        self._ignore_samples = 0

        self._epoch_samples: int = int(self.sfreq * self.epoch_duration)

        self._shutdown: threading.Event = threading.Event()
        self._eeg_ready: threading.Event = threading.Event()
        self._marker_queue: queue.Queue[dict] = queue.Queue()

        self._eeg_buffer = EEGBuffer(max_samples=self._epoch_samples)
        self._buffer_lock: threading.Lock = threading.Lock()

        self._state_machine = BCIStateMachine()
        self._state_lock: threading.Lock = threading.Lock()

        self._detector = SSVEPDetector(
            target_freq=self.target_freq,
            sfreq=self.sfreq,
            detection_threshold=self.detection_threshold,
        )
        self.active_model = ImageryClassifier(sfreq=self.sfreq, n_channels=N_CHANNELS)
        self.imagery_model = ImageryClassifier(sfreq=self.sfreq, n_channels=N_CHANNELS)
        self.mixed_model = MixedClassifier(sfreq=self.sfreq, n_channels=N_CHANNELS)

        self.active_obj1_epochs = EpochBuffer(max_epochs=self.n_train_epochs)
        self.active_obj2_epochs = EpochBuffer(max_epochs=self.n_train_epochs)
        self.imagery_obj1_epochs = EpochBuffer(max_epochs=self.n_train_epochs)
        self.imagery_obj2_epochs = EpochBuffer(max_epochs=self.n_train_epochs)
        self._flicker_results: List[FlickerResult] = []
        
        predict_buffer_len = max(1, int(self.predict_accumulation_time / self.step_size))
        self._predict_accumulator = PredictionAccumulator(
            maxlen=predict_buffer_len,
            agreement_threshold=self.predict_agreement_threshold,
            confidence_threshold=self.predict_confidence_threshold
        )

        self._last_unity_event: str = ""
        self._last_unity_detail: str = ""
        self._unity_event_lock: threading.Lock = threading.Lock()

        self.is_eye_closed: bool = False

        self._lsl_manager = LslManager(resolve_timeout=self.resolve_timeout)
        self._threads: List[threading.Thread] = []

    def start(self) -> None:
        logger.info("BCIBackend starting …")
        self._lsl_manager.resolve_streams()
        self._lsl_manager.setup_output_stream(
            name="BCIBackend",
            stream_type="BCIResult",
            channel_count=1,
            source_id="bci_backend_001",
        )

        t1 = threading.Thread(target=self._eeg_ingestion_loop, name="Thread-EEG", daemon=True)
        t2 = threading.Thread(target=self._marker_ingestion_loop, name="Thread-Marker", daemon=True)
        t3 = threading.Thread(target=self._logic_loop, name="Thread-Logic", daemon=True)
        self._threads = [t1, t2, t3]
        for t in self._threads:
            t.start()

        logger.info(
            "All threads started. State=%s | Target=%.1f Hz | Epoch=%.1f s",
            self._get_state(), self.target_freq, self.epoch_duration,
        )

    def stop(self) -> None:
        logger.info("Shutdown requested.")
        self._shutdown.set()
        for t in self._threads:
            t.join(timeout=5.0)
        logger.info("BCIBackend stopped.")

    def _eeg_ingestion_loop(self) -> None:
        logger.info("[Thread-EEG] started.")
        from bci.infrastructure.lsl_io import LSL_AVAILABLE
        while not self._shutdown.is_set():
            if not LSL_AVAILABLE or self._lsl_manager._eeg_inlet is None:
                sample = np.random.randn(N_CHANNELS).astype(np.float32)
                time.sleep(1.0 / self.sfreq)
            else:
                sample_list = self._lsl_manager.pull_eeg_sample(timeout=1.0)
                if sample_list is None:
                    continue
                sample = np.array(sample_list, dtype=np.float32)

            with self._buffer_lock:
                if self._ignore_samples > 0:
                    self._ignore_samples -= 1
                    continue

                self._eeg_buffer.append(sample)
                if self._eeg_buffer.is_ready():
                    self._eeg_ready.set()

        logger.info("[Thread-EEG] stopped.")

    def _marker_ingestion_loop(self) -> None:
        logger.info("[Thread-Marker] started.")
        from bci.infrastructure.lsl_io import LSL_AVAILABLE
        while not self._shutdown.is_set():
            if not LSL_AVAILABLE or self._lsl_manager._marker_inlet is None:
                time.sleep(0.05)
                continue

            sample_list = self._lsl_manager.pull_marker_sample(timeout=0.0)
            if sample_list is not None and len(sample_list) > 0:
                raw: str = str(sample_list[0])
                logger.debug("[Thread-Marker] received: %s", raw)

                parts = raw.split(",")
                if len(parts) >= 6:
                    event_str = parts[3].strip()
                    detail_str = parts[4].strip()
                    action_str = parts[5].strip()
                else:
                    event_str = raw
                    detail_str = ""
                    action_str = raw

                payload = {"action": action_str, "event": event_str, "detail": detail_str}
                self._marker_queue.put_nowait(payload)

            time.sleep(0.001)

        logger.info("[Thread-Marker] stopped.")

    def _logic_loop(self) -> None:
        logger.info("[Thread-Logic] started. State: %s", self._get_state())

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

    def _handle_marker_payload(self, payload: dict) -> None:
        action = payload["action"]
        logger.info("[Logic] Action received: '%s'", action)

        if action == "Eye_Closed":
            self.is_eye_closed = True
            logger.info("[Logic] Eye closed. Pausing processing (preprocessor will return None).")
            return

        if action == "Eye_Opened":
            self.is_eye_closed = False
            logger.info("[Logic] Eye opened. Resuming processing.")
            return

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

        new_state = self._state_machine.get_next_state(action)

        if new_state is not None:
            if self._get_state() == BCIState.SSVEP_TEST and new_state == BCIState.IDLE:
                self._finalize_flicker()
            else:
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

    def _process_epoch(self, epoch: np.ndarray, state: str) -> None:
        processed = preprocess_global(epoch, self.sfreq, is_eye_closed=self.is_eye_closed, current_state=state)
        if processed is None:
            return
        epoch = processed
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
        self._predict_accumulator.append(pred, conf, imagery_str, imagery_conf)
        stable = self._predict_accumulator.get_stable_prediction(model_name)
        if stable is not None:
            agreement = stable["agreement"]
            avg_conf = stable["confidence"]
            most_common_pred = stable["prediction"]
            last_imagery_str = stable["imagery_prediction"]
            avg_imagery_conf = stable["imagery_confidence"]

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
            self._predict_accumulator.clear()

    def _attempt_training(self) -> None:
        act_obj1 = self.active_obj1_epochs.get_all()
        act_obj2 = self.active_obj2_epochs.get_all()
        img_obj1 = self.imagery_obj1_epochs.get_all()
        img_obj2 = self.imagery_obj2_epochs.get_all()

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

    def _finalize_flicker(self) -> None:
        unity_event, unity_detail = self._read_unity_event()
        if not self._flicker_results:
            logger.warning("[Logic] No SSVEP epochs were processed during flicker.")
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
        ssvep_present = max_conf >= self.detection_threshold
        
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
            logger.info("[Logic] State: %s → %s", self._state_machine.state, new_state)
            self._state_machine.set_state(new_state)
            
        if new_state == BCIState.SSVEP_TEST:
            with self._buffer_lock:
                self._flicker_results.clear()

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
                self._predict_accumulator.clear()

    def _get_state(self) -> str:
        with self._state_lock:
            return self._state_machine.state

    def _snapshot_epoch(self) -> Optional[np.ndarray]:
        with self._buffer_lock:
            return self._eeg_buffer.snapshot()

    def _push_flicker_result(self, result: FlickerResult, unity_event: str = "", unity_detail: str = "") -> None:
        msg = build_message(code=result.code, event=unity_event, detail=unity_detail, remark=result.to_dict())
        self._push_raw(msg)
        logger.info("[Output] %s", msg)

    def _push_message(self, code: BCICode, unity_event: str = "", unity_detail: str = "", remark: object = "") -> None:
        msg = build_message(code=code, event=unity_event, detail=unity_detail, remark=remark)
        self._push_raw(msg)
        logger.info("[Output] %s", msg)

    def _push_raw(self, json_str: str) -> None:
        self._lsl_manager.push_sample([json_str])

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
