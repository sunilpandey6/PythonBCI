from PythonBCI.bci.application import backend
from __future__ import annotations

import argparse
import logging
import time

from bci.domain.config import BCIConfig
from bci.application.backend import BCIBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bci.main")


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

    try:
        time.sleep(30)

        logger.info("Sending test prediction")

        backend._push_raw(
            """
            {
                "Code":300,
                "Event":"Predict_Active_Start",
                "Detail":"Demo",
                "Remark":{
                    "Model":"ACTIVE",
                    "Prediction":"OBJ1",
                    "Confidence":0.95
                }
            }
            """
        )

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Ctrl-C received.")
    finally:
        backend.stop()

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
