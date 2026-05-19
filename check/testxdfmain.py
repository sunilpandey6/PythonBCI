import argparse
import logging
import time
from typing import Any, List, Tuple

try:
    import pyxdf
except ImportError:
    pyxdf = None

from pylsl import StreamInfo, StreamOutlet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("xdf_replayer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an XDF file over LSL to simulate live BCI streams.")
    parser.add_argument("file", help="Path to the XDF file (e.g. mixdata2104.xdf).")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (e.g. 2.0 for double speed).")
    args = parser.parse_args()

    if pyxdf is None:
        logger.error("The 'pyxdf' library is not installed. Please run: pip install pyxdf")
        return

    logger.info("Loading XDF file: %s", args.file)
    try:
        streams, header = pyxdf.load_xdf(args.file)
    except Exception as e:
        logger.error("Failed to load XDF file. Error: %s", e)
        return

    eeg_stream = None
    marker_stream = None

    # Locate streams by type
    for s in streams:
        stype = s["info"]["type"][0]
        if stype == "EEG" and eeg_stream is None:
            eeg_stream = s
        elif stype == "Markers" and marker_stream is None:
            marker_stream = s

    if not eeg_stream:
        logger.warning("No stream with type='EEG' found in the XDF.")
    if not marker_stream:
        logger.warning("No stream with type='Markers' found in the XDF.")

    # Create LSL Outlets
    eeg_outlet = None
    if eeg_stream:
        n_channels = int(eeg_stream["info"]["channel_count"][0])
        srate = float(eeg_stream["info"]["nominal_srate"][0])
        info = StreamInfo("OpenBCI_EEG_Replay", "EEG", n_channels, srate, "float32", "xdf_eeg_001")
        eeg_outlet = StreamOutlet(info)
        logger.info("Created EEG LSL Outlet: %d channels @ %.1f Hz", n_channels, srate)

    marker_outlet = None
    if marker_stream:
        info = StreamInfo("Unity_Markers_Replay", "Markers", 1, 0, "string", "xdf_markers_001")
        marker_outlet = StreamOutlet(info)
        logger.info("Created Markers LSL Outlet.")

    # Merge all time-series data into a single chronologically sorted queue
    events: List[Tuple[float, str, Any]] = []

    if eeg_stream:
        ts = eeg_stream["time_stamps"]
        data = eeg_stream["time_series"]
        for t, d in zip(ts, data):
            events.append((t, "EEG", d))

    if marker_stream:
        ts = marker_stream["time_stamps"]
        data = marker_stream["time_series"]
        for t, d in zip(ts, data):
            # marker data is usually a list containing a single string per sample
            events.append((t, "Markers", d[0]))

    if not events:
        logger.error("No data samples found in the selected streams.")
        return

    # Sort events strictly by timestamp
    events.sort(key=lambda x: x[0])

    logger.info("Ready. Starting playback of %d events at %.1fx speed...", len(events), args.speed)
    logger.info("Make sure main.py is running in another terminal!")
    logger.info("Press Ctrl+C to stop playback.")
    
    # Wait until main.py (or another LSL consumer) connects to the outlets
    logger.info("Waiting for main.py to connect to LSL streams...")
    while True:
        eeg_ready = True if eeg_outlet is None else eeg_outlet.have_consumers()
        marker_ready = True if marker_outlet is None else marker_outlet.have_consumers()
        if eeg_ready and marker_ready:
            break
        time.sleep(0.5)
    
    logger.info("Connection detected! Starting playback...")
    start_time_xdf = events[0][0]
    start_time_real = time.time()

    try:
        for ev_time, ev_type, ev_data in events:
            # Calculate when this sample should be played in real-time
            target_time = start_time_real + ((ev_time - start_time_xdf) / args.speed)
            now = time.time()
            
            # Sleep if we are ahead of schedule
            if target_time > now:
                time.sleep(target_time - now)

            # Push the sample
            if ev_type == "EEG" and eeg_outlet:
                eeg_outlet.push_sample(ev_data.tolist())
            elif ev_type == "Markers" and marker_outlet:
                marker_outlet.push_sample([ev_data])
                logger.info("[Pushed Marker] %s", ev_data)

    except KeyboardInterrupt:
        logger.info("Playback interrupted by user.")
    
    logger.info("Playback complete.")

if __name__ == "__main__":
    main()
