import time
import json
import threading
try:
    from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
except ImportError:
    print("pylsl not installed. Cannot run the connection test.")
    exit(1)

# Stream Settings
MARKER_STREAM_NAME = "UnityMarkers"
MARKER_STREAM_TYPE = "Markers"
OUTPUT_STREAM_TYPE = "BCIResult"

def mock_unity_sender():
    """Simulate Unity sending markers."""
    info = StreamInfo(
        name=MARKER_STREAM_NAME,
        type=MARKER_STREAM_TYPE,
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id="mock_unity"
    )
    outlet = StreamOutlet(info)
    print(f"[Mock Unity] Started sending markers on stream type '{MARKER_STREAM_TYPE}'.")
    
    # Wait for Python backend to initialize
    time.sleep(2)
    
    actions = [
        # Time,Experiment,Phase,Event,Detail,Action
        "12:00:00,Exp,Phase,Event1,Detail1,Flicker_Start",
        "12:00:04,Exp,Phase,Event1,Detail1,Flicker_End",
        
        "12:00:05,Exp,Phase,Event1,Door1,Training_Active_Door1_Start",
        "12:00:09,Exp,Phase,Event1,Door1,Training_Door1_End",
        
        "12:00:10,Exp,Phase,Event2,Door2,Training_Active_Door2_Start",
        "12:00:14,Exp,Phase,Event2,Door2,Training_Door2_End",

        "12:00:15,Exp,Phase,Event_Predict,Door1,predict",
    ]
    
    for action_log in actions:
        print(f"[Mock Unity] -> Sending: {action_log}")
        outlet.push_sample([action_log])
        time.sleep(3)
        
def mock_unity_receiver():
    """Listen to BCIBackend responses."""
    print(f"[Mock Unity] Resolving incoming stream of type '{OUTPUT_STREAM_TYPE}'...")
    streams = resolve_byprop("type", OUTPUT_STREAM_TYPE, timeout=10.0)
    if not streams:
        print(f"[Mock Unity] Could not resolve stream of type '{OUTPUT_STREAM_TYPE}'.")
        return
        
    inlet = StreamInlet(streams[0])
    print(f"[Mock Unity] Connected to BCIBackend. Listening for JSON results...")
    
    while True:
        sample, ts = inlet.pull_sample(timeout=1.0)
        if sample:
            print(f"[Mock Unity] <- Received from BCI: {sample[0]}")

if __name__ == "__main__":
    t_recv = threading.Thread(target=mock_unity_receiver, daemon=True)
    t_recv.start()
    
    mock_unity_sender()
    
    # Wait a bit longer to catch the last message
    time.sleep(2)
    print("[Mock Unity] Test complete.")
