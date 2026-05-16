import time
import sys

try:
    from pylsl import resolve_byprop, StreamInlet
except ImportError as e:
    print("Error: pylsl not installed. Please run `pip install pylsl`.")
    sys.exit(1)

def main():
    print("Searching for the BCIResult LSL stream (from main.py)...")
    
    # Resolve an LSL stream that has the type "BCIResult"
    streams = resolve_byprop("type", "BCIResult")
    
    if not streams:
        print("No stream found! Make sure main.py is running and pushing results.")
        sys.exit(1)
        
    inlet = StreamInlet(streams[0])
    print(f"Connected to stream: {streams[0].name()}")
    print("Listening for JSON results from PythonBCI...")
    print("-" * 50)

    try:
        while True:
            # Receive sample from the stream
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            
            if sample is not None:
                # The sample is a list containing the JSON string
                json_str = sample[0]
                print(f"[Result Received at {timestamp:.3f}]")
                print(json_str)
                print("-" * 50)
            else:
                # Just keep waiting
                pass
                
    except KeyboardInterrupt:
        print("\nStopped listening.")

if __name__ == '__main__':
    main()
