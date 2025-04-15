import os
from datetime import datetime

def format_time(seconds):
    """Convert seconds (float) to HH:MM:SS.mmm format."""
    millis = int((seconds - int(seconds)) * 1000)
    time_str = datetime.utcfromtimestamp(int(seconds)).strftime('%H:%M:%S')
    return f"{time_str}.{millis:03d}"

def make_dirs(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def log(message):
    """Log message with timestamp."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{now}] {message}")

if __name__ == "__main__":
    print(format_time(3661.789))   # Example output: 01:01:01.789
    make_dirs("outputs/test")      # Will create folder if it doesn’t exist
    log("This is a test log entry.")
