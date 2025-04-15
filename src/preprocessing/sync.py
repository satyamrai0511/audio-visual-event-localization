import librosa
import cv2

def get_audio_duration(audio_path):
    """Return duration of audio in seconds."""
    duration = librosa.get_duration(path=audio_path)
    return duration

def get_video_duration(video_path):
    """Return duration of video in seconds using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return duration

def sync_durations(video_path, audio_path):
    """Compare durations and return the minimum valid time."""
    video_dur = get_video_duration(video_path)
    audio_dur = get_audio_duration(audio_path)
    sync_duration = min(video_dur, audio_dur)

    print(f"Video Duration: {video_dur:.2f}s")
    print(f"Audio Duration: {audio_dur:.2f}s")
    print(f"Synced Duration: {sync_duration:.2f}s")

    return sync_duration

if __name__ == "__main__":
    video_path = "data/sample_video_1.mp4"
    audio_path = "data/sample_audio_1.wav"
    sync_durations(video_path, audio_path)
