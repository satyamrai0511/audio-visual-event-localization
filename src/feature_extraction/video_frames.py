import cv2
import os

def extract_frames(video_path, output_dir, fps=1, resize=(224, 224)):
    """
    Extract frames from video at fixed fps and save as images.

    Args:
        video_path: path to input video
        output_dir: folder to save extracted frames
        fps: frames per second to extract
        resize: (width, height) tuple for resizing frames
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.resize(frame, resize)
            frame_path = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"Extracted {saved} frames to '{output_dir}'")

if __name__ == "__main__":
    video_path = "data/sample_video_1.mp4"
    output_dir = "outputs/frames/sample_video_1"
    extract_frames(video_path, output_dir, fps=1)
