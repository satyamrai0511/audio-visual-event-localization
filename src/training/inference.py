import torch
import numpy as np
import os

from src.models.fusion_model import FusionModel
from dataset import AudioVisualDataset
from src.feature_extraction.audio_features import extract_log_mel
from src.feature_extraction.video_frames import extract_frames
from src.preprocessing.utils import format_time

from torchvision import transforms
from PIL import Image

def load_video_tensor(frame_dir, resize=(224, 224)):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])
    frames = []
    for f in frame_files:
        img_path = os.path.join(frame_dir, f)
        img = Image.open(img_path).convert("RGB")
        frames.append(transform(img))
    return torch.stack(frames)  # (T, 3, H, W)

@torch.no_grad()
def run_inference(audio_path, video_path, model_path, output_dir="outputs/inference", num_classes=5):
    os.makedirs(output_dir, exist_ok=True)
    sample_id = os.path.splitext(os.path.basename(audio_path))[0]

    print("🔁 Extracting features...")
    log_mel = extract_log_mel(audio_path)  # (n_mels, T)
    audio_tensor = torch.from_numpy(log_mel).unsqueeze(0).float()  # (1, F, T)

    frame_dir = os.path.join(output_dir, sample_id)
    extract_frames(video_path, frame_dir, fps=1)
    video_tensor = load_video_tensor(frame_dir).unsqueeze(0)  # (1, T, 3, H, W)

    print("🧠 Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(audio_dim=64, hidden_dim=128, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("🎯 Running inference...")
    audio_tensor, video_tensor = audio_tensor.to(device), video_tensor.to(device)
    logits = model(audio_tensor, video_tensor)
    preds = torch.argmax(logits, dim=-1).cpu().numpy()[0]

    print("\n📋 Predicted Event Timeline:")
    for t, cls in enumerate(preds):
        print(f"{format_time(t)} → Class {cls}")

    # Optionally save
    np.save(os.path.join(output_dir, f"{sample_id}_preds.npy"), preds)

if __name__ == "__main__":
    run_inference(
        audio_path="data/sample_audio_1.wav",
        video_path="data/sample_video_1.mp4",
        model_path="latest_model.pth"
    )
