import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, average_precision_score
import numpy as np

from dataset import AudioVisualDataset
from src.models.fusion_model import FusionModel

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []

    for audio, video, labels in loader:
        audio, video, labels = audio.to(device), video.to(device), labels.to(device)

        logits = model(audio, video)  # (B, T, C)
        T = min(logits.shape[1], labels.shape[1])
        logits = logits[:, :T, :]
        labels = labels[:, :T]

        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.cpu().numpy().reshape(-1))
        all_labels.append(labels.cpu().numpy().reshape(-1))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    print("📈 Per-Class mAP:")
    y_true = np.eye(num_classes)[all_labels]  # One-hot
    y_pred = np.eye(num_classes)[all_preds]  # One-hot predicted
    for i in range(num_classes):
        ap = average_precision_score(y_true[:, i], y_pred[:, i])
        print(f"Class {i}: AP = {ap:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AudioVisualDataset(
        audio_dir="outputs/features",
        video_dir="outputs/frames",
        label_dir="outputs/labels"
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = FusionModel(audio_dim=64, hidden_dim=128, num_classes=5).to(device)
    model.load_state_dict(torch.load("latest_model.pth", map_location=device))  # replace with your checkpoint
    evaluate(model, loader, device, num_classes=5)
