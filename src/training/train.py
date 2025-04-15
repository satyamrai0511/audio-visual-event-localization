import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import AudioVisualDataset
from src.models.fusion_model import FusionModel

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for i, (audio, video) in enumerate(loader):
        audio, video = audio.to(device), video.to(device)

        logits = model(audio, video)  # (B, T, C)
        targets = torch.randint(0, logits.shape[-1], (logits.shape[0], logits.shape[1])).to(device)  # Dummy targets

        loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(loader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AudioVisualDataset(
        audio_dir="outputs/features",
        video_dir="outputs/frames"
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = FusionModel(audio_dim=64, hidden_dim=128, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Starting training...")
    avg_loss = train(model, loader, criterion, optimizer, device)
    print(f"Average Training Loss: {avg_loss:.4f}")
