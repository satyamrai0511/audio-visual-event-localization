import torch
from fusion_model import FusionModel

if __name__ == "__main__":
    B = 2             # batch size
    F = 64            # audio mel bands
    T_audio = 100     # audio timesteps
    T_video = 90      # video frames
    H, W = 224, 224   # image size

    # Dummy inputs
    audio = torch.randn(B, F, T_audio)
    video = torch.randn(B, T_video, 3, H, W)

    model = FusionModel(audio_dim=F, hidden_dim=128, num_classes=5)
    logits = model(audio, video)  # Output: (B, T, num_classes)

    print(f"Output shape: {logits.shape}")
