from dataset import AudioVisualDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset = AudioVisualDataset(
        audio_dir="outputs/features",
        video_dir="outputs/frames"
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, (audio, video) in enumerate(loader):
        print(f"Sample {i + 1}")
        print(f"Audio shape: {audio.shape}")  # (B, n_mels, time)
        print(f"Video shape: {video.shape}")  # (B, num_frames, 3, 224, 224)
        break
