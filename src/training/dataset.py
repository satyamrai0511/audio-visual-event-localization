import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class AudioVisualDataset(Dataset):
    def __init__(self, audio_dir, video_dir, label_dir=None, transform=None):
        """
        Args:
            audio_dir (str): Directory with .npy audio features
            video_dir (str): Directory with frame folders per sample
            label_dir (str): Directory with .npy label files (optional)
            transform: Optional torchvision transforms for images
        """
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.sample_names = self._get_sample_names()

    def _get_sample_names(self):
        audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith(".npy")]
        sample_names = [os.path.splitext(f)[0].replace("_logmel", "") for f in audio_files]
        return sample_names

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample = self.sample_names[idx]
        audio_path = os.path.join(self.audio_dir, f"{sample}_logmel.npy")
        frame_folder = os.path.join(self.video_dir, sample)

        # Load audio
        audio = np.load(audio_path)
        audio_tensor = torch.from_numpy(audio).float()

        # Load video frames
        frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".jpg")])
        frames = []
        for frame_file in frame_files:
            img_path = os.path.join(frame_folder, frame_file)
            image = Image.open(img_path).convert("RGB")
            frames.append(self.transform(image))
        video_tensor = torch.stack(frames)

        # Load labels if provided
        if self.label_dir:
            label_path = os.path.join(self.label_dir, f"{sample}_labels.npy")
            labels = np.load(label_path)
            label_tensor = torch.from_numpy(labels).long()
            return audio_tensor, video_tensor, label_tensor

        return audio_tensor, video_tensor
