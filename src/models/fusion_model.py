import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(AudioEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: (B, F, T) → (B, T, F)
        x = x.permute(0, 2, 1)
        output, _ = self.lstm(x)  # (B, T, 2*hidden_dim)
        return output

class VideoEncoder(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), hidden_dim=128):
        super(VideoEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # (B, 16, 112, 112)
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))  # (B, 32, 1, 1)
        )
        self.fc = nn.Linear(32, hidden_dim)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        feats = self.cnn(frames).view(B, T, -1)  # (B, T, 32)
        return self.fc(feats)  # (B, T, hidden_dim)

class FusionModel(nn.Module):
    def __init__(self, audio_dim=64, hidden_dim=128, num_classes=10):
        super(FusionModel, self).__init__()
        self.audio_encoder = AudioEncoder(audio_dim, hidden_dim)
        self.video_encoder = VideoEncoder(hidden_dim=hidden_dim)
        self.fusion = nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, audio, video):
        """
        audio: (B, F, T_audio)
        video: (B, T_video, 3, H, W)
        """
        a_feat = self.audio_encoder(audio)        # (B, T_audio, 2*H)
        v_feat = self.video_encoder(video)        # (B, T_video, H)
        
        # Simple fusion: take aligned time steps (trim to match shorter)
        T = min(a_feat.shape[1], v_feat.shape[1])
        a_feat = a_feat[:, :T, :]
        v_feat = v_feat[:, :T, :]
        fused = torch.cat([a_feat, v_feat], dim=-1)  # (B, T, 3H)
        
        fused = self.fusion(fused)                  # (B, T, H)
        output, _ = self.decoder(fused)             # (B, T, H)
        logits = self.classifier(output)            # (B, T, num_classes)

        return logits  # raw scores over time
