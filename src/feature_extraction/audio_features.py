import librosa
import numpy as np
import os

def extract_log_mel(audio_path, sr=16000, n_fft=1024, hop_length=512, n_mels=64):
    """
    Load an audio file and extract log-mel spectrogram.
    
    Returns:
        log_mel: np.ndarray of shape (n_mels, time)
    """
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel

def save_feature(feature, output_path):
    """
    Save a numpy array to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, feature)

if __name__ == "__main__":
    # Example test
    audio_path = "data/sample_audio_1.wav"
    output_path = "outputs/features/sample_audio_1_logmel.npy"
    
    features = extract_log_mel(audio_path)
    save_feature(features, output_path)
    print(f"Saved features to {output_path}, shape: {features.shape}")
